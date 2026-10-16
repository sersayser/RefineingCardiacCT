"""FastAPI REST uç noktaları ve WebSocket sunucusu.

Çalıştırma: `.venv/bin/uvicorn backend.api.main:app --reload --port 8000`

Mimari kurallar (CLAUDE.md):
- Kesit bazlı motorlar senkron uç noktalarda; NAB-GS 3B süper çözünürlük
  daima BackgroundTasks kuyruğunda, ilerleme /api/v1/ws/jobs WebSocket'inde.
- /api/v1/enhance-slice, I_orig ve ΔI (+ harmanlama bileşenleri) float32
  matrislerini Base64 tamponu olarak döndürür; 60 FPS kaydırıcı işlemleri
  ön yüzde bu önbellekle yapılır, model yeniden çalıştırılmaz.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend import config
from backend.config import BlendParams
from backend.core.exporter import PhaseExporter
from backend.core.ingestion import CaseIngestor, IngestedCase
from backend.core.phase_selector import PerceptualScorer, PhaseSelector
from backend.core.residual_blender import ResidualBlender
from backend.engines import build_engine, build_fast_pool, SLICE_ENGINE_IDS
from backend.engines.auto_best import select_best_enhancement
from backend.engines.contrast_gan_wrapper import ContrastGANEnhancer
from backend.engines.nab_gs_3d import NABGS3DWorker
from backend.engines.noise2sim_wrapper import Noise2SimAdapter
from backend.engines.temporal_mar_wrapper import TemporalMAREnhancer
from backend.learning.meta_optimizer import MetaOptimizer, case_to_feature_vector
from backend.learning.models_db import CaseRecord, Preset, create_engine_and_session

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    meta_optimizer.maybe_retrain(_SessionFactory())
    yield


app = FastAPI(title="CT Clinical Enhancer", version="1.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------- #
# Uygulama durumu
# ---------------------------------------------------------------------- #
DB_PATH = os.environ.get("CT_ENHANCER_DB", config.DATABASE_PATH)

ingestor = CaseIngestor()
phase_selector = PhaseSelector(use_pyiqa=True)
blender = ResidualBlender()
nab_gs_worker = NABGS3DWorker()
phase_exporter = PhaseExporter()
temporal_mar = TemporalMAREnhancer()
noise2sim_adapter = Noise2SimAdapter()
meta_optimizer = MetaOptimizer()

_engine_cache: dict[str, object] = {}
_cases: dict[str, IngestedCase] = {}
_db_engine, _SessionFactory = create_engine_and_session(DB_PATH)
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def get_engine(model_id: str):
    if model_id not in _engine_cache:
        _engine_cache[model_id] = build_engine(model_id)
    return _engine_cache[model_id]


def get_case(case_id: str) -> IngestedCase:
    case = _cases.get(case_id)
    if case is None:
        raise HTTPException(404, f"Vaka bellekte yok (önce /ingest): {case_id}")
    return case


# ---------------------------------------------------------------------- #
# WebSocket iş ilerleme yayını
# ---------------------------------------------------------------------- #
class ConnectionManager:
    def __init__(self) -> None:
        self.connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self.connections:
                self.connections.remove(ws)

    async def broadcast(self, message: dict) -> None:
        with self._lock:
            targets = list(self.connections)
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)

    def broadcast_from_thread(self, message: dict) -> None:
        """Arka plan iş parçacığından (BackgroundTasks) güvenli yayın."""
        if _event_loop is not None and not _event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), _event_loop)


ws_manager = ConnectionManager()


@app.websocket("/api/v1/ws/jobs")
async def ws_jobs(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # ping/pong; istemci mesajı beklenmez
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------- #
# Yardımcılar
# ---------------------------------------------------------------------- #
def _b64_f32(arr: np.ndarray) -> str:
    """float32 little-endian ham tamponun Base64 kodu."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode()


def _phase_or_404(case: IngestedCase, phase: Optional[str]):
    if phase is None or phase == "":
        phase = next(iter(case.phases.keys()))
    if phase not in case.phases:
        raise HTTPException(404, f"Faz yok: {phase}; mevcut: {list(case.phases)}")
    return case.phases[phase]


def _case_payload(case: IngestedCase) -> dict:
    """ingest / ingest-upload ortak yanıtı (tembel fazlarda piksel çözmez)."""
    return {
        "case_id": case.case_id,
        "is_multiphase": case.is_multiphase,
        "metadata": case.metadata,
        "phases": {
            name: {
                "num_slices": pv.num_slices,
                "shape": list(pv.shape),
                "spacing": list(pv.spacing),
                "metadata": pv.metadata,
            }
            for name, pv in case.phases.items()
        },
    }


# ---------------------------------------------------------------------- #
# İstek gövdeleri
# ---------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    path: str = Field(..., description="Dosya/klasör/arşiv yolu (.dcm .nii .nii.gz .hdf5 .zip .rar)")


class RecommendPhaseRequest(BaseModel):
    case_id: str


class EnhanceSliceRequest(BaseModel):
    case_id: str
    phase: Optional[str] = None
    slice_index: int = 0
    model_id: str = "rdn"           # veya "auto_best"
    params: Optional[dict] = None   # alpha, gamma, beta, lambda_tv, kappa_hu
    include_contrast_delta: bool = True


class PresetRequest(BaseModel):
    name: str
    model_type: str = "rdn"
    alpha: float = config.DEFAULT_PARAMS["alpha"]
    gamma: float = config.DEFAULT_PARAMS["gamma"]
    lambda_tv: float = config.DEFAULT_PARAMS["lambda_tv"]
    beta: float = config.DEFAULT_PARAMS["beta"]
    kappa_hu: float = config.DEFAULT_PARAMS["kappa_hu"]
    window_center: float = 50.0
    window_width: float = 400.0


class BatchAnalyzeRequest(BaseModel):
    paths: list[str]


class BatchCommitCase(BaseModel):
    case_id: str
    folder_path: str = ""
    selected_phase: str = ""
    phase_score: float = 0.0
    kvp: float = 0.0
    mas: float = 0.0
    slice_thickness: float = 0.0
    kernel_code: float = 0.0
    sigma_hu_before: float = 0.0
    sigma_hu_after: float = 0.0
    gmm_peak_hu: float = 0.0
    niqe_before: float = 0.0
    niqe_after: float = 0.0
    ssim_inter: float = 0.0
    approved_model: str = "rdn"
    alpha: float = config.DEFAULT_PARAMS["alpha"]
    gamma: float = config.DEFAULT_PARAMS["gamma"]
    lambda_tv: float = config.DEFAULT_PARAMS["lambda_tv"]
    beta: float = config.DEFAULT_PARAMS["beta"]
    kappa_hu: float = config.DEFAULT_PARAMS["kappa_hu"]


class BatchCommitRequest(BaseModel):
    cases: list[BatchCommitCase]
    run_noise2sim_adaptation: bool = True


class PredictParamsRequest(BaseModel):
    case_id: Optional[str] = None
    features: Optional[list[float]] = None  # doğrudan X_i (8 boyut)


class SR3DSubmitRequest(BaseModel):
    case_id: str
    phase: Optional[str] = None
    scale: int = 4


class ExportPhaseRequest(BaseModel):
    case_id: str
    phase: Optional[str] = None
    model_id: str = "rdn"           # somut model; "auto_best" kabul edilmez
    params: Optional[dict] = None   # alpha, gamma, beta, lambda_tv, kappa_hu


# ---------------------------------------------------------------------- #
# Uç noktalar
# ---------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Kök adres: etkileşimli API dokümantasyonuna yönlendir."""
    return RedirectResponse("/docs")


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": SLICE_ENGINE_IDS,
        "cases_in_memory": list(_cases.keys()),
        "meta_optimizer": meta_optimizer.status(),
    }


@app.post("/api/v1/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        case = ingestor.ingest(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    _cases[case.case_id] = case
    return _case_payload(case)


@app.post("/api/v1/ingest-upload")
async def ingest_upload(request: Request) -> dict:
    """Tarayıcıdan sürükle-bırak / klasör seçimiyle gelen dosyaları alır.

    Tarayıcılar güvenlik gereği mutlak dosya yolu veremez; bu uç nokta dosya
    İÇERİĞİNİ multipart olarak alır, uploads/ altına göreli yollarıyla yazar
    ve klasörü normal ingest akışından geçirir.

    Büyük PACS vakaları (binlerce .dcm) için PARÇALI yükleme desteklenir:
    istemci aynı `upload_id` ile dosyaları paketler halinde gönderir; yalnız
    `finalize=true` taşıyan son pakette klasör ingest edilir. Form, Starlette
    varsayılan 1000 dosya sınırının üzerinde bir tavanla ayrıştırılır.
    """
    form = await request.form(max_files=50_000, max_fields=50_100)
    files = [f for f in form.getlist("files") if isinstance(f, StarletteUploadFile)]
    case_name = str(form.get("case_name") or "")
    upload_id = re.sub(r"[^\w-]+", "", str(form.get("upload_id") or ""))[:32]
    finalize = str(form.get("finalize") or "true").strip().lower() not in (
        "false", "0", "hayir", "no")
    if not files:
        raise HTTPException(422, "Dosya gelmedi")

    safe = re.sub(r"[^\w.-]+", "_", case_name).strip("_")[:64]
    if not upload_id:  # tek paketlik yükleme: her çağrı yeni klasör
        upload_id = uuid.uuid4().hex[:8]
    target = os.path.join(config.UPLOADS_DIR, f"{safe or 'vaka'}_{upload_id}")

    saved = 0
    for uf in files:
        rel = (uf.filename or "").replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        if not parts:
            continue
        dest = os.path.join(target, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(uf.file, fh)
        saved += 1
    if saved == 0:
        raise HTTPException(422, "Kaydedilebilir dosya yok")

    if not finalize:  # ara paket: yalnız kaydet, ingest son pakette
        return {"upload_id": upload_id, "saved": saved, "finalized": False}

    try:
        case = ingestor.ingest(target)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    if safe:
        case.case_id = safe
    _cases[case.case_id] = case
    payload = _case_payload(case)
    payload["uploaded_files"] = saved
    payload["upload_id"] = upload_id
    payload["server_path"] = target
    payload["finalized"] = True
    return payload


@app.post("/api/v1/recommend-phase")
def recommend_phase(req: RecommendPhaseRequest) -> dict:
    case = get_case(req.case_id)
    results = phase_selector.analyze_case(case)
    recommended = next((m.phase_name for m in results if m.recommended), None)
    return {
        "case_id": case.case_id,
        "recommended_phase": recommended,
        "phases": [m.as_dict() for m in results],
    }


@app.post("/api/v1/enhance-slice")
def enhance_slice(req: EnhanceSliceRequest) -> dict:
    case = get_case(req.case_id)
    phase = _phase_or_404(case, req.phase)
    if not (0 <= req.slice_index < phase.num_slices):
        raise HTTPException(422, f"Kesit indeksi aralık dışı: {req.slice_index}")
    i_orig = phase.get_slice(req.slice_index)

    candidates_report = None
    if req.model_id == "auto_best":
        best, all_candidates = select_best_enhancement(
            i_orig, build_fast_pool(), scorer=phase_selector.scorer
        )
        result = best.result
        candidates_report = [c.public_state() for c in all_candidates]
    else:
        if req.model_id not in SLICE_ENGINE_IDS:
            raise HTTPException(422,
                                f"Geçersiz model: {req.model_id}; "
                                f"geçerli: {SLICE_ENGINE_IDS + ['auto_best']}")
        result = get_engine(req.model_id).enhance_slice(i_orig)

    delta_contrast = None
    if req.include_contrast_delta:
        contrast_engine = get_engine("contrast_gan")
        assert isinstance(contrast_engine, ContrastGANEnhancer)
        delta_contrast = contrast_engine.contrast_delta(i_orig)

    cache = blender.build_cache(i_orig, result.i_dl,
                                delta_contrast=delta_contrast,
                                model_id=result.model_id,
                                fallback=result.fallback)

    params = BlendParams(**(req.params or {}))
    i_out = blender.blend(cache, params)

    h, w = i_orig.shape
    return {
        "case_id": case.case_id,
        "phase": phase.name,
        "slice_index": req.slice_index,
        "model_id": result.model_id,
        "fallback": result.fallback,
        "runtime_s": round(result.runtime_s, 4),
        "sigma_before": round(result.sigma_before, 2),
        "sigma_after": round(result.sigma_after, 2),
        "shape": [h, w],
        "dtype": "float32-le",
        "params": params.as_dict(),
        "auto_best_candidates": candidates_report,
        # 60 FPS harmanlama önbelleği (Base64 ham float32 tamponları)
        "buffers": {name: _b64_f32(arr)
                    for name, arr in cache.payload_arrays().items()},
        # HU sadakati skalerleri: ön yüz α·(m⁺+γ·m⁻) sabitini geri çıkarır
        "bias": {"phi_pos": cache.phi_bias_pos, "phi_neg": cache.phi_bias_neg},
        # Sunucu tarafı doğrulama örneği (ön yüz kendi hesabını karşılaştırabilir)
        "i_out_mean_hu": float(np.mean(i_out)),
        "diff_mean_hu": float(np.mean(np.abs(i_out - i_orig))),
    }


# ---------------------------------------------------------------------- #
# Ön ayarlar
# ---------------------------------------------------------------------- #
@app.get("/api/v1/presets")
def list_presets() -> dict:
    session = _SessionFactory()
    try:
        presets = session.query(Preset).order_by(Preset.created_at.desc()).all()
        return {"presets": [p.as_dict() for p in presets]}
    finally:
        session.close()


@app.post("/api/v1/presets")
def create_preset(req: PresetRequest) -> dict:
    session = _SessionFactory()
    try:
        existing = session.query(Preset).filter_by(name=req.name).one_or_none()
        if existing is None:
            existing = Preset(name=req.name)
            session.add(existing)
        for field_name in ("model_type", "alpha", "gamma", "lambda_tv", "beta",
                           "kappa_hu", "window_center", "window_width"):
            setattr(existing, field_name, getattr(req, field_name))
        session.commit()
        return existing.as_dict()
    finally:
        session.close()


@app.delete("/api/v1/presets/{preset_id}")
def delete_preset(preset_id: int) -> dict:
    session = _SessionFactory()
    try:
        preset = session.get(Preset, preset_id)
        if preset is None:
            raise HTTPException(404, f"Ön ayar yok: {preset_id}")
        session.delete(preset)
        session.commit()
        return {"deleted": preset_id}
    finally:
        session.close()


# ---------------------------------------------------------------------- #
# Toplu vaka analizi ve sürekli öğrenme
# ---------------------------------------------------------------------- #
@app.post("/api/v1/batch-analyze")
def batch_analyze(req: BatchAnalyzeRequest) -> dict:
    rows = []
    for path in req.paths:
        row: dict = {"path": path}
        try:
            case = ingestor.ingest(path)
            _cases[case.case_id] = case
            results = phase_selector.analyze_case(case)
            best = max(results, key=lambda m: m.s_faz)
            row.update({
                "case_id": case.case_id,
                "metadata": case.metadata,
                "recommended_phase": best.phase_name,
                "phase_score": round(best.s_faz, 2),
                "sigma_hu": round(best.sigma_hu, 2),
                "mu_peak_hu": round(best.mu_peak_hu, 1),
                "niqe": round(best.niqe, 3),
                "ssim_inter": round(best.ssim_inter, 4),
                "phases": [m.as_dict() for m in results],
            })
            # Öğrenilmiş parametre önerisi (model eğitildiyse)
            if meta_optimizer.is_trained:
                features = [
                    case.metadata.get("kvp", 0.0),
                    case.metadata.get("mas", 0.0),
                    case.metadata.get("slice_thickness", 0.0),
                    float(case.metadata.get("kernel_code", 0)),
                    best.sigma_hu, best.mu_peak_hu, best.niqe, best.ssim_inter,
                ]
                row["suggested_params"] = meta_optimizer.predict(features)
            else:
                row["suggested_params"] = {"model_id": "rdn", **config.DEFAULT_PARAMS}
            row["status"] = "ok"
        except Exception as exc:
            logger.exception("Toplu analiz hatası: %s", path)
            row.update({"status": "error", "error": str(exc)})
        rows.append(row)
    return {"rows": rows}


def _run_noise2sim_adaptation(case_ids: list[str]) -> None:
    """Arka plan görevi: onaylı vakaların hacimleriyle öz-denetimli ince ayar."""
    volumes = []
    for cid in case_ids:
        case = _cases.get(cid)
        if case is None:
            continue
        for pv in case.phases.values():
            volumes.append(pv.volume)
    if not volumes:
        logger.info("Noise2Sim adaptasyonu: bellekte hacim yok, atlandı")
        return

    def progress(fraction: float, loss: float) -> None:
        ws_manager.broadcast_from_thread({
            "type": "noise2sim_adaptation",
            "progress": round(fraction, 3),
            "loss": round(loss, 6),
        })

    path = noise2sim_adapter.fine_tune(volumes, progress_cb=progress)
    ws_manager.broadcast_from_thread({
        "type": "noise2sim_adaptation",
        "progress": 1.0,
        "checkpoint": path or "atlandı (torch yok / veri yetersiz)",
    })


@app.post("/api/v1/batch-commit-and-learn")
def batch_commit_and_learn(req: BatchCommitRequest,
                           background_tasks: BackgroundTasks) -> dict:
    session = _SessionFactory()
    try:
        for item in req.cases:
            session.add(CaseRecord(**item.model_dump()))
        session.commit()
        retrained = meta_optimizer.maybe_retrain(session)
        total = session.query(CaseRecord).count()
    finally:
        session.close()

    if req.run_noise2sim_adaptation:
        background_tasks.add_task(_run_noise2sim_adaptation,
                                  [c.case_id for c in req.cases])

    return {
        "committed": len(req.cases),
        "total_cases": total,
        "meta_optimizer_retrained": retrained,
        "meta_optimizer": meta_optimizer.status(),
        "noise2sim_adaptation_queued": req.run_noise2sim_adaptation,
    }


@app.post("/api/v1/predict-learned-params")
def predict_learned_params(req: PredictParamsRequest) -> dict:
    if req.features is not None:
        features = req.features
    elif req.case_id is not None:
        case = get_case(req.case_id)
        results = phase_selector.analyze_case(case)
        best = max(results, key=lambda m: m.s_faz)
        features = [
            case.metadata.get("kvp", 0.0),
            case.metadata.get("mas", 0.0),
            case.metadata.get("slice_thickness", 0.0),
            float(case.metadata.get("kernel_code", 0)),
            best.sigma_hu, best.mu_peak_hu, best.niqe, best.ssim_inter,
        ]
    else:
        raise HTTPException(422, "case_id veya features (8 boyut) gerekli")

    try:
        prediction = meta_optimizer.predict(features)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"features": features, "prediction": prediction}


# ---------------------------------------------------------------------- #
# Asenkron 3B süper çözünürlük (NAB-GS)
# ---------------------------------------------------------------------- #
@app.post("/api/v1/sr3d/submit")
def sr3d_submit(req: SR3DSubmitRequest, background_tasks: BackgroundTasks) -> dict:
    case = get_case(req.case_id)
    phase = _phase_or_404(case, req.phase)
    job = nab_gs_worker.submit(scale=req.scale)

    def progress(job_id: str, fraction: float, stage: str) -> None:
        ws_manager.broadcast_from_thread({
            "type": "sr3d_progress",
            "job_id": job_id,
            "progress": round(fraction, 4),
            "stage": stage,
        })

    def run() -> None:
        # Tembel hacim istek yolunda değil, arka plan görevinde çözülür
        nab_gs_worker.run_job(job.job_id, phase.volume, progress)

    background_tasks.add_task(run)
    return job.public_state()


@app.get("/api/v1/sr3d/status/{job_id}")
def sr3d_status(job_id: str) -> dict:
    job = nab_gs_worker.get(job_id)
    if job is None:
        raise HTTPException(404, f"İş yok: {job_id}")
    return job.public_state()


# ---------------------------------------------------------------------- #
# İyileştirilmiş fazı dışa aktarma
# ---------------------------------------------------------------------- #
@app.post("/api/v1/export-phase")
def export_phase(req: ExportPhaseRequest, background_tasks: BackgroundTasks) -> dict:
    """Seçili fazın tüm kesitlerini o anki ayarlarla işleyip orijinal vaka
    klasörü içine yeni bir klasör olarak yazar (arka plan işi)."""
    case = get_case(req.case_id)
    phase = _phase_or_404(case, req.phase)
    if req.model_id not in SLICE_ENGINE_IDS:
        raise HTTPException(422,
                            f"Geçersiz model: {req.model_id}; dışa aktarma somut "
                            f"bir model ister (auto_best sonucu çözümlenmiş olmalı)")
    engine = get_engine(req.model_id)
    params = BlendParams(**(req.params or {}))
    contrast_engine = get_engine("contrast_gan") if params.kappa_hu > 0.0 else None

    job = phase_exporter.submit()

    def progress(job_id: str, fraction: float, stage: str) -> None:
        state = phase_exporter.get(job_id)
        ws_manager.broadcast_from_thread({
            "type": "export_progress",
            "job_id": job_id,
            "progress": round(fraction, 4),
            "stage": stage,
            "output_dir": state.output_dir if state else "",
            "status": state.status if state else "running",
        })

    background_tasks.add_task(
        phase_exporter.run_job, job.job_id, case, phase, engine, params,
        contrast_engine, progress,
    )
    return job.public_state()


@app.get("/api/v1/export/status/{job_id}")
def export_status(job_id: str) -> dict:
    job = phase_exporter.get(job_id)
    if job is None:
        raise HTTPException(404, f"İş yok: {job_id}")
    return job.public_state()
