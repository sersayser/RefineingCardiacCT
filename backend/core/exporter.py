"""İyileştirilmiş fazı diske aktarma (orijinal vaka klasörü içine).

Seçili fazın TÜM kesitleri, seçili motor + o anki harmanlama parametreleriyle
(denklem 2.1) işlenir ve orijinal vaka klasörünün içinde yeni bir klasöre
yazılır: `IYILESTIRILMIS_<faz>_<model>_<zaman>/`.

Çıktı biçimi:
- Kaynak DICOM ise: kesit başına DICOM — orijinal başlıklar korunur, piksel
  verisi iyileştirilmiş HU ile değiştirilir, yeni SeriesInstanceUID ve
  "... IYILESTIRILMIS" seri açıklaması atanır, sıkıştırmasız
  ExplicitVRLittleEndian yazılır (her PACS görüntüleyicide açılır).
- Kaynak NIfTI/HDF5 ise: tek `.nii.gz` hacmi.
Ek olarak izlenebilirlik için `parametreler.json` yazılır.

Uzun sürdüğünden (256 kesit ~20-60 sn) FastAPI BackgroundTasks içinde koşar;
ilerleme WebSocket üzerinden yayınlanır (type: "export_progress").
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from backend import config
from backend.config import BlendParams
from backend.core.ingestion import IngestedCase, PhaseVolume, _sanitize_name
from backend.core.residual_blender import ResidualBlender

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]  # (job_id, 0-1, aşama)


@dataclass
class ExportJob:
    job_id: str
    status: str = "queued"      # queued | running | done | error
    progress: float = 0.0
    stage: str = ""
    output_dir: str = ""
    file_count: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def public_state(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "output_dir": self.output_dir,
            "file_count": self.file_count,
            "error": self.error,
        }


def export_root_for(case: IngestedCase) -> str:
    """Çıktı kök klasörü.

    - Sunucu yoluyla yüklenen vaka: orijinal klasörün kendisi (dosyaysa üstü).
    - Tarayıcıdan yüklenen vaka (uploads/ altındaki kopya): kullanıcının
      orijinal klasörü bilinemez; görünür `disa_aktarimlar/<vaka>/` kullanılır.
    """
    src = os.path.abspath(case.source_path)
    uploads_root = os.path.abspath(config.UPLOADS_DIR)
    if src == uploads_root or src.startswith(uploads_root + os.sep):
        return os.path.join(config.EXPORTS_DIR, _sanitize_name(case.case_id))
    return src if os.path.isdir(src) else os.path.dirname(src)


def _params_slug(params: BlendParams) -> str:
    """Parametreleri klasör adına yazar: her ayar kombinasyonu ayrı klasör."""
    return ("a{alpha:.2f}_g{gamma:.2f}_b{beta:.2f}"
            "_tv{lambda_tv:.2f}_k{kappa_hu:.2f}").format(**params.as_dict())


class PhaseExporter:
    """Dışa aktarma işlerini yöneten kayıt defteri + yürütücü."""

    def __init__(self) -> None:
        self.jobs: dict[str, ExportJob] = {}
        self._lock = threading.Lock()

    def submit(self) -> ExportJob:
        job = ExportJob(job_id=uuid.uuid4().hex[:12])
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[ExportJob]:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------------ #
    def run_job(
        self,
        job_id: str,
        case: IngestedCase,
        phase: PhaseVolume,
        engine,                      # BaseCTEnhancer
        params: BlendParams,
        contrast_engine=None,        # ContrastGANEnhancer (kappa_hu > 0 ise)
        progress_cb: Optional[ProgressCallback] = None,
    ) -> None:
        job = self.jobs[job_id]
        job.status = "running"

        def report(fraction: float, stage: str) -> None:
            job.progress = float(np.clip(fraction, 0.0, 1.0))
            job.stage = stage
            if progress_cb:
                try:
                    progress_cb(job_id, job.progress, stage)
                except Exception:
                    logger.exception("Dışa aktarma ilerleme geri çağrısı başarısız")

        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            dir_name = "IYILESTIRILMIS_{}_{}_{}_{}".format(
                _sanitize_name(phase.name), _sanitize_name(engine.model_id),
                _params_slug(params), stamp)
            out_dir = os.path.join(export_root_for(case), dir_name)
            os.makedirs(out_dir, exist_ok=True)
            job.output_dir = out_dir

            source_is_dicom = bool(phase.source_paths) and all(
                p.lower().endswith((".dcm", ".dicom", ".ima")) or "." not in os.path.basename(p)
                for p in phase.source_paths
            ) and len(phase.source_paths) == phase.num_slices

            enhanced_slices: list[np.ndarray] = [] if not source_is_dicom else None
            total = phase.num_slices
            fallback_any = False

            for z in range(total):
                i_orig = phase.get_slice(z)
                result = engine.enhance_slice(i_orig)
                fallback_any = fallback_any or result.fallback

                delta_contrast = None
                if params.kappa_hu > 0.0 and contrast_engine is not None:
                    delta_contrast = contrast_engine.contrast_delta(i_orig)

                cache = ResidualBlender.build_cache(
                    i_orig, result.i_dl, delta_contrast=delta_contrast,
                    model_id=result.model_id, fallback=result.fallback,
                )
                i_out = ResidualBlender.blend(cache, params)

                if source_is_dicom:
                    _write_enhanced_dicom(
                        phase.source_paths[z], i_out, out_dir, z,
                        series_uid_cache=job.job_id, model_id=engine.model_id,
                    )
                else:
                    enhanced_slices.append(i_out)

                report((z + 1) / total * 0.95,
                       f"kesit {z + 1}/{total} işleniyor ({engine.model_id})")

            if not source_is_dicom:
                _write_enhanced_nifti(enhanced_slices, phase, out_dir)

            # İzlenebilirlik: hangi model + parametrelerle üretildi
            with open(os.path.join(out_dir, "parametreler.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({
                    "vaka": case.case_id,
                    "faz": phase.name,
                    "model": engine.model_id,
                    "fallback_kullanildi": fallback_any,
                    "parametreler": params.as_dict(),
                    "tarih": stamp,
                    "kesit_sayisi": total,
                }, fh, ensure_ascii=False, indent=2)

            job.file_count = len(os.listdir(out_dir))
            job.status = "done"
            report(1.0, "tamamlandı")
        except Exception as exc:
            logger.exception("Dışa aktarma başarısız: %s", job_id)
            job.status = "error"
            job.error = str(exc)


# ---------------------------------------------------------------------- #
# Biçim yazıcıları
# ---------------------------------------------------------------------- #
_SERIES_UID_CACHE: dict[str, str] = {}


def _write_enhanced_dicom(source_path: str, i_out: np.ndarray, out_dir: str,
                          index: int, series_uid_cache: str, model_id: str) -> None:
    """Orijinal DICOM başlıklarını koruyarak iyileştirilmiş kesiti yazar.

    Pikseller sıkıştırmasız, işaretsiz 16 bit olarak normalize edilir:
    RescaleSlope=1, RescaleIntercept=-1024 -> pixel = HU + 1024 (0..4095).
    """
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = pydicom.dcmread(source_path, stop_before_pixels=True)

    if series_uid_cache not in _SERIES_UID_CACHE:
        _SERIES_UID_CACHE[series_uid_cache] = generate_uid()
    ds.SeriesInstanceUID = _SERIES_UID_CACHE[series_uid_cache]
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    desc = str(getattr(ds, "SeriesDescription", "") or "").strip()
    ds.SeriesDescription = (desc + " IYILESTIRILMIS " + model_id.upper())[:64]
    ds.SeriesNumber = int(getattr(ds, "SeriesNumber", 0) or 0) + 1000

    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # işaretsiz
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = int(i_out.shape[0]), int(i_out.shape[1])

    pixels = np.clip(i_out + 1024.0, 0.0, 4095.0).round().astype(np.uint16)
    ds.PixelData = pixels.tobytes()
    ds["PixelData"].VR = "OW"  # ExplicitVR yazımı belirsiz 'OB or OW' kabul etmez
    # Kayıpsız yeniden kodlama ile tutarsız kalabilecek etiketleri temizle
    for tag in ("LossyImageCompression", "LossyImageCompressionRatio",
                "LossyImageCompressionMethod"):
        if hasattr(ds, tag):
            delattr(ds, tag)

    ds.save_as(os.path.join(out_dir, f"iyilestirilmis_{index:04d}.dcm"),
               enforce_file_format=True)


def _write_enhanced_nifti(slices: list[np.ndarray], phase: PhaseVolume,
                          out_dir: str) -> None:
    import nibabel as nib

    volume = np.stack(slices, axis=0).astype(np.float32)  # (Z, Y, X)
    sz, sy, sx = phase.spacing
    affine = np.diag([sx, sy, sz, 1.0])
    # Dahili (Z, Y, X) -> NIfTI (X, Y, Z)
    nii = nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), affine=affine)
    nib.save(nii, os.path.join(out_dir, "iyilestirilmis.nii.gz"))
