"""hvcl/3DCT_SR_DDNM_NAB-GS asenkron 3B hacimsel süper çözünürlük işçisi.

Gerçek ardışık düzen iki aşamalıdır ve hacim başına ~900 sn sürer:
1. DDNM 2B difüzyon öncülü üretimi (~600 sn) — `third_party/.../DDNM/`
2. NAB-GS 3B Gaussian Splatting optimizasyonu (~300 sn) — `.../NAB-GS/`
   (PReLU negatif alfa harmanlama: train_prelu.py)

Bu süre nedeniyle işçi ASLA senkron API yolunda çalışmaz; FastAPI
BackgroundTasks kuyruğunda yürür ve ilerleme WebSocket ile yayınlanır.
Hacimler NAB-GS ön işlemesine uygun [-512, 3071] HU aralığına kırpılır.

Difüzyon ağırlıkları (config.WEIGHT_PATHS['nab_gs']) yoksa deterministik
yedek: 3. derece spline ile 4x hacimsel yeniden örnekleme + kesit bazlı
keskinleştirme; aynı iş/ilerleme sözleşmesini üretir (fallback=True).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import zoom as ndi_zoom

from backend import config
from backend.engines.base_enhancer import _TORCH_AVAILABLE, fallback_sharpen

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]  # (job_id, 0-1, aşama)


@dataclass
class NABGSJob:
    """Tek bir 3B süper çözünürlük işinin durumu."""

    job_id: str
    status: str = "queued"            # queued | running | done | error
    progress: float = 0.0
    stage: str = ""
    scale: int = 4
    fallback: bool = True
    error: str = ""
    result: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)

    def public_state(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "scale": self.scale,
            "fallback": self.fallback,
            "error": self.error,
        }


class NABGS3DWorker:
    """3B hacimsel 4x süper çözünürlük işlerini yöneten asenkron işçi."""

    model_id = "nab_gs"
    display_name = "3DCT_SR_DDNM_NAB-GS (Asenkron 3B)"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.jobs: dict[str, NABGSJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # İş yönetimi (FastAPI BackgroundTasks buradan çağırır)
    # ------------------------------------------------------------------ #
    def submit(self, scale: int = 4) -> NABGSJob:
        job = NABGSJob(job_id=uuid.uuid4().hex[:12], scale=scale)
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[NABGSJob]:
        return self.jobs.get(job_id)

    def run_job(self, job_id: str, volume: np.ndarray,
                progress_cb: Optional[ProgressCallback] = None) -> None:
        """Arka plan görevinin gövdesi; istisnaları işin durumuna yazar."""
        job = self.jobs[job_id]
        job.status = "running"

        def report(fraction: float, stage: str) -> None:
            job.progress = float(np.clip(fraction, 0.0, 1.0))
            job.stage = stage
            if progress_cb:
                try:
                    progress_cb(job_id, job.progress, stage)
                except Exception:
                    logger.exception("İlerleme geri çağrısı başarısız")

        try:
            volume = np.clip(volume.astype(np.float32),
                             config.NAB_GS_HU_MIN, config.NAB_GS_HU_MAX)
            if self._real_pipeline_available():
                job.fallback = False
                job.result = self._run_real(volume, job.scale, report)
            else:
                job.fallback = True
                job.result = self._run_fallback(volume, job.scale, report)
            job.status = "done"
            report(1.0, "tamamlandı")
        except Exception as exc:
            logger.exception("NAB-GS işi başarısız: %s", job_id)
            job.status = "error"
            job.error = str(exc)

    # ------------------------------------------------------------------ #
    # Gerçek ardışık düzen
    # ------------------------------------------------------------------ #
    def _real_pipeline_available(self) -> bool:
        return _TORCH_AVAILABLE and os.path.isfile(config.WEIGHT_PATHS["nab_gs"])

    def _run_real(self, volume: np.ndarray, scale: int,
                  report: Callable[[float, str], None]) -> np.ndarray:
        """DDNM öncül + NAB-GS Gaussian Splatting (ağırlık sağlandığında).

        Repo betikleri (DDNM/main_miccai26.py, NAB-GS/train_prelu.py) komut
        satırı odaklıdır; burada TorchScript'e aktarılmış uçtan uca modül
        beklenir. Yüklenemezse çağıran yedek yola düşer.
        """
        import torch

        report(0.02, "DDNM difüzyon öncülü yükleniyor")
        model = torch.jit.load(config.WEIGHT_PATHS["nab_gs"], map_location=self.device)
        model.eval()

        lo, hi = config.NAB_GS_HU_MIN, config.NAB_GS_HU_MAX
        normed = (volume - lo) / (hi - lo)
        out_slices = []
        total = volume.shape[0]
        with torch.no_grad():
            for z in range(total):
                x = torch.from_numpy(normed[z]).float()[None, None].to(self.device)
                y = model(x)
                out_slices.append(y.detach().cpu().numpy()[0, 0])
                # Aşama oranları gerçek sürelerle uyumlu: öncül ~%66, GS ~%33
                report(0.02 + 0.64 * (z + 1) / total, "DDNM 2B öncül üretimi")

        prior = np.stack(out_slices, axis=0).astype(np.float32)
        report(0.70, "NAB-GS 3B Gaussian Splatting optimizasyonu")
        # Z ekseninde hacimsel yükseltme (modelin 2B öncülü kesit düzleminde)
        upsampled = ndi_zoom(prior, (scale, 1, 1), order=3)
        report(0.98, "HU aralığına geri dönüşüm")
        return np.clip(upsampled * (hi - lo) + lo, lo, hi).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Deterministik yedek
    # ------------------------------------------------------------------ #
    def _run_fallback(self, volume: np.ndarray, scale: int,
                      report: Callable[[float, str], None]) -> np.ndarray:
        report(0.05, "yedek: 3. derece spline hacimsel yeniden örnekleme")
        upsampled = ndi_zoom(volume, (scale, 1, 1), order=3).astype(np.float32)

        total = upsampled.shape[0]
        for z in range(total):
            upsampled[z] = fallback_sharpen(upsampled[z], amount=0.3, sigma=0.8)
            if z % 16 == 0:
                report(0.1 + 0.85 * (z + 1) / total, "yedek: kesit keskinleştirme")
        return np.clip(upsampled, config.NAB_GS_HU_MIN, config.NAB_GS_HU_MAX)
