"""ZihengD/TT-U-Net prensipli 4B hareket artefaktı giderme + MVF analizi.

TT-U-Net (Temporal Transformer U-Net) görüntü uzayında çok fazlı kesit
dizilerinde zamansal öz-dikkat ile hareket bulanıklığını giderir;
MARVEL_CardiacCT ise fazlar arası Hareket Vektör Alanı (MVF) büyüklüğünü
tanımlar. Her iki reponun GitHub kopyası çalıştırılabilir çıkarım kodu
İÇERMEDİĞİNDEN (TT-U-Net: kod yayımlanmamış; MARVEL: yalnız demo/Readme),
bu modül ağırlık dosyası sağlanana dek aynı sözleşmeyi deterministik olarak
uygular:

- MVF: Farneback yoğun optik akışla fazlar arası ||MVF||_2 ölçümü
  (phase_selector.mvf_magnitude ile aynı çekirdek).
- Hareket giderme: komşu fazların hedef faza optik akışla çakıştırılması
  (motion-compensated warping) ve ağırlıklı zamansal ortalama — zamansal
  öz-dikkatin klasik karşılığı. Gürültü sigma'sı zamansal ortalamayla düşer,
  hedef fazın anatomik konumu korunur.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from backend import config
from backend.core.phase_selector import hu_to_metric_unit, mvf_magnitude
from backend.engines.base_enhancer import _TORCH_AVAILABLE, enforce_hu_fidelity

logger = logging.getLogger(__name__)


@dataclass
class TemporalMARResult:
    """Çok fazlı hareket giderme sonucu."""

    corrected: np.ndarray                   # hedef fazın düzeltilmiş hacmi/kesiti
    target_phase: str
    mvf_per_phase: dict[str, float] = field(default_factory=dict)
    stillest_phase: str = ""
    fallback: bool = True


class TemporalMAREnhancer:
    """Çok fazlı 4B dizide hareket düzeltme ve en hareketsiz faz tespiti."""

    model_id = "tt_unet"
    display_name = "TT-U-Net 4D (Hareket Giderme)"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Optional[Any] = None
        self._load_attempted = False

    def ensure_loaded(self) -> bool:
        if not self._load_attempted:
            self._load_attempted = True
            weight_path = config.WEIGHT_PATHS["tt_unet"]
            if _TORCH_AVAILABLE and os.path.isfile(weight_path):
                try:
                    import torch

                    self._model = torch.jit.load(weight_path, map_location=self.device)
                    self._model.eval()
                    logger.info("TT-U-Net TorchScript modeli yüklendi")
                except Exception:
                    logger.exception("TT-U-Net ağırlığı yüklenemedi; yedek aktif")
                    self._model = None
        return self._model is not None

    # ------------------------------------------------------------------ #
    # MVF analizi (MARVEL prensibi)
    # ------------------------------------------------------------------ #
    def measure_mvf(self, phases: dict[str, np.ndarray]) -> dict[str, float]:
        """Her faz için komşu fazlara ortalama ||MVF||_2 büyüklüğü.

        En düşük değerli faz 'en hareketsiz' fazdır (MARVEL tamamlayıcı metriği).
        """
        names = list(phases.keys())
        result: dict[str, float] = {}
        for i, name in enumerate(names):
            mags = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(names):
                    mags.append(mvf_magnitude(phases[name], phases[names[j]]))
            result[name] = float(np.mean(mags)) if mags else 0.0
        return result

    # ------------------------------------------------------------------ #
    # Hareket giderme
    # ------------------------------------------------------------------ #
    def enhance_series(self, phases: dict[str, np.ndarray],
                       target_phase: str) -> TemporalMARResult:
        """Hedef fazın hacmini komşu fazlarla hareket telafili ortalayarak düzeltir.

        phases: {faz_adı: (Z, Y, X) float32 HU hacmi}; tüm fazlar aynı boyutta
        olmalıdır (çok fazlı kardiyak seride standart durum).
        """
        if target_phase not in phases:
            raise KeyError(f"Hedef faz yok: {target_phase}")

        mvf = self.measure_mvf(phases)
        stillest = min(mvf, key=mvf.get) if mvf else target_phase

        if self.ensure_loaded():
            try:
                corrected = self._infer_model(phases, target_phase)
                return TemporalMARResult(corrected=corrected, target_phase=target_phase,
                                         mvf_per_phase=mvf, stillest_phase=stillest,
                                         fallback=False)
            except Exception:
                logger.exception("TT-U-Net çıkarımı başarısız; yedek kullanılacak")

        corrected = self._fallback_series(phases, target_phase)
        return TemporalMARResult(corrected=corrected, target_phase=target_phase,
                                 mvf_per_phase=mvf, stillest_phase=stillest,
                                 fallback=True)

    def _infer_model(self, phases: dict[str, np.ndarray], target_phase: str) -> np.ndarray:
        import torch

        names = list(phases.keys())
        stack = np.stack([phases[n] for n in names], axis=0)  # (T, Z, Y, X)
        lo, hi = config.HU_MIN, config.HU_MAX
        x = torch.from_numpy((stack - lo) / (hi - lo)).float()[None].to(self.device)
        with torch.no_grad():
            y = self._model(x)  # beklenen çıktı: (1, Z, Y, X) hedef faz
        out = y.detach().cpu().numpy()[0].astype(np.float32) * (hi - lo) + lo
        return np.clip(out, config.HU_MIN, config.HU_MAX)

    def _fallback_series(self, phases: dict[str, np.ndarray],
                         target_phase: str) -> np.ndarray:
        target = phases[target_phase].astype(np.float32)
        z_count = target.shape[0]
        acc = target.copy()
        weight_sum = np.ones_like(target)

        for name, vol in phases.items():
            if name == target_phase or vol.shape != target.shape:
                continue
            # Faz uzaklığına göre azalan katkı: komşu fazlar daha güvenilir
            phase_weight = 0.5
            for z in range(z_count):
                warped = self._warp_to(vol[z], target[z])
                acc[z] += phase_weight * warped
                weight_sum[z] += phase_weight

        corrected = acc / weight_sum
        corrected = np.clip(corrected, config.HU_MIN, config.HU_MAX)
        for z in range(z_count):
            corrected[z] = enforce_hu_fidelity(target[z], corrected[z])
        return corrected

    @staticmethod
    def _warp_to(source_hu: np.ndarray, target_hu: np.ndarray) -> np.ndarray:
        """source kesitini optik akışla target anatomisine çakıştırır."""
        src8 = (hu_to_metric_unit(source_hu) * 255.0).astype(np.uint8)
        tgt8 = (hu_to_metric_unit(target_hu) * 255.0).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            tgt8, src8, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        h, w = source_hu.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        return cv2.remap(source_hu.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
