"""RoelvH97/ContrastGAN sarmalayıcısı: atenüasyon kısıtlı kontrast fark haritası.

ContrastGAN, CCTA hacimlerinde kontrast düzeyini dönüştüren bir CycleGAN
türevi üretir; bu yazılımda görevi iki yönlüdür:
1. ΔI_contrast fark haritası üretimi (denklem 2.1'in κ_HU terimi).
2. GMM tabanlı optimal kontrast fazı kriteri (mu_peak >= 300 HU) —
   `phase_selector.gmm_contrast_peak` bu reponun yöntemini uygular.

Repo eğitim odaklıdır (Trainer sınıfı); çıkarım ağırlığı
(config.WEIGHT_PATHS['contrast_gan']) TorchScript/state_dict olarak
sağlanırsa yüklenir, aksi halde vasküler eşikleme tabanlı deterministik
yedek (`fallback_contrast_delta`) aynı ΔI_contrast sözleşmesini üretir.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from backend import config
from backend.engines.base_enhancer import (
    BaseCTEnhancer,
    _TORCH_AVAILABLE,
    fallback_contrast_delta,
    from_torch_batch,
    to_torch_batch,
)

logger = logging.getLogger(__name__)


class ContrastGANEnhancer(BaseCTEnhancer):
    model_id = "contrast_gan"
    display_name = "ContrastGAN (Vasküler Lümen)"
    quality_guard = False  # amacı gürültü gidermek değil kontrast eklemek

    def _load_model(self) -> Optional[Any]:
        if not _TORCH_AVAILABLE:
            return None
        weight_path = config.WEIGHT_PATHS["contrast_gan"]
        if not os.path.isfile(weight_path):
            return None
        import torch

        # TorchScript olarak dışa aktarılmış üreteci tercih et; değilse
        # state_dict'i yüklenebilecek bir üreteç mimarisi gerekir.
        try:
            model = torch.jit.load(weight_path, map_location=self.device)
            model.eval()
            logger.info("ContrastGAN TorchScript üreteci yüklendi: %s", weight_path)
            return model
        except Exception:
            logger.warning("ContrastGAN ağırlığı TorchScript değil; yedek aktif")
            return None

    def _infer(self, model: Any, slice_hu: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = to_torch_batch(slice_hu, device=self.device)
            y = model(x)
        return from_torch_batch(y)

    def _fallback(self, slice_hu: np.ndarray) -> np.ndarray:
        return slice_hu.astype(np.float32) + fallback_contrast_delta(slice_hu)

    # ------------------------------------------------------------------ #
    # κ_HU kaydırıcısı için fark haritası (ΔI_contrast)
    # ------------------------------------------------------------------ #
    def contrast_delta(self, slice_hu: np.ndarray) -> np.ndarray:
        """İki görüntü arası fark haritası ΔI_contrast = Î_GAN − I_orig."""
        result = self.enhance_slice(slice_hu)
        return result.delta
