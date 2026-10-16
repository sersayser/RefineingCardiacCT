"""labcisne/CT-Super-Resolution 5 model havuzu sarmalayıcısı.

Referans repo chaiNNer şablonu + değerlendirme notebook'larından oluşur
(SRCNN, Real-ESRGAN, SwinIR, HAT, DAT; `pyiqa` ile NIQE/NRQM/PI ölçümü).
İçe aktarılabilir mimari kodu bulunmadığından gerçek ağırlıklar, chaiNNer'ın
da kullandığı `spandrel` genel .pth yükleyicisiyle (kuruluysa) yüklenir;
aksi halde her model varyantı, kendine özgü deterministik profil
(gürültü bastırma + keskinleştirme karışımı) ile yedek çekirdekte çalışır —
böylece `auto_best` havuzu ağırlıksız ortamda da ayrışan adaylar üretir.
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
    fallback_denoise,
    fallback_sharpen,
    from_torch_batch,
    to_torch_batch,
)

logger = logging.getLogger(__name__)

# Varyant profilleri: (denoise_ratio_hedefi, sharpen_amount, nlm)
# Oranlar görecelidir: SRCNN en yumuşak, HAT/DAT en keskin aday.
SR_VARIANT_PROFILES: dict[str, dict] = {
    "srcnn":       {"target_ratio": 0.45, "sharpen": 0.15, "nlm": False},
    "real_esrgan": {"target_ratio": 0.35, "sharpen": 0.45, "nlm": True},
    "swinir":      {"target_ratio": 0.30, "sharpen": 0.35, "nlm": False},
    "hat":         {"target_ratio": 0.28, "sharpen": 0.55, "nlm": False},
    "dat":         {"target_ratio": 0.30, "sharpen": 0.50, "nlm": True},
}

SR_DISPLAY_NAMES = {
    "srcnn": "SRCNN",
    "real_esrgan": "Real-ESRGAN",
    "swinir": "SwinIR",
    "hat": "HAT",
    "dat": "DAT",
}


class SRBenchmarkEnhancer(BaseCTEnhancer):
    """Havuzdaki tek bir SR modelinin kesit bazlı sarmalayıcısı."""

    def __init__(self, variant: str, device: str = "cpu") -> None:
        if variant not in SR_VARIANT_PROFILES:
            raise ValueError(f"Bilinmeyen SR varyantı: {variant}")
        super().__init__(device=device)
        self.variant = variant
        self.model_id = variant
        self.display_name = f"{SR_DISPLAY_NAMES[variant]} (CT-SR Havuzu)"

    def _load_model(self) -> Optional[Any]:
        if not _TORCH_AVAILABLE:
            return None
        weight_path = config.WEIGHT_PATHS.get(self.variant, "")
        if not os.path.isfile(weight_path):
            return None
        try:
            from spandrel import ModelLoader  # chaiNNer'ın model yükleyicisi
        except ImportError:
            logger.info("spandrel kurulu değil; %s yedek profille çalışacak",
                        self.variant)
            return None
        descriptor = ModelLoader().load_from_file(weight_path)
        descriptor.model.to(self.device).eval()
        logger.info("%s ağırlıkları spandrel ile yüklendi", self.variant)
        return descriptor

    def _infer(self, descriptor: Any, slice_hu: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            x = to_torch_batch(slice_hu, device=self.device)
            if getattr(descriptor, "input_channels", 1) == 3:
                x = x.repeat(1, 3, 1, 1)
            y = descriptor.model(x)
            if y.shape[1] == 3:
                y = y.mean(dim=1, keepdim=True)
            # SR ölçek > 1 ise kesit çözünürlüğüne geri örnekle (kesit bazlı mod)
            if y.shape[-2:] != x.shape[-2:]:
                y = F.interpolate(y, size=x.shape[-2:], mode="bicubic",
                                  align_corners=False)
        return from_torch_batch(y)

    def _fallback(self, slice_hu: np.ndarray) -> np.ndarray:
        profile = SR_VARIANT_PROFILES[self.variant]
        denoised = fallback_denoise(slice_hu,
                                    target_ratio=profile["target_ratio"],
                                    use_nlm=profile["nlm"])
        return fallback_sharpen(denoised, amount=profile["sharpen"])


def build_sr_pool(device: str = "cpu") -> dict[str, SRBenchmarkEnhancer]:
    """Havuzdaki 5 SR modelinin sarmalayıcılarını kurar."""
    return {v: SRBenchmarkEnhancer(v, device=device) for v in SR_VARIANT_PROFILES}
