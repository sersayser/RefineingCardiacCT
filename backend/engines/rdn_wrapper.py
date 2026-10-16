"""tkobayashi6/RDN-for-Denoising-in-CardiacCT sarmalayıcısı.

Birincil hızlı gürültü giderme motoru (0.255 sn/kesit sınıfı). Gerçek mimari
`third_party/RDN-for-Denoising-in-CardiacCT/model/rdn_denoise.py` içindeki
`RDN_denoise(args)` sınıfıdır; varsayılan hiperparametreler reponun
`option.py` dosyasındaki değerlerdir (D=20, C=6, G=32, G0=32, PReLU).

Ağırlık dosyası (config.WEIGHT_PATHS['rdn']) yoksa bilateral tabanlı
deterministik yedek, RDN makalesinin referans istatistiğini
(sigma_HU 30.0 -> 8.4) hedefleyerek ΔI üretir.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from backend import config
from backend.engines.base_enhancer import (
    BaseCTEnhancer,
    _TORCH_AVAILABLE,
    fallback_denoise,
    from_torch_batch,
    to_torch_batch,
)

logger = logging.getLogger(__name__)

RDN_REPO_DIR = os.path.join(config.THIRD_PARTY_DIR, "RDN-for-Denoising-in-CardiacCT")

# option.py varsayılanları (repo ile birebir)
RDN_DEFAULT_ARGS = SimpleNamespace(
    n_channels=1, D=20, C=6, G=32, G0=32,
    RDN_ksize=3, SFENet_ksize=3, last_conv_ksize=3,
    activation="PReLU",
)


class RDNEnhancer(BaseCTEnhancer):
    model_id = "rdn"
    display_name = "RDN-CardiacCT (Gürültü Giderme)"

    def _load_model(self) -> Optional[Any]:
        if not _TORCH_AVAILABLE:
            return None
        weight_path = config.WEIGHT_PATHS["rdn"]
        if not os.path.isfile(weight_path):
            return None
        if RDN_REPO_DIR not in sys.path:
            sys.path.insert(0, RDN_REPO_DIR)
        from model.rdn_denoise import RDN_denoise  # type: ignore
        import torch

        model = RDN_denoise(RDN_DEFAULT_ARGS)
        state = torch.load(weight_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device).eval()
        logger.info("RDN ağırlıkları yüklendi: %s", weight_path)
        return model

    def _infer(self, model: Any, slice_hu: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = to_torch_batch(slice_hu, device=self.device)
            y = model(x)
        return from_torch_batch(y)

    def _fallback(self, slice_hu: np.ndarray) -> np.ndarray:
        return fallback_denoise(slice_hu)
