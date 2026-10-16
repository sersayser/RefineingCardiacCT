"""niuchuangnn/noise2sim sarmalayıcısı: unet2 çıkarımı + öz-denetimli adaptasyon.

Gerçek mimari `third_party/noise2sim/noise2sim/modeling/networks/unet.py`
içindeki `UNet2` sınıfıdır (ldct konfigürasyonlarında model_type = "unet2",
ps3_ns8: patch_size=3, num_select=8 benzerlik örneklemesi).

İki görev:
1. Çıkarım: referanssız vakalarda tek kesit gürültü bastırma (~0.3 sn/kesit).
2. Arka plan adaptasyonu (`Noise2SimAdapter`): onaylanan vakaların gürültülü
   kesitleriyle noise2sim'in öz-denetimli ilkesini (benzer gürültülü örnekten
   benzer gürültülü örneğe eşleme; CT'de komşu kesitler doğal benzer çifttir)
   uygulayan kısa ince ayar döngüsü. Güncellenen ağırlık
   `checkpoints/noise2sim_clinic_adapted.pth` dosyasına yazılır.
"""

from __future__ import annotations

import logging
import os
import sys
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

NOISE2SIM_REPO_DIR = os.path.join(config.THIRD_PARTY_DIR, "noise2sim")


def _build_unet2(device: str):
    """Repodaki UNet2'yi (unet2 yapılandırması) kurar."""
    if NOISE2SIM_REPO_DIR not in sys.path:
        sys.path.insert(0, NOISE2SIM_REPO_DIR)
    from noise2sim.modeling.networks.unet import UNet2  # type: ignore

    model = UNet2(n_channels=1, n_classes=1, bilinear=False,
                  residual=False, activation_type="relu", use_bn=True)
    return model.to(device)


class Noise2SimEnhancer(BaseCTEnhancer):
    model_id = "noise2sim"
    display_name = "Noise2Sim unet2 (Öz-Denetimli)"

    def _load_model(self) -> Optional[Any]:
        if not _TORCH_AVAILABLE:
            return None
        weight_path = config.WEIGHT_PATHS["noise2sim"]
        if not os.path.isfile(weight_path):
            return None
        import torch

        model = _build_unet2(self.device)
        state = torch.load(weight_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        logger.info("Noise2Sim ağırlıkları yüklendi: %s", weight_path)
        return model

    def _infer(self, model: Any, slice_hu: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = to_torch_batch(slice_hu, device=self.device)
            y = model(x)
        return from_torch_batch(y)

    def _fallback(self, slice_hu: np.ndarray) -> np.ndarray:
        # Öz-denetimli karaktere en yakın klasik eşdeğer: bilateral + hızlı NLM
        return fallback_denoise(slice_hu, use_nlm=True)


class Noise2SimAdapter:
    """Arka plan öz-denetimli ince ayar: kliniğin BT cihazına adaptasyon.

    Noise2Sim ilkesi: gürültülü görüntü, BENZER bir gürültülü görüntüye
    eşlenerek eğitilir; bağımsız gürültü ortalamada sıfırlanır ve ağ sinyali
    öğrenir. CT hacimlerinde komşu kesitler (anatomi ~aynı, gürültü bağımsız)
    doğal benzer çifttir — ldct yapılandırmalarının kesit-komşuluğu stratejisi.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def available(self) -> bool:
        return _TORCH_AVAILABLE

    def fine_tune(self, volumes: list[np.ndarray], steps: int = 200,
                  lr: float = 1e-4, batch_size: int = 4,
                  patch: int = 128, progress_cb=None) -> Optional[str]:
        """Onaylanan vakaların hacimleriyle kısa ince ayar; checkpoint yolu döner."""
        if not _TORCH_AVAILABLE:
            logger.warning("torch yok; noise2sim adaptasyonu atlandı")
            return None
        import torch

        pairs = self._neighbor_pairs(volumes)
        if len(pairs) < batch_size:
            logger.info("Adaptasyon için yeterli kesit çifti yok (%d)", len(pairs))
            return None

        model = _build_unet2(self.device)
        weight_path = config.WEIGHT_PATHS["noise2sim"]
        if os.path.isfile(weight_path):
            try:
                state = torch.load(weight_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(
                    {k.replace("module.", "", 1): v for k, v in state.items()},
                    strict=False,
                )
            except Exception:
                logger.exception("Mevcut checkpoint okunamadı; sıfırdan ince ayar")

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        rng = np.random.default_rng(0)

        for step in range(steps):
            batch_in, batch_tgt = [], []
            for _ in range(batch_size):
                a, b = pairs[rng.integers(len(pairs))]
                ay, ax = a.shape
                if ay > patch and ax > patch:
                    y0 = int(rng.integers(0, ay - patch))
                    x0 = int(rng.integers(0, ax - patch))
                    a = a[y0:y0 + patch, x0:x0 + patch]
                    b = b[y0:y0 + patch, x0:x0 + patch]
                batch_in.append(to_torch_batch(a, device=self.device))
                batch_tgt.append(to_torch_batch(b, device=self.device))
            x = torch.cat(batch_in, dim=0)
            t = torch.cat(batch_tgt, dim=0)

            optimizer.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            optimizer.step()
            if progress_cb and step % 20 == 0:
                progress_cb(step / steps, float(loss.item()))

        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, weight_path)
        logger.info("Noise2Sim klinik adaptasyonu kaydedildi: %s", weight_path)
        return weight_path

    @staticmethod
    def _neighbor_pairs(volumes: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Komşu kesit çiftleri (noise2sim 'benzer gürültülü örnek' stratejisi)."""
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for vol in volumes:
            if vol.ndim != 3 or vol.shape[0] < 2:
                continue
            for i in range(vol.shape[0] - 1):
                pairs.append((vol[i].astype(np.float32), vol[i + 1].astype(np.float32)))
        return pairs
