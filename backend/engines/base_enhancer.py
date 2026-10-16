"""Soyut BaseCTEnhancer sınıfı ve paylaşılan deterministik yedek çekirdekler.

Her sarmalayıcı şu sözleşmeye uyar:
- Girdi/çıktı daima float32 HU uzayındadır.
- Gerçek model ağırlığı yüklenemiyorsa sistem ÇÖKMEZ: bilateral / non-local
  means + vasküler eşikleme tabanlı deterministik yedek çekirdek, referans
  ΔI istatistiklerini (homojen ROI sigma_HU ≈ 30 → 8.4) üretir ve sonuç
  `fallback=True` ile işaretlenir.
- Çıkış, HU atenüasyon sadakati korumasından geçer: doku maskesinde ortalama
  HU sapması ±1.5 HU'yu aşarsa DC ofset düzeltilir (yalnız sigma düşmelidir).
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from backend import config
from backend.core.phase_selector import measure_homogeneous_sigma

logger = logging.getLogger(__name__)

try:  # pragma: no cover - ortam bağımlı
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False


@dataclass
class EnhanceResult:
    """Tek kesit çıkarım sonucu."""

    i_dl: np.ndarray            # model (veya yedek) çıktısı, float32 HU
    delta: np.ndarray           # ΔI = i_dl − i_orig
    model_id: str
    fallback: bool              # True -> deterministik yedek çekirdek kullanıldı
    runtime_s: float
    sigma_before: float = 0.0
    sigma_after: float = 0.0
    extras: dict = field(default_factory=dict)


class BaseCTEnhancer(ABC):
    """Tüm kesit bazlı iyileştirme motorlarının soyut tabanı."""

    model_id: str = "base"
    display_name: str = "Base Enhancer"
    # Kalite korkuluğu: gürültü giderici bir modelin çıktısı homojen ROI
    # sigma'sını DÜŞÜRMEK zorundadır; artırıyorsa (bozuk/uyumsuz checkpoint)
    # deterministik yedeğe düşülür. Kontrast motoru gibi amacı gürültü
    # gidermek olmayan sınıflar bunu False yapar.
    quality_guard: bool = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Any = None
        self._load_attempted = False

    # ------------------------------------------------------------------ #
    # Alt sınıf sözleşmesi
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _load_model(self) -> Optional[Any]:
        """Gerçek modeli kurup ağırlıkları yükler; başarısızsa None döner."""

    @abstractmethod
    def _infer(self, model: Any, slice_hu: np.ndarray) -> np.ndarray:
        """Gerçek modelle tek kesit çıkarımı (float32 HU -> float32 HU)."""

    @abstractmethod
    def _fallback(self, slice_hu: np.ndarray) -> np.ndarray:
        """Deterministik yedek çekirdek (float32 HU -> float32 HU)."""

    # ------------------------------------------------------------------ #
    # Ortak akış
    # ------------------------------------------------------------------ #
    def ensure_loaded(self) -> bool:
        if not self._load_attempted:
            self._load_attempted = True
            try:
                self._model = self._load_model()
            except Exception:
                logger.exception("%s modeli yüklenemedi; yedek çekirdek aktif",
                                 self.model_id)
                self._model = None
            if self._model is None:
                logger.info("%s: ağırlık yok/yüklenemedi -> deterministik yedek",
                            self.model_id)
        return self._model is not None

    def enhance_slice(self, slice_hu: np.ndarray) -> EnhanceResult:
        i_orig = np.ascontiguousarray(slice_hu, dtype=np.float32)
        sigma_before = measure_homogeneous_sigma(i_orig)

        start = time.perf_counter()
        used_fallback = not self.ensure_loaded()
        if not used_fallback:
            try:
                i_dl = self._infer(self._model, i_orig)
            except Exception:
                logger.exception("%s çıkarımı başarısız; yedek çekirdeğe düşüldü",
                                 self.model_id)
                used_fallback = True
        if used_fallback:
            i_dl = self._fallback(i_orig)

        i_dl = np.clip(i_dl.astype(np.float32), config.HU_MIN, config.HU_MAX)
        i_dl = enforce_hu_fidelity(i_orig, i_dl)
        sigma_after = measure_homogeneous_sigma(i_dl)

        # Kalite korkuluğu: model gürültüyü artırdıysa yedeğe düş
        if (not used_fallback and self.quality_guard
                and sigma_after > sigma_before * 1.05):
            logger.warning(
                "%s model çıktısı gürültüyü artırdı (%.1f -> %.1f HU); "
                "deterministik yedek kullanılıyor", self.model_id,
                sigma_before, sigma_after,
            )
            used_fallback = True
            i_dl = np.clip(self._fallback(i_orig).astype(np.float32),
                           config.HU_MIN, config.HU_MAX)
            i_dl = enforce_hu_fidelity(i_orig, i_dl)
            sigma_after = measure_homogeneous_sigma(i_dl)
        runtime = time.perf_counter() - start

        return EnhanceResult(
            i_dl=i_dl,
            delta=i_dl - i_orig,
            model_id=self.model_id,
            fallback=used_fallback,
            runtime_s=runtime,
            sigma_before=sigma_before,
            sigma_after=sigma_after,
        )


# ---------------------------------------------------------------------- #
# HU sadakati koruması
# ---------------------------------------------------------------------- #
def enforce_hu_fidelity(i_orig: np.ndarray, i_dl: np.ndarray,
                        tolerance: float = config.HU_FIDELITY_TOLERANCE) -> np.ndarray:
    """Doku maskesinde ortalama HU sapması toleransı aşarsa DC ofseti düzeltir.

    RDN ve ContrastGAN çalışmalarının şartı: gürültü giderme yalnız sigma_HU
    düşürür, mutlak atenüasyon ortalamasını (mu_HU) kaydırmaz.
    """
    mask = (i_orig > -300.0) & (i_orig < 300.0)
    if mask.sum() < 64:
        return i_dl
    bias = float(np.mean(i_dl[mask]) - np.mean(i_orig[mask]))
    if abs(bias) > tolerance:
        i_dl = i_dl - np.float32(bias)
        i_dl = np.clip(i_dl, config.HU_MIN, config.HU_MAX)
    return i_dl


# ---------------------------------------------------------------------- #
# Paylaşılan deterministik yedek çekirdekler
# ---------------------------------------------------------------------- #
def fallback_denoise(
    slice_hu: np.ndarray,
    target_ratio: float = config.REF_SIGMA_HU_DENOISED / config.REF_SIGMA_HU_ORIGINAL,
    bilateral_d: int = 9,
    sigma_color: float = 60.0,
    sigma_space: float = 5.0,
    use_nlm: bool = False,
) -> np.ndarray:
    """Bilateral (+ isteğe bağlı hızlı NLM) ile referans sigma oranını hedefler.

    Referans istatistik (RDN makalesi): sigma 30.0 HU -> 8.4 HU, yani oran
    ~0.28. Filtre sonrası artık, hedef sigmaya ulaşacak şekilde doğrusal
    ölçeklenir; böylece yedek çekirdek, gerçek modelin ürettiği ΔI
    istatistiklerini deterministik olarak taklit eder.
    """
    img = slice_hu.astype(np.float32)
    denoised = cv2.bilateralFilter(img, d=bilateral_d,
                                   sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if use_nlm:
        # 8-bit pencerelenmiş kopya üzerinde hızlı NLM, artığı HU'ya geri taşı
        lo, hi = -160.0, 240.0
        win = np.clip((denoised - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        nlm = cv2.fastNlMeansDenoising(win, None, h=7, templateWindowSize=7,
                                       searchWindowSize=21)
        denoised = denoised + (nlm.astype(np.float32) - win.astype(np.float32)) * ((hi - lo) / 255.0)

    sigma_before = measure_homogeneous_sigma(img)
    sigma_after = measure_homogeneous_sigma(denoised)
    target = max(sigma_before * target_ratio, 0.1)

    # i_out = img + s*(denoised-img); s'yi hedef sigmaya ulaşacak şekilde seç
    if sigma_before - sigma_after > 1e-3:
        scale = (sigma_before - target) / (sigma_before - sigma_after)
        scale = float(np.clip(scale, 0.0, 1.5))
    else:
        scale = 1.0
    return img + np.float32(scale) * (denoised - img)


def fallback_contrast_delta(slice_hu: np.ndarray,
                            lumen_threshold: float = 150.0,
                            boost_hu: float = 80.0) -> np.ndarray:
    """ContrastGAN yedeği: vasküler eşikleme ile ΔI_contrast fark haritası.

    Kontrastlı lümen adayı voksellere (>= eşik HU) yumuşatılmış, HU ile
    orantılı bir güçlendirme haritası üretir. κ_HU kaydırıcısı bu haritayı
    çarpan olarak kullanır (denklem 2.1'in son terimi).
    """
    img = slice_hu.astype(np.float32)
    lumen = np.clip((img - lumen_threshold) / 150.0, 0.0, 1.0)
    lumen = cv2.GaussianBlur(lumen, (0, 0), sigmaX=1.5)
    return (lumen * boost_hu).astype(np.float32)


def fallback_sharpen(slice_hu: np.ndarray, amount: float = 0.6,
                     sigma: float = 1.0) -> np.ndarray:
    """Unsharp-masking tabanlı deterministik keskinleştirme (SR yedekleri için)."""
    img = slice_hu.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return img + np.float32(amount) * (img - blurred)


def to_torch_batch(slice_hu: np.ndarray, device: str = "cpu",
                   norm: tuple[float, float] = (config.HU_MIN, config.HU_MAX)):
    """HU kesitini (1,1,H,W) normalize torch tensörüne çevirir."""
    lo, hi = norm
    arr = (slice_hu.astype(np.float32) - lo) / (hi - lo)
    return torch.from_numpy(arr)[None, None, ...].to(device)


def from_torch_batch(tensor, norm: tuple[float, float] = (config.HU_MIN, config.HU_MAX)) -> np.ndarray:
    """(1,1,H,W) normalize tensörü float32 HU kesitine geri çevirir."""
    lo, hi = norm
    arr = tensor.detach().cpu().numpy()[0, 0].astype(np.float32)
    return arr * (hi - lo) + lo
