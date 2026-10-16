"""Çok fazlı vakalarda otomatik en iyi faz seçimi (S_faz skoru).

Dört kriter, her fazın merkez 15 kesiti üzerinde hesaplanır:

1. C_k — GMM kontrast uygunluğu (RoelvH97/ContrastGAN): [-100, 1000] HU
   histogramına 3 bileşenli GaussianMixture; en yüksek ortalamalı bileşenin
   tepe noktası mu_peak >= 300 HU ise C_k = 1.0, değilse mu_peak/300.
2. N_k — homojen doku gürültüsü (tkobayashi6/RDN): homojen ROI std sapması
   sigma_HU üzerinden N_k = clip(1 - sigma/40, 0, 1).
3. Q_k — referanssız algısal kalite (labcisne/CT-Super-Resolution): pyiqa
   NIQE/NRQM ile PI = ((10 - NRQM) + NIQE)/2; Q_k = clip(1 - PI/10, 0, 1).
4. M_k — hareket tutarlılığı (ZihengD/TT-U-Net + MARVEL_CardiacCT): ardışık
   kesit SSIM'i ile komşu fazlara Farneback optik akış MVF büyüklüğünden
   M_k = SSIM_inter * exp(-||MVF||_2).

S_faz = 100 * (0.30*N_k + 0.25*C_k + 0.25*M_k + 0.20*Q_k)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity
from sklearn.mixture import GaussianMixture

from backend import config
from backend.core.ingestion import IngestedCase, PhaseVolume

logger = logging.getLogger(__name__)

# pyiqa / torch ağır ve opsiyonel: import hatasında deterministik vekile düşülür
try:  # pragma: no cover - ortam bağımlı
    import torch
    import pyiqa

    _PYIQA_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    pyiqa = None
    _PYIQA_AVAILABLE = False

# Algısal metrikler için standart yumuşak doku penceresi (WL=50, WW=400)
_METRIC_WINDOW_CENTER = 50.0
_METRIC_WINDOW_WIDTH = 400.0
_NEUTRAL_NRQM = 5.0  # NRQM hesaplanamazsa kullanılan nötr değer

# Büyük gerçek vakalarda (10+ faz) analiz süresini sınırlamak için ortalama
# metrikler alt örneklenir; eşit aralıklı seçim yansız bir kestirim verir.
_MAX_PERCEPTUAL_SLICES = 5   # NIQE/NRQM için faz başına en çok kesit
_MAX_MVF_SLICES = 3          # Farneback optik akış için en çok kesit çifti


def _subsample(stack: np.ndarray, max_n: int) -> np.ndarray:
    """Eşit aralıklı en çok max_n kesit seçer (sıra korunur)."""
    n = stack.shape[0]
    if n <= max_n:
        return stack
    idx = np.linspace(0, n - 1, max_n).round().astype(int)
    return stack[idx]


@dataclass
class PhaseMetrics:
    """Tek fazın tüm ölçümleri ve bileşen skorları."""

    phase_name: str
    num_slices: int = 0
    sigma_hu: float = 0.0
    mu_peak_hu: float = 0.0
    niqe: float = 0.0
    nrqm: float = _NEUTRAL_NRQM
    pi: float = 0.0
    ssim_inter: float = 0.0
    mvf_norm: float = 0.0
    n_k: float = 0.0
    c_k: float = 0.0
    m_k: float = 0.0
    q_k: float = 0.0
    s_faz: float = 0.0
    recommended: bool = False
    pyiqa_used: bool = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------- #
# Tekrar kullanılabilir metrik yardımcıları (engines/ de bunları çağırır)
# ---------------------------------------------------------------------- #
def measure_homogeneous_sigma(slice_hu: np.ndarray, window: int = 9) -> float:
    """Homojen doku ROI'sindeki gürültü std'sini (sigma_HU) kestirir.

    Yumuşak doku maskesi ([-100, 200] HU) içinde kayan pencere yerel std
    haritası çıkarılır; en homojen bölgeyi temsilen 5. yüzdelik alınır.
    """
    img = slice_hu.astype(np.float32)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img * img, size=window)
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    local_std = np.sqrt(local_var)

    mask = (img > -100.0) & (img < 200.0)
    # Pencere kenar etkisini dışla
    pad = window // 2 + 1
    mask[:pad, :] = mask[-pad:, :] = False
    mask[:, :pad] = mask[:, -pad:] = False

    if mask.sum() < window * window * 4:
        # Yeterli doku yoksa merkez kırpmanın global std'si
        h, w = img.shape
        crop = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        return float(np.std(crop))
    return float(np.percentile(local_std[mask], 5.0))


def gmm_contrast_peak(slices_hu: np.ndarray, random_state: int = 0) -> float:
    """ContrastGAN yöntemi: [-100, 1000] HU piksellerine 3 bileşenli GMM.

    En yüksek ortalamalı Gauss bileşeninin tepe noktasını (mu_peak) döndürür.
    """
    lo, hi = config.GMM_HU_RANGE
    values = slices_hu[(slices_hu >= lo) & (slices_hu <= hi)].reshape(-1, 1)
    if values.shape[0] < 100:
        return 0.0
    # GMM maliyetini sınırlamak için deterministik alt örnekleme
    if values.shape[0] > 200_000:
        step = values.shape[0] // 200_000
        values = values[::step]
    try:
        gmm = GaussianMixture(
            n_components=config.GMM_COMPONENTS,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
        ).fit(values)
    except Exception:
        logger.exception("GMM uydurulamadı; mu_peak=0 döndürülüyor")
        return 0.0
    return float(np.max(gmm.means_))


def hu_to_metric_unit(slice_hu: np.ndarray) -> np.ndarray:
    """HU kesitini algısal metrikler için [0, 1] pencereli görüntüye çevirir."""
    lo = _METRIC_WINDOW_CENTER - _METRIC_WINDOW_WIDTH / 2.0
    hi = _METRIC_WINDOW_CENTER + _METRIC_WINDOW_WIDTH / 2.0
    return np.clip((slice_hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class PerceptualScorer:
    """pyiqa NIQE/NRQM sarmalayıcısı; kullanılamazsa deterministik vekil.

    Vekil (fallback) PI: gürültü (yerel std) ve keskinlik (Laplacian enerji)
    birleşiminden türetilen, [0, 10] aralığında monoton bir kalite endeksi.
    pyiqa yüklüyse gerçek NIQE ve NRQM hesaplanır; NRQM (çok yavaş olabilir)
    başarısız olursa nötr 5.0 kullanılır ve PI yalnız NIQE'den etkilenir.
    """

    def __init__(self, use_pyiqa: bool = True, compute_nrqm: bool = False,
                 device: str = "cpu") -> None:
        self.compute_nrqm = compute_nrqm
        self._niqe = None
        self._nrqm = None
        self.available = False
        if use_pyiqa and _PYIQA_AVAILABLE:
            try:
                self._niqe = pyiqa.create_metric("niqe", device=device)
                self.available = True
            except Exception:
                logger.exception("pyiqa niqe metriği oluşturulamadı; vekile düşülüyor")
            if compute_nrqm and self.available:
                try:
                    self._nrqm = pyiqa.create_metric("nrqm", device=device)
                except Exception:
                    logger.warning("pyiqa nrqm oluşturulamadı; nötr NRQM=%.1f", _NEUTRAL_NRQM)

    def score_slices(self, slices_hu: np.ndarray) -> tuple[float, float, float, bool]:
        """(niqe, nrqm, pi, pyiqa_kullanıldı_mı) döndürür."""
        if self.available:
            try:
                return self._score_pyiqa(slices_hu)
            except Exception:
                logger.exception("pyiqa skoru başarısız; vekile düşülüyor")
        return self._score_proxy(slices_hu)

    def _score_pyiqa(self, slices_hu: np.ndarray) -> tuple[float, float, float, bool]:
        niqe_vals, nrqm_vals = [], []
        for sl in _subsample(slices_hu, _MAX_PERCEPTUAL_SLICES):
            img = hu_to_metric_unit(sl)
            tensor = torch.from_numpy(img)[None, None, ...].repeat(1, 3, 1, 1)
            niqe_vals.append(float(self._niqe(tensor)))
            if self._nrqm is not None:
                nrqm_vals.append(float(self._nrqm(tensor)))
        niqe = float(np.mean(niqe_vals)) if niqe_vals else 10.0
        nrqm = float(np.mean(nrqm_vals)) if nrqm_vals else _NEUTRAL_NRQM
        pi = 0.5 * ((10.0 - nrqm) + niqe)
        return niqe, nrqm, pi, True

    def _score_proxy(self, slices_hu: np.ndarray) -> tuple[float, float, float, bool]:
        import cv2

        niqe_like, sharpness = [], []
        for sl in _subsample(slices_hu, _MAX_PERCEPTUAL_SLICES):
            sigma = measure_homogeneous_sigma(sl)
            # NIQE tipik aralığı ~[2, 10]; sigma_HU [5, 45] -> [2, 10] eşleme
            niqe_like.append(float(np.clip(2.0 + (sigma - 5.0) * (8.0 / 40.0), 2.0, 10.0)))
            img = hu_to_metric_unit(sl)
            lap = cv2.Laplacian(img, cv2.CV_32F)
            sharpness.append(float(np.var(lap)))
        niqe = float(np.mean(niqe_like))
        # Keskinlik arttıkça NRQM (iyi yönde) artar; [0, 0.02] varyansı [3, 8]'e eşle
        nrqm = float(np.clip(3.0 + np.mean(sharpness) * 250.0, 3.0, 8.0))
        pi = 0.5 * ((10.0 - nrqm) + niqe)
        return niqe, nrqm, pi, False


def interslice_ssim(slices_hu: np.ndarray) -> float:
    """Ardışık kesitler arası ortalama SSIM (TT-U-Net hareket tutarlılığı)."""
    if slices_hu.shape[0] < 2:
        return 1.0
    scores = []
    for a, b in zip(slices_hu[:-1], slices_hu[1:]):
        scores.append(
            structural_similarity(a, b, data_range=float(config.HU_MAX - config.HU_MIN))
        )
    return float(np.mean(scores))


def mvf_magnitude(slices_a: np.ndarray, slices_b: np.ndarray) -> float:
    """İki faz arasındaki ortalama hareket vektör alanı büyüklüğü ||MVF||_2.

    MARVEL_CardiacCT'nin MVF prensibinin görüntü uzayı karşılığı: eşleşen
    kesit çiftlerinde Farneback yoğun optik akış, piksel başına akış
    büyüklüğünün ortalaması (piksel biriminde).
    """
    import cv2

    n = min(slices_a.shape[0], slices_b.shape[0])
    if n == 0:
        return 0.0
    indices = np.linspace(0, n - 1, min(n, _MAX_MVF_SLICES)).round().astype(int)
    magnitudes = []
    for i in indices:
        a = (hu_to_metric_unit(slices_a[i]) * 255.0).astype(np.uint8)
        b = (hu_to_metric_unit(slices_b[i]) * 255.0).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitudes.append(float(np.mean(np.linalg.norm(flow, axis=2))))
    return float(np.mean(magnitudes))


# ---------------------------------------------------------------------- #
# Faz seçici
# ---------------------------------------------------------------------- #
class PhaseSelector:
    """Bir vakanın tüm fazlarını puanlar ve en iyisini önerir."""

    def __init__(self, use_pyiqa: bool = True, compute_nrqm: bool = False,
                 device: str = "cpu") -> None:
        self.scorer = PerceptualScorer(use_pyiqa=use_pyiqa,
                                       compute_nrqm=compute_nrqm, device=device)

    def analyze_case(self, case: IngestedCase) -> list[PhaseMetrics]:
        # Hacimsel olmayan serileri (scout, doz/EKG raporu...) analizden çıkar;
        # hiçbiri eşiği geçmiyorsa hepsiyle devam et (tek ince seri durumu).
        phase_names = [n for n, pv in case.phases.items()
                       if pv.num_slices >= config.MIN_ANALYSIS_SLICES]
        if not phase_names:
            phase_names = list(case.phases.keys())
        center_stacks = {
            name: case.phases[name].center_slices() for name in phase_names
        }

        results: list[PhaseMetrics] = []
        for idx, name in enumerate(phase_names):
            stack = center_stacks[name]
            metrics = self.analyze_phase(name, stack)

            # Komşu fazlara göre MVF (tek fazlı vakada 0 kalır -> exp(0)=1).
            # Farklı kesit boyutlu komşular (örn. scout) karşılaştırılamaz.
            neighbor_mags = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(phase_names):
                    other = center_stacks[phase_names[j]]
                    if other.shape[1:] == stack.shape[1:]:
                        neighbor_mags.append(mvf_magnitude(stack, other))
            metrics.mvf_norm = float(np.mean(neighbor_mags)) if neighbor_mags else 0.0
            metrics.m_k = float(
                np.clip(metrics.ssim_inter * np.exp(-metrics.mvf_norm), 0.0, 1.0)
            )
            metrics.s_faz = self._total_score(metrics)
            results.append(metrics)

        if results:
            best = max(results, key=lambda m: m.s_faz)
            best.recommended = True
        return results

    def analyze_phase(self, name: str, center_slices: np.ndarray) -> PhaseMetrics:
        """Tek fazın MVF dışındaki tüm metriklerini hesaplar."""
        m = PhaseMetrics(phase_name=name, num_slices=int(center_slices.shape[0]))

        sigmas = [measure_homogeneous_sigma(sl) for sl in center_slices]
        m.sigma_hu = float(np.mean(sigmas))
        m.n_k = float(np.clip(1.0 - m.sigma_hu / config.NOISE_SCORE_SCALE, 0.0, 1.0))

        m.mu_peak_hu = gmm_contrast_peak(center_slices)
        if m.mu_peak_hu >= config.CONTRAST_OPTIMAL_HU:
            m.c_k = 1.0
        else:
            m.c_k = float(np.clip(m.mu_peak_hu / config.CONTRAST_OPTIMAL_HU, 0.0, 1.0))

        m.niqe, m.nrqm, m.pi, m.pyiqa_used = self.scorer.score_slices(center_slices)
        m.q_k = float(np.clip(1.0 - m.pi / config.PI_SCORE_SCALE, 0.0, 1.0))

        m.ssim_inter = interslice_ssim(center_slices)
        return m

    @staticmethod
    def _total_score(m: PhaseMetrics) -> float:
        w = config.PHASE_WEIGHTS
        return float(
            100.0 * (w["noise"] * m.n_k + w["contrast"] * m.c_k
                     + w["motion"] * m.m_k + w["quality"] * m.q_k)
        )
