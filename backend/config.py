"""Merkezi yapılandırma: HU sınırları, model ağırlık yolları, parametre aralıkları.

Tüm sabitler referans repoların doğrulanmış değerlerinden alınmıştır:
- HU kırpma [-1024, 3071]: DICOM 12-bit CT standardı (RDN, ContrastGAN).
- NAB-GS hacimsel aralık [-512, 3071]: hvcl/3DCT_SR_DDNM_NAB-GS ön işleme.
- Gürültü referansları 30.0 ± 4.2 → 8.4 ± 1.3 HU: tkobayashi6/RDN makalesi.
- GMM kontrast kriteri mu_peak >= 300 HU: RoelvH97/ContrastGAN faz uygunluğu.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "clinical_learning.sqlite")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
# Tarayıcıdan yüklenen vakalar (testler env ile geçici klasöre yönlendirir)
UPLOADS_DIR = os.environ.get("CT_ENHANCER_UPLOADS",
                             os.path.join(PROJECT_ROOT, "uploads"))
# Sürükle-bırakla yüklenen vakaların dışa aktarımları (orijinal klasör
# tarayıcı güvenliği nedeniyle bilinemez; görünür bir yere yazılır)
EXPORTS_DIR = os.environ.get("CT_ENHANCER_EXPORTS",
                             os.path.join(PROJECT_ROOT, "disa_aktarimlar"))

# --- HU uzayı sınırları ---
HU_MIN = -1024.0
HU_MAX = 3071.0
NAB_GS_HU_MIN = -512.0   # 3DCT_SR_DDNM_NAB-GS hacimsel ön işleme aralığı
NAB_GS_HU_MAX = 3071.0

# --- Klinik referans gürültü değerleri (RDN makalesi, homojen ROI) ---
REF_SIGMA_HU_ORIGINAL = 30.0   # orijinal görüntü: 30.0 ± 4.2 HU
REF_SIGMA_HU_DENOISED = 8.4    # RDN sonrası: 8.4 ± 1.3 HU
HU_FIDELITY_TOLERANCE = 1.5    # ortalama HU sapması üst sınırı (± HU)

# --- Faz seçici (phase_selector.py) sabitleri ---
GMM_HU_RANGE = (-100.0, 1000.0)   # ContrastGAN histogram aralığı
GMM_COMPONENTS = 3
CONTRAST_OPTIMAL_HU = 300.0        # mu_peak >= 300 HU -> C_k = 1.0
NOISE_SCORE_SCALE = 40.0           # N_k = clip(1 - sigma/40, 0, 1)
PI_SCORE_SCALE = 10.0              # Q_k = clip(1 - PI/10, 0, 1)
PHASE_CENTER_SLICES = 15           # her fazın analiz edilen merkez kesit sayısı
MIN_ANALYSIS_SLICES = 8            # daha az kesitli seriler (scout, rapor) analiz dışı
PHASE_WEIGHTS = {"noise": 0.30, "contrast": 0.25, "motion": 0.25, "quality": 0.20}

# --- Kaydırıcı parametre aralıkları (denklem 2.1) ---
PARAM_RANGES = {
    "alpha":     (0.0, 1.5),    # gürültü giderme harmanlama oranı
    "gamma":     (0.01, 1.0),   # NAB-GS PReLU negatif alfa eğimi
    "beta":      (0.0, 1.0),    # Laplacian kenar keskinleştirme
    "lambda_tv": (0.0, 0.5),    # Total Variation pürüzsüzleştirme
    "kappa_hu":  (0.0, 2.0),    # ContrastGAN vasküler lümen kazancı
}
DEFAULT_PARAMS = {"alpha": 1.0, "gamma": 0.25, "beta": 0.0, "lambda_tv": 0.0, "kappa_hu": 0.0}

# --- Sürekli öğrenme (meta_optimizer.py) ---
MIN_CASES_FOR_TRAINING = 5  # N >= 5 olunca regresör kendini günceller
FEATURE_COLUMNS = [
    "kvp", "mas", "slice_thickness", "kernel_code",
    "sigma_hu_init", "mu_peak_init", "niqe_init", "ssim_inter_init",
]
TARGET_COLUMNS = ["model_id", "alpha", "gamma", "lambda_tv", "beta", "kappa_hu"]

# --- Model ağırlık yolları (yoksa sarmalayıcılar deterministik yedeğe düşer) ---
WEIGHT_PATHS = {
    "rdn": os.path.join(CHECKPOINTS_DIR, "rdn_cardiac_ct.pth"),
    "noise2sim": os.path.join(CHECKPOINTS_DIR, "noise2sim_clinic_adapted.pth"),
    "contrast_gan": os.path.join(CHECKPOINTS_DIR, "contrast_gan.pth"),
    "tt_unet": os.path.join(CHECKPOINTS_DIR, "tt_unet_4d.pth"),
    "nab_gs": os.path.join(CHECKPOINTS_DIR, "nab_gs_3d.pth"),
    "srcnn": os.path.join(CHECKPOINTS_DIR, "srcnn.pth"),
    "real_esrgan": os.path.join(CHECKPOINTS_DIR, "real_esrgan.pth"),
    "swinir": os.path.join(CHECKPOINTS_DIR, "swinir.pth"),
    "hat": os.path.join(CHECKPOINTS_DIR, "hat.pth"),
    "dat": os.path.join(CHECKPOINTS_DIR, "dat.pth"),
}

# --- Otomatik en iyi görüntü skoru ağırlıkları (denklem 2.2) ---
AUTO_BEST_WEIGHTS = {"pi": 0.4, "cnr": 0.4, "hu_fidelity": 0.2}
# Hızlı çıkarım havuzu: auto_best modunda paralel denenen modeller
FAST_MODEL_POOL = ["rdn", "noise2sim", "swinir", "hat", "real_esrgan"]


@dataclass
class BlendParams:
    """Denklem 2.1'in beş kaydırıcı parametresi; aralık dışı değerler kırpılır."""

    alpha: float = DEFAULT_PARAMS["alpha"]
    gamma: float = DEFAULT_PARAMS["gamma"]
    beta: float = DEFAULT_PARAMS["beta"]
    lambda_tv: float = DEFAULT_PARAMS["lambda_tv"]
    kappa_hu: float = DEFAULT_PARAMS["kappa_hu"]

    def __post_init__(self) -> None:
        for name, (lo, hi) in PARAM_RANGES.items():
            value = float(getattr(self, name))
            setattr(self, name, min(max(value, lo), hi))

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in PARAM_RANGES}
