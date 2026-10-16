"""Meta-regresör: onaylanan vakalardan optimal parametre kombinasyonu öğrenme.

Özellik vektörü X_i ∈ R^8:
    [kVp, mAs, SliceThickness, KernelCode,
     sigma_HU_init, mu_peak_init, NIQE_init, SSIM_inter_init]
Hedef vektörü Y_i ∈ R^6:
    [Model_ID, alpha*, gamma*, lambda_TV*, beta*, kappa_HU*]

Model: MultiOutputRegressor(HistGradientBoostingRegressor()). Onaylanan vaka
sayısı N >= config.MIN_CASES_FOR_TRAINING (5) olduğunda otomatik yeniden
eğitilir. Model_ID kategorik olduğundan dahili tamsayı koda eşlenir ve
tahminde en yakın koda yuvarlanır.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from backend import config
from backend.learning.models_db import CaseRecord

logger = logging.getLogger(__name__)

# Model_ID <-> tamsayı kod eşlemesi (kararlı sıra)
MODEL_ID_CODES = {
    "rdn": 0,
    "noise2sim": 1,
    "swinir": 2,
    "hat": 3,
    "real_esrgan": 4,
    "srcnn": 5,
    "dat": 6,
    "contrast_gan": 7,
    "tt_unet": 8,
    "nab_gs": 9,
}
CODE_TO_MODEL_ID = {v: k for k, v in MODEL_ID_CODES.items()}


def case_to_feature_vector(record: CaseRecord) -> list[float]:
    """X_i: DICOM + başlangıç gürültü/kalite profili."""
    return [
        float(record.kvp),
        float(record.mas),
        float(record.slice_thickness),
        float(record.kernel_code),
        float(record.sigma_hu_before),
        float(record.gmm_peak_hu),
        float(record.niqe_before),
        float(record.ssim_inter),
    ]


def case_to_target_vector(record: CaseRecord) -> list[float]:
    """Y_i: onaylanan model + 5 kaydırıcı parametresi."""
    return [
        float(MODEL_ID_CODES.get(record.approved_model, 0)),
        float(record.alpha),
        float(record.gamma),
        float(record.lambda_tv),
        float(record.beta),
        float(record.kappa_hu),
    ]


class MetaOptimizer:
    """Arka planda kendini güncelleyen çoklu-çıktı parametre regresörü."""

    def __init__(self) -> None:
        self._model: MultiOutputRegressor | None = None
        self._lock = threading.Lock()
        self.trained_on = 0
        self.last_train_score = 0.0

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Eğitim
    # ------------------------------------------------------------------ #
    def maybe_retrain(self, session) -> bool:
        """Onaylı vaka sayısı eşiği geçtiyse yeniden eğitir; eğitildi mi döner."""
        records = session.query(CaseRecord).all()
        if len(records) < config.MIN_CASES_FOR_TRAINING:
            logger.info("Meta-regresör için yetersiz vaka: %d < %d",
                        len(records), config.MIN_CASES_FOR_TRAINING)
            return False
        return self.train(records)

    def train(self, records: list[CaseRecord]) -> bool:
        X = np.array([case_to_feature_vector(r) for r in records], dtype=np.float64)
        Y = np.array([case_to_target_vector(r) for r in records], dtype=np.float64)
        if X.shape[0] < 2:
            return False

        model = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0))
        model.fit(X, Y)
        score = float(model.score(X, Y))

        with self._lock:
            self._model = model
            self.trained_on = X.shape[0]
            self.last_train_score = score
        logger.info("Meta-regresör eğitildi: N=%d, R²(eğitim)=%.3f",
                    X.shape[0], score)
        return True

    # ------------------------------------------------------------------ #
    # Tahmin
    # ------------------------------------------------------------------ #
    def predict(self, features: list[float]) -> dict:
        """X_i -> Ŷ_i; parametreler geçerli aralıklara kırpılır."""
        with self._lock:
            model = self._model
        if model is None:
            raise RuntimeError(
                "Meta-regresör henüz eğitilmedi "
                f"(en az {config.MIN_CASES_FOR_TRAINING} onaylı vaka gerekli)"
            )
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(config.FEATURE_COLUMNS):
            raise ValueError(
                f"Özellik vektörü {len(config.FEATURE_COLUMNS)} boyutlu olmalı, "
                f"{x.shape[1]} geldi"
            )
        y = model.predict(x)[0]

        model_code = int(np.clip(round(y[0]), 0, max(CODE_TO_MODEL_ID)))
        while model_code not in CODE_TO_MODEL_ID and model_code > 0:
            model_code -= 1
        params = {}
        for value, name in zip(y[1:], ["alpha", "gamma", "lambda_tv", "beta", "kappa_hu"]):
            lo, hi = config.PARAM_RANGES[name]
            params[name] = float(np.clip(value, lo, hi))
        return {
            "model_id": CODE_TO_MODEL_ID.get(model_code, "rdn"),
            **params,
            "trained_on": self.trained_on,
            "train_score": round(self.last_train_score, 4),
        }

    def status(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "trained_on": self.trained_on,
            "train_score": round(self.last_train_score, 4),
            "min_cases_required": config.MIN_CASES_FOR_TRAINING,
        }
