"""Otomatik en iyi görüntü çıkarımı (denklem 2.2).

Aktif kesit, hızlı çıkarım havuzundaki modellerden geçirilir; her aday için:

    PI  = ((10 − NRQM) + NIQE) / 2            [labcisne/CT-Super-Resolution]
    CNR = (mu_kontrast − mu_doku) / sigma_HU  [RDN kontrast-gürültü oranı]

    S^(m) = w1·norm(1/PI) + w2·norm(CNR) − w3·norm(|mu_HU(Î) − mu_HU(I)|)

Üçüncü terim RDN/ContrastGAN'ın şart koştuğu mutlak HU atenüasyon sadakatini
cezalandırma olarak uygular. En yüksek skorlu modelin sonucu döndürülür.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture

from backend import config
from backend.core.phase_selector import (
    PerceptualScorer,
    measure_homogeneous_sigma,
)
from backend.engines.base_enhancer import BaseCTEnhancer, EnhanceResult

logger = logging.getLogger(__name__)


def estimate_cnr(slice_hu: np.ndarray) -> float:
    """GMM bileşenlerinden kontrast-gürültü oranı kestirimi.

    [-100, 1000] HU piksellerine 3 bileşenli GMM; en yüksek ve orta ortalamalı
    bileşenler kontrast lümeni ile yumuşak dokuyu temsil eder. CNR, bu iki
    ortalamanın farkının homojen ROI sigma'sına oranıdır.
    """
    lo, hi = config.GMM_HU_RANGE
    values = slice_hu[(slice_hu >= lo) & (slice_hu <= hi)].reshape(-1, 1)
    sigma = max(measure_homogeneous_sigma(slice_hu), 0.5)
    if values.shape[0] < 100:
        return 0.0
    if values.shape[0] > 100_000:
        values = values[:: values.shape[0] // 100_000]
    try:
        gmm = GaussianMixture(n_components=config.GMM_COMPONENTS,
                              covariance_type="full", random_state=0).fit(values)
    except Exception:
        return 0.0
    means = np.sort(gmm.means_.ravel())
    return float((means[-1] - means[-2]) / sigma)


@dataclass
class AutoBestCandidate:
    model_id: str
    result: EnhanceResult
    pi: float = 0.0
    cnr: float = 0.0
    hu_deviation: float = 0.0
    score: float = 0.0

    def public_state(self) -> dict:
        return {
            "model_id": self.model_id,
            "pi": round(self.pi, 4),
            "cnr": round(self.cnr, 4),
            "hu_deviation": round(self.hu_deviation, 4),
            "score": round(self.score, 4),
            "fallback": self.result.fallback,
            "runtime_s": round(self.result.runtime_s, 4),
            "sigma_before": round(self.result.sigma_before, 2),
            "sigma_after": round(self.result.sigma_after, 2),
        }


def _normalize(values: np.ndarray) -> np.ndarray:
    """[0, 1] min-maks normalizasyonu; sabit dizide 0.5 döner."""
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin < 1e-9:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def select_best_enhancement(
    slice_hu: np.ndarray,
    pool: dict[str, BaseCTEnhancer],
    scorer: PerceptualScorer | None = None,
) -> tuple[AutoBestCandidate, list[AutoBestCandidate]]:
    """Havuzdaki her modeli çalıştırır, S^(m) skoruyla en iyiyi seçer."""
    scorer = scorer or PerceptualScorer(use_pyiqa=True)
    mu_orig = _tissue_mean(slice_hu)

    candidates: list[AutoBestCandidate] = []
    for model_id, engine in pool.items():
        try:
            result = engine.enhance_slice(slice_hu)
        except Exception:
            logger.exception("auto_best: %s çalıştırılamadı", model_id)
            continue
        _, _, pi, _ = scorer.score_slices(result.i_dl[np.newaxis, ...])
        candidates.append(AutoBestCandidate(
            model_id=model_id,
            result=result,
            pi=pi,
            cnr=estimate_cnr(result.i_dl),
            hu_deviation=abs(_tissue_mean(result.i_dl) - mu_orig),
        ))

    if not candidates:
        raise RuntimeError("auto_best: hiçbir model sonuç üretemedi")

    w = config.AUTO_BEST_WEIGHTS
    inv_pi = _normalize(np.array([1.0 / max(c.pi, 1e-3) for c in candidates]))
    cnr = _normalize(np.array([c.cnr for c in candidates]))
    dev = _normalize(np.array([c.hu_deviation for c in candidates]))
    for i, c in enumerate(candidates):
        c.score = float(w["pi"] * inv_pi[i] + w["cnr"] * cnr[i]
                        - w["hu_fidelity"] * dev[i])

    best = max(candidates, key=lambda c: c.score)
    return best, candidates


def _tissue_mean(slice_hu: np.ndarray) -> float:
    mask = (slice_hu > -300.0) & (slice_hu < 300.0)
    if mask.sum() < 64:
        return float(np.mean(slice_hu))
    return float(np.mean(slice_hu[mask]))
