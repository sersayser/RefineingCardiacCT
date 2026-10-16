"""PhaseSelector testleri: S_faz skoru en iyi fazı seçmeli (test protokolü 3)."""

import numpy as np

from backend.core.ingestion import CaseIngestor
from backend.core.phase_selector import (
    PhaseSelector,
    gmm_contrast_peak,
    interslice_ssim,
    measure_homogeneous_sigma,
    mvf_magnitude,
)
from tests.conftest import make_ct_slice, make_volume


def test_homogeneous_sigma_tracks_noise():
    quiet = make_ct_slice(noise_sigma=8.0, seed=3)
    noisy = make_ct_slice(noise_sigma=35.0, seed=3)
    s_quiet = measure_homogeneous_sigma(quiet)
    s_noisy = measure_homogeneous_sigma(noisy)
    assert s_quiet < s_noisy
    assert 2.0 < s_quiet < 15.0
    assert 15.0 < s_noisy < 50.0


def test_gmm_peak_detects_contrast():
    contrast = make_volume(n_slices=3, noise_sigma=10.0, vessel_hu=450.0)
    plain = make_volume(n_slices=3, noise_sigma=10.0, vessel_hu=0.0)
    assert gmm_contrast_peak(contrast) >= 300.0
    assert gmm_contrast_peak(plain) < 300.0


def test_interslice_ssim_high_for_static_series():
    vol = make_volume(n_slices=5, noise_sigma=5.0, vessel_hu=300.0)
    assert interslice_ssim(vol) > 0.8


def test_mvf_magnitude_positive_for_motion():
    still = make_volume(n_slices=3, noise_sigma=5.0, vessel_hu=400.0, shift=0)
    moved = make_volume(n_slices=3, noise_sigma=5.0, vessel_hu=400.0, shift=8)
    assert mvf_magnitude(still, moved) > mvf_magnitude(still, still) >= 0.0


def test_recommends_clean_contrast_phase(multiphase_case_dir):
    """4 fazlı vakada temiz + kontrastlı + hareketsiz %75 fazı önerilmeli."""
    case = CaseIngestor().ingest(multiphase_case_dir)
    selector = PhaseSelector(use_pyiqa=False)  # hız için deterministik vekil
    results = selector.analyze_case(case)

    assert len(results) == 4
    recommended = [m for m in results if m.recommended]
    assert len(recommended) == 1
    assert recommended[0].phase_name == "phase_75"

    by_name = {m.phase_name: m for m in results}
    # Optimal kontrast kriteri: mu_peak >= 300 HU -> C_k = 1.0
    assert by_name["phase_75"].c_k == 1.0
    assert by_name["phase_75"].mu_peak_hu >= 300.0
    # Gürültü skoru: temiz faz, gürültülü fazdan yüksek N_k almalı
    assert by_name["phase_75"].n_k > by_name["phase_30"].n_k
    # Skorlar 0-100 bandında
    for m in results:
        assert 0.0 <= m.s_faz <= 100.0
