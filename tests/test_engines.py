"""Motor sarmalayıcı testleri: yedek çekirdekler, HU sadakati, asenkron işçi."""

import numpy as np

from backend.core.phase_selector import PerceptualScorer, measure_homogeneous_sigma
from backend.engines import build_engine, build_fast_pool
from backend.engines.auto_best import estimate_cnr, select_best_enhancement
from backend.engines.nab_gs_3d import NABGS3DWorker
from backend.engines.temporal_mar_wrapper import TemporalMAREnhancer
from tests.conftest import make_ct_slice, make_volume


def test_rdn_fallback_reduces_sigma_and_preserves_mean():
    """Test protokolü 1: sigma düşer, doku ortalaması ±1.5 HU içinde kalır."""
    noisy = make_ct_slice(noise_sigma=30.0, vessel_hu=300.0, seed=7)
    result = build_engine("rdn").enhance_slice(noisy)

    assert result.fallback is True  # ağırlık yok -> deterministik yedek
    assert result.sigma_after < result.sigma_before * 0.6

    mask = (noisy > -300) & (noisy < 300)
    deviation = abs(float(np.mean(result.i_dl[mask]) - np.mean(noisy[mask])))
    assert deviation <= 1.5, f"HU sadakati ihlali: {deviation:.2f} HU"


def test_noise2sim_fallback_runs():
    noisy = make_ct_slice(noise_sigma=28.0, seed=11)
    result = build_engine("noise2sim").enhance_slice(noisy)
    assert result.fallback is True
    assert result.sigma_after < result.sigma_before


def test_contrast_gan_fallback_boosts_lumen_only():
    sl = make_ct_slice(noise_sigma=5.0, vessel_hu=350.0, seed=5)
    engine = build_engine("contrast_gan")
    delta = engine.contrast_delta(sl)
    lumen = sl >= 200.0
    background = (sl > -100) & (sl < 100)
    assert float(np.mean(delta[lumen])) > float(np.mean(delta[background]))


def test_sr_pool_variants_differ():
    sl = make_ct_slice(noise_sigma=25.0, vessel_hu=300.0, seed=9)
    outputs = {}
    for variant in ("srcnn", "hat"):
        outputs[variant] = build_engine(variant).enhance_slice(sl).i_dl
    # Farklı profiller ayrışan adaylar üretmeli (auto_best bunun üstünde çalışır)
    assert float(np.mean(np.abs(outputs["srcnn"] - outputs["hat"]))) > 0.1


def test_auto_best_selects_a_candidate():
    sl = make_ct_slice(noise_sigma=30.0, vessel_hu=350.0, seed=13)
    scorer = PerceptualScorer(use_pyiqa=False)
    best, candidates = select_best_enhancement(sl, build_fast_pool(), scorer=scorer)
    assert len(candidates) == 5
    assert best.score == max(c.score for c in candidates)
    assert best.result.i_dl.shape == sl.shape


def test_estimate_cnr_higher_with_contrast():
    contrast = make_ct_slice(noise_sigma=10.0, vessel_hu=450.0, seed=2)
    plain = make_ct_slice(noise_sigma=10.0, vessel_hu=0.0, seed=2)
    assert estimate_cnr(contrast) > estimate_cnr(plain)


def test_temporal_mar_fallback_and_mvf():
    phases = {
        "p30": make_volume(n_slices=3, noise_sigma=20.0, vessel_hu=400.0, shift=6),
        "p75": make_volume(n_slices=3, noise_sigma=20.0, vessel_hu=400.0, shift=0),
    }
    mar = TemporalMAREnhancer()
    result = mar.enhance_series(phases, target_phase="p75")

    assert result.fallback is True
    assert result.corrected.shape == phases["p75"].shape
    assert set(result.mvf_per_phase) == {"p30", "p75"}
    assert result.mvf_per_phase["p30"] > 0.0
    # Zamansal ortalama gürültüyü düşürmeli
    s_before = measure_homogeneous_sigma(phases["p75"][1])
    s_after = measure_homogeneous_sigma(result.corrected[1])
    assert s_after < s_before


def test_nab_gs_fallback_job_upscales():
    worker = NABGS3DWorker()
    vol = make_volume(n_slices=4, shape=(64, 64), noise_sigma=10.0)
    job = worker.submit(scale=2)

    stages = []
    worker.run_job(job.job_id, vol,
                   progress_cb=lambda jid, frac, stage: stages.append(stage))

    assert job.status == "done"
    assert job.fallback is True
    assert job.result.shape == (8, 64, 64)  # Z ekseni 2x
    assert float(job.result.min()) >= -512.0  # NAB-GS HU aralığı
    assert job.progress == 1.0
    assert len(stages) > 0
