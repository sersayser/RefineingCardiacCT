"""ResidualBlender testleri: denklem 2.1 doğruluğu ve <10 ms performans."""

import numpy as np

from backend.config import BlendParams
from backend.core.residual_blender import ResidualBlender
from tests.conftest import make_ct_slice


def _cache(shape=(128, 128)):
    i_orig = make_ct_slice(shape, noise_sigma=30.0, vessel_hu=300.0, seed=1)
    i_dl = make_ct_slice(shape, noise_sigma=5.0, vessel_hu=300.0, seed=1)
    return ResidualBlender.build_cache(i_orig, i_dl)


def test_phi_gamma_prelu():
    delta = np.array([-10.0, -2.0, 0.0, 3.0, 8.0], dtype=np.float32)
    out = ResidualBlender.phi_gamma(delta, gamma=0.25)
    np.testing.assert_allclose(out, [-2.5, -0.5, 0.0, 3.0, 8.0])


def test_alpha_one_other_zero_reproduces_phi_blend():
    cache = _cache()
    params = BlendParams(alpha=1.0, gamma=1.0, beta=0.0, lambda_tv=0.0, kappa_hu=0.0)
    out = ResidualBlender.blend(cache, params)
    # gamma=1 -> phi = delta; alpha=1 -> I_DL, eksi doku-ortalama düzeltmesi
    bias = cache.phi_bias_pos + cache.phi_bias_neg  # = E[ΔI | doku]
    np.testing.assert_allclose(out, np.clip(cache.i_dl - bias, -1024, 3071),
                               atol=1e-3)
    assert abs(bias) < 1.5  # model zaten HU sadakati korumalı


def test_alpha_zero_returns_original():
    cache = _cache()
    params = BlendParams(alpha=0.0, beta=0.0, lambda_tv=0.0, kappa_hu=0.0)
    out = ResidualBlender.blend(cache, params)
    np.testing.assert_allclose(out, cache.i_orig, atol=1e-4)


def test_output_clipped_to_hu_bounds():
    i_orig = np.full((32, 32), 3000.0, dtype=np.float32)
    i_dl = np.full((32, 32), 3500.0, dtype=np.float32)
    cache = ResidualBlender.build_cache(i_orig, i_dl)
    out = ResidualBlender.blend(cache, BlendParams(alpha=1.5))
    assert float(out.max()) <= 3071.0
    assert float(out.min()) >= -1024.0


def test_param_ranges_are_enforced():
    p = BlendParams(alpha=99.0, gamma=-5.0, beta=2.0, lambda_tv=1.0, kappa_hu=10.0)
    assert p.alpha == 1.5
    assert p.gamma == 0.01
    assert p.beta == 1.0
    assert p.lambda_tv == 0.5
    assert p.kappa_hu == 2.0


def test_diff_map_is_absolute():
    cache = _cache()
    out = ResidualBlender.blend(cache, BlendParams(alpha=1.0))
    diff = ResidualBlender.diff_map(cache, out)
    assert float(diff.min()) >= 0.0
    np.testing.assert_allclose(diff, np.abs(out - cache.i_orig))


def test_all_sliders_preserve_tissue_mean():
    """HU sadakati: λ_TV ve β aktifken bile doku ortalaması ±1.5 HU içinde.

    TV ve Laplacian haritaları doku maskesinde sıfır ortalamalı tutulduğundan
    kaydırıcılar yalnız yerel yapıyı değiştirir, mutlak atenüasyonu değil.
    (κ_HU hariç: kontrast kazancı tanımı gereği lümen HU'sunu yükseltir.)
    """
    i_orig = make_ct_slice((256, 256), noise_sigma=25.0, vessel_hu=450.0, seed=21)
    i_dl = make_ct_slice((256, 256), noise_sigma=6.0, vessel_hu=450.0, seed=21)
    cache = ResidualBlender.build_cache(i_orig, i_dl)
    mask = (i_orig > -300.0) & (i_orig < 300.0)

    for params in (
        BlendParams(alpha=1.0, gamma=0.25, lambda_tv=0.25, beta=0.0, kappa_hu=0.0),
        BlendParams(alpha=1.3, gamma=0.80, lambda_tv=0.50, beta=0.0, kappa_hu=0.0),
        BlendParams(alpha=0.6, gamma=0.10, lambda_tv=0.25, beta=0.5, kappa_hu=0.0),
    ):
        out = ResidualBlender.blend(cache, params)
        shift = abs(float(np.mean(out[mask]) - np.mean(i_orig[mask])))
        assert shift <= 1.5, (
            f"HU sadakati ihlali: {shift:.2f} HU @ {params.as_dict()}"
        )


def test_blend_under_10ms_for_512():
    """Test protokolü 2: 512² kesitte harmanlama < 10 ms olmalı."""
    cache = _cache(shape=(512, 512))
    params = BlendParams(alpha=1.2, gamma=0.3, beta=0.4, lambda_tv=0.2, kappa_hu=0.5)
    mean_s = ResidualBlender.benchmark(cache, params, repeats=10)
    assert mean_s < 0.010, f"Harmanlama çok yavaş: {mean_s * 1000:.2f} ms"
