"""Artık alan (residual field) önbellekleme ve anlık harmanlama motoru.

Denklem 2.1'in tek doğruluk kaynağı. Derin öğrenme modeli bir kesiti BİR KEZ
işler; ΔI = Î_DL − I_orig ile türev haritaları önbelleğe alınır ve her
kaydırıcı değişimi yalnızca eleman bazlı vektörize işlemlerle (<10 ms, 512²)
yeniden harmanlanır:

    φ_γ(ΔI) = ΔI          (ΔI ≥ 0)
              γ · ΔI      (ΔI < 0)        [NAB-GS parametrik PReLU]

    I_out = clip( I_orig
                  + α · φ_γ(ΔI)
                  + β · ∇_lap(Î_DL)       [keskinleştirme: negatif Laplacian]
                  − λ_TV · TV(ΔI)         [yerel anizotropik TV haritası]
                  + κ_HU · ΔI_contrast,   [ContrastGAN lümen fark haritası]
                  −1024, 3071 )

Ön yüz (App.tsx) aynı denklemi HTML5 Canvas üzerinde 60 FPS uygular; backend
ile frontend arasında yalnızca I_orig, ΔI (ve varsa ΔI_contrast) taşınır.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from backend import config
from backend.config import BlendParams


@dataclass
class ResidualCache:
    """Bir kesit için önbelleğe alınan tüm harmanlama bileşenleri (float32)."""

    i_orig: np.ndarray          # orijinal kesit (HU)
    i_dl: np.ndarray            # model çıktısı (HU)
    delta: np.ndarray           # ΔI = Î_DL − I_orig
    lap_sharpen: np.ndarray     # ∇_lap(Î_DL): keskinleştirme operatörü (−Laplacian)
    tv_map: np.ndarray          # TV(ΔI): |∂x ΔI| + |∂y ΔI| yerel haritası
    delta_contrast: np.ndarray  # ΔI_contrast (ContrastGAN; yoksa sıfır)
    model_id: str = ""
    fallback: bool = False
    # HU sadakati skalerleri: doku maskesinde E[ΔI⁺] ve E[ΔI⁻]. φ_γ, γ<1 iken
    # negatif artıkları bastırdığından doku ortalamasını α·(m⁺+γ·m⁻) kadar
    # kaydırır; harmanlamada bu sabit geri çıkarılır (O(1), 60 FPS'e etkisiz).
    phi_bias_pos: float = 0.0
    phi_bias_neg: float = 0.0

    def payload_arrays(self) -> dict[str, np.ndarray]:
        """Ön yüze binary tampon olarak gönderilecek matrisler."""
        return {
            "i_orig": self.i_orig,
            "delta": self.delta,
            "lap_sharpen": self.lap_sharpen,
            "tv_map": self.tv_map,
            "delta_contrast": self.delta_contrast,
        }


class ResidualBlender:
    """ΔI önbelleği kurar ve kaydırıcı parametreleriyle anlık harmanlar."""

    @staticmethod
    def build_cache(
        i_orig: np.ndarray,
        i_dl: np.ndarray,
        delta_contrast: Optional[np.ndarray] = None,
        model_id: str = "",
        fallback: bool = False,
    ) -> ResidualCache:
        i_orig = np.ascontiguousarray(i_orig, dtype=np.float32)
        i_dl = np.ascontiguousarray(i_dl, dtype=np.float32)
        if i_orig.shape != i_dl.shape:
            raise ValueError(f"Boyut uyumsuz: {i_orig.shape} vs {i_dl.shape}")

        delta = i_dl - i_orig

        # Keskinleştirme operatörü: ∇_lap = −Laplacian(Î_DL). Pozitif β,
        # klasik unsharp-masking yönünde kenarları güçlendirir.
        lap_sharpen = -cv2.Laplacian(i_dl, cv2.CV_32F, ksize=3)

        # Yerel anizotropik TV haritası: ΔI'nin yüksek gradyanlı (granüler)
        # bölgelerini λ_TV ile orantılı bastırır.
        gy, gx = np.gradient(delta)
        tv_map = np.abs(gx, dtype=np.float32) + np.abs(gy, dtype=np.float32)

        # HU sadakati (CLAUDE.md kural 2): TV haritası her yerde >= 0 olduğu
        # için −λ_TV·TV terimi doku ortalamasını sistematik aşağı kaydırır;
        # aynı şekilde β·∇_lap kenar halkalarında ortalamayı oynatabilir.
        # Her iki harita doku maskesi üzerinde SIFIR ORTALAMALI hale getirilir:
        # kaydırıcılar yalnız yerel yapıyı değiştirir, mutlak atenüasyonu değil.
        # Ön yüz aynı tamponları kullandığından 60 FPS harmanlama da korunur.
        tissue = (i_orig > -300.0) & (i_orig < 300.0)
        if tissue.sum() > 64:
            tv_map -= np.float32(tv_map[tissue].mean())
            lap_sharpen -= np.float32(lap_sharpen[tissue].mean())
            d_t = delta[tissue]
            phi_bias_pos = float(np.sum(d_t[d_t > 0.0]) / d_t.size)
            phi_bias_neg = float(np.sum(d_t[d_t < 0.0]) / d_t.size)
        else:
            tv_map -= np.float32(tv_map.mean())
            lap_sharpen -= np.float32(lap_sharpen.mean())
            phi_bias_pos = float(np.sum(delta[delta > 0.0]) / delta.size)
            phi_bias_neg = float(np.sum(delta[delta < 0.0]) / delta.size)

        if delta_contrast is None:
            delta_contrast = np.zeros_like(i_orig)
        else:
            delta_contrast = np.ascontiguousarray(delta_contrast, dtype=np.float32)

        return ResidualCache(
            i_orig=i_orig, i_dl=i_dl, delta=delta,
            lap_sharpen=lap_sharpen.astype(np.float32),
            tv_map=tv_map.astype(np.float32),
            delta_contrast=delta_contrast,
            model_id=model_id, fallback=fallback,
            phi_bias_pos=phi_bias_pos, phi_bias_neg=phi_bias_neg,
        )

    @staticmethod
    def phi_gamma(delta: np.ndarray, gamma: float) -> np.ndarray:
        """NAB-GS parametrik PReLU: pozitif artık aynen, negatif γ ile ölçekli."""
        return np.where(delta >= 0.0, delta, np.float32(gamma) * delta)

    @staticmethod
    def blend(cache: ResidualCache, params: BlendParams) -> np.ndarray:
        """Denklem 2.1'i uygular; tamamen eleman bazlı, <10 ms hedefli."""
        out = cache.i_orig + np.float32(params.alpha) * ResidualBlender.phi_gamma(
            cache.delta, params.gamma
        )
        # φ_γ'nin doku ortalamasına verdiği sabit kaymayı geri al (HU sadakati)
        out -= np.float32(params.alpha
                          * (cache.phi_bias_pos + params.gamma * cache.phi_bias_neg))
        if params.beta > 0.0:
            out += np.float32(params.beta) * cache.lap_sharpen
        if params.lambda_tv > 0.0:
            out -= np.float32(params.lambda_tv) * cache.tv_map
        if params.kappa_hu > 0.0:
            out += np.float32(params.kappa_hu) * cache.delta_contrast
        return np.clip(out, config.HU_MIN, config.HU_MAX, out=out)

    @staticmethod
    def diff_map(cache: ResidualCache, i_out: np.ndarray) -> np.ndarray:
        """Fark görüntüleme paneli: |I_out − I_orig| mutlak HU değişimi."""
        return np.abs(i_out - cache.i_orig)

    @staticmethod
    def benchmark(cache: ResidualCache, params: BlendParams, repeats: int = 20) -> float:
        """Ortalama harmanlama süresini (saniye) ölçer; test protokolü 2 için."""
        start = time.perf_counter()
        for _ in range(repeats):
            ResidualBlender.blend(cache, params)
        return (time.perf_counter() - start) / repeats
