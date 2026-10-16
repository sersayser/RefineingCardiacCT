"""Motor kayıt defteri: model_id -> sarmalayıcı fabrikası.

Kesit bazlı motorlar (BaseCTEnhancer) senkron API yolunda kullanılır;
TemporalMAREnhancer (çok fazlı) ve NABGS3DWorker (asenkron 3B) ayrı
sözleşmelere sahiptir ve api/main.py tarafından doğrudan kurulur.
"""

from __future__ import annotations

from backend import config
from backend.engines.base_enhancer import BaseCTEnhancer, EnhanceResult
from backend.engines.contrast_gan_wrapper import ContrastGANEnhancer
from backend.engines.noise2sim_wrapper import Noise2SimAdapter, Noise2SimEnhancer
from backend.engines.rdn_wrapper import RDNEnhancer
from backend.engines.sr_benchmark_wrapper import (
    SR_VARIANT_PROFILES,
    SRBenchmarkEnhancer,
    build_sr_pool,
)

_SLICE_ENGINE_FACTORIES = {
    "rdn": RDNEnhancer,
    "noise2sim": Noise2SimEnhancer,
    "contrast_gan": ContrastGANEnhancer,
    **{variant: (lambda v: (lambda device="cpu": SRBenchmarkEnhancer(v, device=device)))(variant)
       for variant in SR_VARIANT_PROFILES},
}

SLICE_ENGINE_IDS = list(_SLICE_ENGINE_FACTORIES.keys())


def build_engine(model_id: str, device: str = "cpu") -> BaseCTEnhancer:
    """Kesit bazlı motoru kurar; bilinmeyen model_id -> ValueError."""
    factory = _SLICE_ENGINE_FACTORIES.get(model_id)
    if factory is None:
        raise ValueError(
            f"Bilinmeyen model: {model_id!r}; geçerli: {SLICE_ENGINE_IDS}"
        )
    return factory(device=device)


def build_fast_pool(device: str = "cpu") -> dict[str, BaseCTEnhancer]:
    """auto_best için hızlı çıkarım havuzu (config.FAST_MODEL_POOL)."""
    return {mid: build_engine(mid, device=device) for mid in config.FAST_MODEL_POOL}


__all__ = [
    "BaseCTEnhancer",
    "EnhanceResult",
    "RDNEnhancer",
    "Noise2SimEnhancer",
    "Noise2SimAdapter",
    "ContrastGANEnhancer",
    "SRBenchmarkEnhancer",
    "build_sr_pool",
    "build_engine",
    "build_fast_pool",
    "SLICE_ENGINE_IDS",
]
