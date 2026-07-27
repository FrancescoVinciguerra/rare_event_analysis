"""Adapter for the paper's official annealed transition-path sampler."""

from official_annealed_tps_experiment.annealed_tps import (
    OfficialAnnealedTPSConfig,
    OfficialAnnealedTPSResult,
    run_official_annealed_tps,
)

__all__ = [
    "OfficialAnnealedTPSConfig",
    "OfficialAnnealedTPSResult",
    "run_official_annealed_tps",
]
