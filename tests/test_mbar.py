from __future__ import annotations

import numpy as np
import pytest

from official_annealed_tps_experiment.mbar import estimate_mbar_tails


def test_mbar_requires_an_unbiased_state() -> None:
    with pytest.raises(ValueError, match="lambda=0"):
        estimate_mbar_tails({-0.5: [1.0, 2.0]}, thresholds=[1.5])


def test_mbar_tail_estimates_are_monotone_and_finite() -> None:
    rng = np.random.default_rng(71)
    base = rng.normal(0.0, 1.0, 2_000)
    mildly_tilted = rng.normal(0.5, 1.0, 2_000)
    strongly_tilted = rng.normal(1.0, 1.0, 2_000)
    result = estimate_mbar_tails(
        {0.0: base, -0.5: mildly_tilted, -1.0: strongly_tilted},
        thresholds=[0.0, 1.0, 2.0],
    )
    probabilities = result.tails["probability"].to_numpy()
    assert np.isfinite(probabilities).all()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert np.all(probabilities[:-1] >= probabilities[1:])
    assert result.overlap_matrix.shape == (3, 3)
    assert result.log_normalizing_constants[0] == pytest.approx(0.0)
