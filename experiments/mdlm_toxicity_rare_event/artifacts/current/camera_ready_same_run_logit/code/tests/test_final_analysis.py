from __future__ import annotations

import math

import pytest

from diffusion_toxicity_smc_experiment.final_analysis import (
    _with_run_offset,
    bootstrap_ratio,
    normalizer_consistency,
)


def test_independent_cohorts_receive_unique_run_indices() -> None:
    rows = [{"run_index": 0, "value": 1}, {"run_index": 1, "value": 2}]
    shifted = _with_run_offset(rows, offset=32, cohort="confirmatory")
    assert [row["run_index"] for row in shifted] == [32, 33]
    assert [row["source_run_index"] for row in shifted] == [0, 1]
    assert {row["cohort"] for row in shifted} == {"confirmatory"}
    assert rows[0] == {"run_index": 0, "value": 1}


def test_normalizer_consistency_uses_terminal_smc_weights() -> None:
    # pi gives probabilities proportional to exp(lambda * reward) P0.
    rewards = [0.0, math.log(2.0)]
    final_probabilities = [1.0 / 3.0, 2.0 / 3.0]
    log_z = math.log(1.5)
    summary = {"lambda": 1.0, "log_normalizer_estimate": log_z}
    particles = [
        {"final_log_weight": math.log(probability), "reward": reward}
        for probability, reward in zip(final_probabilities, rewards)
    ]
    result = normalizer_consistency(summary, particles)
    assert result["log_identity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["multiplicative_identity"] == pytest.approx(1.0)


def test_normalizer_consistency_rejects_nonfinite_values() -> None:
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        normalizer_consistency(
            {"lambda": 1.0, "log_normalizer_estimate": 0.0},
            [{"final_log_weight": 0.0, "reward": math.nan}],
        )


def test_bootstrap_ratio_is_run_level_and_reproducible() -> None:
    first = bootstrap_ratio(
        [0.1, -0.1, 0.2, -0.2],
        [0.2, -0.2, 0.4, -0.4],
        numerator_runtime=2.0,
        denominator_runtime=1.0,
        replicates=2000,
        seed=17,
    )
    second = bootstrap_ratio(
        [0.1, -0.1, 0.2, -0.2],
        [0.2, -0.2, 0.4, -0.4],
        numerator_runtime=2.0,
        denominator_runtime=1.0,
        replicates=2000,
        seed=17,
    )
    assert first == second
    assert first["rmse_ratio"] == pytest.approx(0.5)
    assert first["mse_times_runtime_ratio"] == pytest.approx(0.5)
