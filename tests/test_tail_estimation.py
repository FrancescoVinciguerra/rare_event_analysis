from __future__ import annotations

import math

import pytest
import torch

from rea_llms.tail_estimation import (
    corrected_base_log_weights,
    estimate_base_expectation,
    estimate_base_tail_probabilities,
    estimate_base_tails_from_final_samples,
)
from suffix_balanced_smc_experiment.run_tail_estimation import (
    _config_for_seed,
    parse_args,
)


def test_lambda_zero_preserves_normalized_smc_weights() -> None:
    phi = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
    smc_weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    corrected = torch.exp(corrected_base_log_weights(phi, smc_weights, 0.0))
    assert torch.allclose(corrected, smc_weights)
    expectation = estimate_base_expectation(
        phi,
        smc_weights,
        0.0,
        torch.tensor([1.0, 3.0, 7.0]),
    )
    assert float(expectation) == pytest.approx(4.6)


def test_corrected_weights_are_finite_nonnegative_and_normalized() -> None:
    corrected_log = corrected_base_log_weights(
        [0.0, 3.0, 10.0],
        [math.log(0.1), math.log(0.4), math.log(0.5)],
        -2.0,
        weights_are_log=True,
    )
    corrected = torch.exp(corrected_log)
    assert torch.isfinite(corrected).all()
    assert torch.all(corrected >= 0.0)
    assert float(corrected.sum()) == pytest.approx(1.0)
    estimates = estimate_base_tail_probabilities(
        [0.0, 3.0, 10.0],
        [math.log(0.1), math.log(0.4), math.log(0.5)],
        -2.0,
        [3.0],
        weights_are_log=True,
    )
    assert 0.0 < float(estimates.corrected_ess_over_n) <= 1.0


def test_tail_probabilities_are_monotone_for_sorted_thresholds() -> None:
    estimates = estimate_base_tail_probabilities(
        [1.0, 4.0, 8.0, 12.0],
        [0.1, 0.2, 0.3, 0.4],
        -0.5,
        [0.0, 4.0, 8.0, 20.0],
    )
    probabilities = estimates.estimated_probabilities
    assert torch.all(probabilities[:-1] >= probabilities[1:])
    assert estimates.raw_counts.tolist() == [4, 3, 2, 0]


def test_snis_converges_to_exact_base_probability_on_toy_space() -> None:
    base_probability = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    state_phi = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    lambda_final = -0.8
    tilted = base_probability * torch.exp(-lambda_final * state_phi)
    tilted /= tilted.sum()
    generator = torch.Generator().manual_seed(701)
    indices = torch.multinomial(tilted, 50_000, replacement=True, generator=generator)
    sampled_phi = state_phi.index_select(0, indices)
    estimates = estimate_base_tail_probabilities(
        sampled_phi,
        torch.ones(sampled_phi.numel(), dtype=torch.float64),
        lambda_final,
        [2.0],
    )
    exact_probability = float(base_probability[state_phi >= 2.0].sum())
    assert float(estimates.estimated_probabilities[0]) == pytest.approx(
        exact_probability,
        abs=0.015,
    )


def test_serialized_estimator_uses_final_lambda_and_final_log_weights_only() -> None:
    samples = [
        {
            "phi": 0.0,
            "log_weight": math.log(0.4),
            "normalized_weight": 0.99,
            "previous_log_weight": 50.0,
            "lambda_final": -1.0,
        },
        {
            "phi": 2.0,
            "log_weight": math.log(0.6),
            "normalized_weight": 0.01,
            "previous_log_weight": -50.0,
            "lambda_final": -1.0,
        },
    ]
    estimates = estimate_base_tails_from_final_samples(
        samples,
        lambda_final=-1.0,
        thresholds=[2.0],
    )
    expected = 0.6 * math.exp(-2.0) / (0.4 + 0.6 * math.exp(-2.0))
    assert float(estimates.estimated_probabilities[0]) == pytest.approx(expected)

    invalid = [dict(samples[0]), {**samples[1], "lambda_final": -0.5}]
    with pytest.raises(ValueError, match="final lambda"):
        estimate_base_tails_from_final_samples(
            invalid,
            lambda_final=-1.0,
            thresholds=[2.0],
        )


def test_tail_runner_builds_only_exact_direct_metropolis_smc() -> None:
    args = parse_args(
        [
            "--num-particles",
            "8",
            "--max-new-tokens",
            "4",
            "--lambdas",
            "0,-0.5",
            "--mcmc-steps-per-level",
            "1",
            "--mask-sizes",
            "1,4",
            "--mask-rho-schedule",
            "0.5,0.5",
        ]
    )
    config = _config_for_seed(
        args,
        seed=17,
        lambdas=(0.0, -0.5),
        thresholds=(2.0, 4.0),
    )
    assert config.balancing_function == "metropolis"
    assert config.metropolis_fast_path
    assert config.proposal_temperature == 1.0
    assert config.resampling
    assert config.implementation == "parallel"
