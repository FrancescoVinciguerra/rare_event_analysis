from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

import suffix_direct_mh_smc_experiment.smc_sampler as sampler
from rea_llms.generation import fixed_completion_log_probabilities
from rea_llms.observables.base import ObservableScore
from suffix_direct_mh_smc_experiment.run_experiment import config_from_args, parse_args
from suffix_direct_mh_smc_experiment.smc_sampler import (
    SuffixDirectMHSMCConfig,
    incremental_log_weight_update,
    incremental_log_weight_update_with_normalizer,
    metropolis_log_acceptance,
    propose_suffixes_parallel,
    run_suffix_direct_mh_smc,
    sample_suffix_sizes,
)


class TinyTokenizer:
    pad_token_id = None

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {0: " a", 1: "Once", 2: " longer", 3: "."}
        return "".join(pieces.get(int(token), " x") for token in token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]


class PrefixDependentTinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, length = input_ids.shape
        prefix_sum = torch.cumsum(input_ids.to(torch.float32), dim=1)
        logits = torch.zeros(
            (batch, length, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., 0] = 0.2 * prefix_sum
        logits[..., 1] = -0.1 * prefix_sum
        logits[..., 2] = 0.3
        logits[..., 3] = -0.2
        return type("Output", (), {"logits": logits})


class CompletionTokenSumObservable:
    mode = "raw"

    def score_token_sequences(
        self,
        *,
        tokenizer,
        prompt,
        token_sequences,
        prompt_token_count,
    ):
        rows = []
        for ids in token_sequences:
            value = float(sum(ids[prompt_token_count:]))
            rows.append(
                ObservableScore(
                    value=value,
                    metadata={"raw_ari": value, "capped_ari": min(value, 15.0)},
                )
            )
        return rows


def _generator(seed: int = 7) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _config(**kwargs) -> SuffixDirectMHSMCConfig:
    defaults = {
        "num_particles": 8,
        "max_new_tokens": 3,
        "lambdas": (0.0, -0.5),
        "mcmc_steps_per_level": 1,
        "suffix_mode": "deterministic",
        "suffix_sizes": (3,),
        "suffix_size_schedule": ((3,),),
        "batch_size": 8,
        "seed": 11,
    }
    defaults.update(kwargs)
    return SuffixDirectMHSMCConfig(**defaults)


def test_config_requires_a_valid_state_independent_suffix_law() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        _config(lambdas=(-0.5, -1.0))
    with pytest.raises(ValueError, match="sum to one"):
        SuffixDirectMHSMCConfig(
            max_new_tokens=3,
            suffix_sizes=(1, 3),
            suffix_rhos=(0.8, 0.3),
        )
    with pytest.raises(ValueError, match="positive probability"):
        SuffixDirectMHSMCConfig(
            max_new_tokens=3,
            suffix_sizes=(1, 3),
            suffix_rhos=(1.0, 0.0),
        )


def test_suffix_sizes_are_sampled_for_all_particles_in_one_draw() -> None:
    config = SuffixDirectMHSMCConfig(
        num_particles=64,
        max_new_tokens=4,
        lambdas=(0.0, -0.5),
        suffix_sizes=(1, 2, 4),
        suffix_rhos=(0.2, 0.3, 0.5),
    )
    sizes = sample_suffix_sizes(
        config,
        transition_index=0,
        step_index=0,
        device=torch.device("cpu"),
        generator=_generator(19),
    )
    assert sizes.shape == (64,)
    assert set(sizes.tolist()) <= {1, 2, 4}


def test_batched_suffix_proposal_preserves_each_selected_prefix() -> None:
    current = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    suffix_sizes = torch.tensor([2, 4], dtype=torch.long)
    proposal = propose_suffixes_parallel(
        PrefixDependentTinyModel(),
        torch.tensor([1], dtype=torch.long),
        current,
        suffix_sizes,
        batch_size=2,
        generator=_generator(23),
    )
    assert proposal.shape == current.shape
    assert torch.equal(proposal[0, :2], current[0, :2])


def test_fixed_completion_log_probabilities_match_autoregressive_formula() -> None:
    model = PrefixDependentTinyModel()
    prompt = torch.tensor([1], dtype=torch.long)
    completions = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
    actual = fixed_completion_log_probabilities(
        model,
        prompt,
        completions,
        batch_size=2,
        position_batch_size=1,
    )
    expected = []
    for completion in completions:
        work = prompt.unsqueeze(0)
        value = 0.0
        for token in completion:
            logits = model(work).logits[:, -1].float()
            value += float(torch.log_softmax(logits, dim=-1)[0, token])
            work = torch.cat((work, token.reshape(1, 1)), dim=1)
        expected.append(value)
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-7)


def test_direct_acceptance_matches_the_complete_mh_ratio() -> None:
    log_r_xy = torch.tensor([-1.2, -4.0, -0.7])
    log_r_yx = torch.tensor([-2.0, -1.5, -3.1])
    phi_x = torch.tensor([1.0, 5.0, 3.0])
    phi_y = torch.tensor([3.0, 2.0, 3.5])
    target_lambda = -0.4
    direct = metropolis_log_acceptance(
        phi_x,
        phi_y,
        target_lambda=target_lambda,
    )
    # Full MH: log pi(y)-log pi(x) includes the base-LM suffix ratio;
    # adding log q(x|y)-log q(y|x) cancels it exactly.
    log_target_ratio = (
        log_r_xy - log_r_yx - target_lambda * (phi_y - phi_x)
    )
    full = torch.minimum(
        torch.zeros_like(phi_x),
        log_target_ratio + log_r_yx - log_r_xy,
    )
    assert torch.allclose(direct, full, atol=1e-7)
    assert torch.allclose(
        direct,
        torch.minimum(torch.zeros_like(phi_x), -target_lambda * (phi_y - phi_x)),
    )


def _next_probability(prefix: tuple[int, ...], token: int) -> float:
    total = sum((1,) + prefix)
    logits = np.asarray([0.2 * total, -0.1 * total], dtype=np.float64)
    probabilities = np.exp(logits - np.max(logits))
    return float(probabilities[token] / probabilities.sum())


def _sequence_probability(sequence: tuple[int, int]) -> float:
    return _next_probability((), sequence[0]) * _next_probability(
        (sequence[0],), sequence[1]
    )


def test_direct_suffix_kernel_is_stationary_on_enumerable_space() -> None:
    states = list(itertools.product(range(2), repeat=2))
    target_lambda = -0.5
    phi = np.asarray([sum(state) for state in states], dtype=np.float64)
    p_model = np.asarray([_sequence_probability(state) for state in states])
    target = p_model * np.exp(-target_lambda * phi)
    target /= target.sum()
    transition = np.zeros((len(states), len(states)), dtype=np.float64)

    for source_index, source in enumerate(states):
        for target_index, target_state in enumerate(states):
            if source_index == target_index:
                continue
            move_probability = 0.0
            for suffix_size, rho in ((1, 0.5), (2, 0.5)):
                prefix_size = 2 - suffix_size
                if source[:prefix_size] != target_state[:prefix_size]:
                    continue
                proposal_probability = (
                    _next_probability(source[:1], target_state[1])
                    if suffix_size == 1
                    else _sequence_probability(target_state)
                )
                log_acceptance = metropolis_log_acceptance(
                    torch.tensor([phi[source_index]]),
                    torch.tensor([phi[target_index]]),
                    target_lambda=target_lambda,
                )
                move_probability += rho * proposal_probability * float(
                    torch.exp(log_acceptance)
                )
            transition[source_index, target_index] = move_probability
        transition[source_index, source_index] = 1.0 - transition[source_index].sum()

    assert np.all(transition >= -1e-12)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(target @ transition, target, atol=1e-7)


def test_incremental_weight_update_is_the_adjacent_gamma_ratio() -> None:
    log_weights = torch.full((4,), -math.log(4), dtype=torch.float64)
    phi = torch.tensor([0.0, 1.0, 2.0, 3.0])
    actual = incremental_log_weight_update(
        log_weights,
        phi,
        previous_lambda=0.0,
        current_lambda=-0.5,
    )
    expected = 0.5 * phi.to(torch.float64)
    expected -= torch.logsumexp(expected, dim=0)
    assert torch.allclose(actual, expected)


def test_log_normalizing_constant_matches_exact_finite_space() -> None:
    base_probability = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    phi = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    log_weights = torch.log(base_probability)
    log_z = torch.zeros((), dtype=torch.float64)
    for previous_lambda, current_lambda in ((0.0, -0.5), (-0.5, -1.0)):
        log_weights, increment = incremental_log_weight_update_with_normalizer(
            log_weights,
            phi,
            previous_lambda=previous_lambda,
            current_lambda=current_lambda,
        )
        log_z += increment
    exact = torch.logsumexp(torch.log(base_probability) + phi, dim=0)
    assert torch.allclose(log_z, exact, atol=1e-12)
    exact_final = torch.log(base_probability) + phi - exact
    assert torch.allclose(log_weights, exact_final, atol=1e-12)


def test_lambda_zero_keeps_equal_weights_and_final_resamples() -> None:
    result = run_suffix_direct_mh_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=_config(lambdas=(0.0, 0.0), seed=31),
    )
    assert result.summary["final_ess_over_n"] == pytest.approx(1.0)
    assert result.summary["resample_final"] is True
    assert result.summary["resampling_steps"] == 1
    assert result.summary["log_normalizing_constant_estimate"] == pytest.approx(0.0)
    assert all(row["normalized_weight"] == pytest.approx(1 / 8) for row in result.samples)


def test_resampling_can_occur_before_and_at_final_level() -> None:
    result = run_suffix_direct_mh_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=_config(
            num_particles=24,
            lambdas=(0.0, -1.0, -20.0),
            mcmc_steps_per_level=0,
            ess_threshold=0.99,
        ),
    )
    levels = result.diagnostics["level_diagnostics"]
    assert levels[1]["resampled"]
    assert levels[2]["resampled"]
    assert result.summary["resample_final"] is True
    assert result.summary["final_ess_over_n"] == pytest.approx(1.0)
    assert result.summary["final_weight_ess_over_n_before_resampling"] < 0.99
    assert all(
        row["normalized_weight"] == pytest.approx(1 / 24)
        for row in result.samples
    )


def test_fixed_seed_run_is_reproducible() -> None:
    config = _config(
        num_particles=6,
        suffix_mode="stochastic",
        suffix_sizes=(1, 3),
        suffix_size_schedule=None,
        suffix_rhos=(0.5, 0.5),
        batch_size=3,
        seed=37,
    )
    runs = [
        run_suffix_direct_mh_smc(
            PrefixDependentTinyModel(),
            TinyTokenizer(),
            CompletionTokenSumObservable(),
            prompt="Once",
            config=config,
        )
        for _ in range(2)
    ]
    fields = ("completion_token_ids", "phi", "normalized_weight")
    assert [tuple(row[key] for key in fields) for row in runs[0].samples] == [
        tuple(row[key] for key in fields) for row in runs[1].samples
    ]


def test_optional_in_memory_level_trace_is_normalized_and_never_writes(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_suffix_direct_mh_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=_config(
            num_particles=8,
            suffix_mode="stochastic",
            suffix_sizes=(1, 3),
            suffix_size_schedule=None,
            suffix_rhos=(0.5, 0.5),
            seed=41,
        ),
        capture_level_particles=True,
        profile_runtime=True,
    )
    traces = result.diagnostics["level_particles"]
    assert len(traces) == 2
    assert list(tmp_path.iterdir()) == []
    for trace in traces:
        weights = torch.tensor(trace["normalized_weights"], dtype=torch.float64)
        assert float(weights.sum()) == pytest.approx(1.0)
        assert torch.isfinite(weights).all()
    level = result.diagnostics["level_diagnostics"][1]
    assert set(level["suffix_size_counts"]) == {"1", "3"}
    for size, rate in level["acceptance_rate_by_suffix_size"].items():
        assert 0.0 <= rate <= 1.0
        assert level["accepted_by_suffix_size"][size] <= level["suffix_size_counts"][size]
    assert result.summary["runtime_profile_enabled"] is True
    assert result.summary["generator_model_forward_calls"] > 0
    assert result.summary["generated_token_transitions"] >= 8 * 3


def test_active_sampler_has_no_balancing_or_reverse_scoring_path() -> None:
    assert not hasattr(sampler, "kernel_components")
    assert not hasattr(sampler, "paired_suffix_log_probabilities")
    args = parse_args(
        [
            "--max-new-tokens",
            "4",
            "--suffix-sizes",
            "1,4",
            "--suffix-rhos",
            "0.5,0.5",
        ]
    )
    config = config_from_args(args)
    assert isinstance(config, SuffixDirectMHSMCConfig)
    assert not hasattr(config, "balancing_function")
    assert not hasattr(config, "proposal_temperature")
