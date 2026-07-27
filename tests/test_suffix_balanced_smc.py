from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rea_llms.observables.base import ObservableScore
import suffix_balanced_smc_experiment.smc_sampler as suffix_sampler
from suffix_balanced_smc_experiment.run_experiment import parse_args
from suffix_balanced_smc_experiment.smc_sampler import (
    SuffixBalancedSMCConfig,
    incremental_log_weight_update,
    kernel_components,
    metropolis_log_acceptance_fast,
    metropolis_log_acceptance_full,
    paired_suffix_log_probabilities,
    propose_suffixes_parallel,
    run_suffix_balanced_smc,
    safe_sqrt_scale,
    sample_suffix_sizes,
    score_fixed_suffix_proposals,
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
        if self.vocab_size > 2:
            logits[..., 2] = 0.3
        if self.vocab_size > 3:
            logits[..., 3] = -0.2
        return type("Output", (), {"logits": logits})


class CompletionTokenSumObservable:
    name = "completion_token_sum"

    def score_token_sequences(self, *, tokenizer, prompt, token_sequences, prompt_token_count):
        rows = []
        for ids in token_sequences:
            value = float(sum(ids[prompt_token_count:]))
            rows.append(
                ObservableScore(
                    value=value,
                    metadata={"raw_ari": value, "capped_ari": value},
                )
            )
        return rows


def _generator(seed: int = 7) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _config(**kwargs) -> SuffixBalancedSMCConfig:
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
    return SuffixBalancedSMCConfig(**defaults)


def _manual_suffix_log_probability(
    model,
    context: torch.Tensor,
    suffix: torch.Tensor,
) -> float:
    work = context.clone()
    total = 0.0
    for token in suffix.tolist():
        logits = model(work).logits[0, -1]
        total += float(F.log_softmax(logits, dim=-1)[int(token)].item())
        work = torch.cat((work, torch.tensor([[token]], dtype=torch.long)), dim=1)
    return total


def test_configuration_requires_temperature_one_and_global_suffix_support() -> None:
    with pytest.raises(ValueError, match="proposal_temperature=1.0"):
        _config(proposal_temperature=0.8)
    with pytest.raises(ValueError, match="positive probability"):
        SuffixBalancedSMCConfig(
            max_new_tokens=4,
            suffix_sizes=(1, 4),
            suffix_rhos=(1.0, 0.0),
        )
    local = SuffixBalancedSMCConfig(
        max_new_tokens=4,
        suffix_sizes=(1,),
        suffix_rhos=(1.0,),
        require_global_irreducibility=False,
    )
    assert local.suffix_sizes == (1,)


def test_suffix_proposal_changes_only_the_selected_suffix() -> None:
    model = PrefixDependentTinyModel()
    prompt_ids = torch.tensor([1], dtype=torch.long)
    current = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    sizes = torch.tensor([2, 2], dtype=torch.long)
    proposal, _, _ = propose_suffixes_parallel(
        model,
        prompt_ids,
        current,
        sizes,
        batch_size=2,
        token_chunk_size=2,
        generator=_generator(19),
    )
    assert torch.equal(proposal[:, :2], current[:, :2])

    sampled = sample_suffix_sizes(
        SuffixBalancedSMCConfig(
            num_particles=64,
            max_new_tokens=4,
            suffix_sizes=(1, 2, 4),
            suffix_rhos=(0.3, 0.3, 0.4),
        ),
        transition_index=0,
        step_index=0,
        device=torch.device("cpu"),
        generator=_generator(20),
    )
    assert set(sampled.tolist()) <= {1, 2, 4}


def test_forward_and_reverse_suffix_probabilities_match_exact_enumeration() -> None:
    model = PrefixDependentTinyModel()
    context = torch.tensor([[1, 0], [1, 3]], dtype=torch.long)
    proposed = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
    old = torch.tensor([[1, 0], [2, 2]], dtype=torch.long)
    forward, reverse = paired_suffix_log_probabilities(
        model,
        context,
        proposed,
        old,
        batch_size=2,
        token_chunk_size=1,
    )
    expected_forward = torch.tensor(
        [
            _manual_suffix_log_probability(model, context[index : index + 1], proposed[index])
            for index in range(2)
        ]
    )
    expected_reverse = torch.tensor(
        [
            _manual_suffix_log_probability(model, context[index : index + 1], old[index])
            for index in range(2)
        ]
    )
    assert torch.allclose(forward, expected_forward, atol=1e-6)
    assert torch.allclose(reverse, expected_reverse, atol=1e-6)


def test_parallel_scoring_matches_particlewise_reference_values() -> None:
    model = PrefixDependentTinyModel()
    prompt_ids = torch.tensor([1], dtype=torch.long)
    current = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]], dtype=torch.long
    )
    proposals = torch.tensor(
        [[0, 1, 3, 0], [3, 2, 2, 3], [1, 0, 0, 1]], dtype=torch.long
    )
    sizes = torch.tensor([2, 2, 3], dtype=torch.long)
    parallel_forward, parallel_reverse = score_fixed_suffix_proposals(
        model,
        prompt_ids,
        current,
        proposals,
        sizes,
        batch_size=3,
        token_chunk_size=2,
    )
    reference_forward = []
    reference_reverse = []
    for index, size in enumerate(sizes.tolist()):
        prefix_size = current.shape[1] - size
        context = torch.cat(
            (prompt_ids.unsqueeze(0), current[index : index + 1, :prefix_size]), dim=1
        )
        reference_forward.append(
            _manual_suffix_log_probability(model, context, proposals[index, prefix_size:])
        )
        reference_reverse.append(
            _manual_suffix_log_probability(model, context, current[index, prefix_size:])
        )
    assert torch.allclose(parallel_forward, torch.tensor(reference_forward), atol=1e-6)
    assert torch.allclose(parallel_reverse, torch.tensor(reference_reverse), atol=1e-6)


@pytest.mark.parametrize("balancing", ["barker", "sqrt"])
def test_symmetrized_balancing_satisfies_detailed_balance(balancing: str) -> None:
    log_r_xy = torch.tensor([-1.2])
    log_r_yx = torch.tensor([-2.0])
    phi_x = torch.tensor([1.0])
    phi_y = torch.tensor([3.0])
    target_lambda = -0.4
    kwargs = {"phi_min": 0.0, "phi_max": 4.0} if balancing == "sqrt" else {}
    xy = kernel_components(
        log_r_xy,
        log_r_yx,
        phi_x,
        phi_y,
        target_lambda=target_lambda,
        balancing_function=balancing,
        **kwargs,
    )
    yx = kernel_components(
        log_r_yx,
        log_r_xy,
        phi_y,
        phi_x,
        target_lambda=target_lambda,
        balancing_function=balancing,
        **kwargs,
    )
    log_pi_x = torch.tensor([0.0])
    log_pi_y = log_pi_x + xy.log_target_ratio
    log_flow_xy = log_pi_x + log_r_xy + xy.log_acceptance
    log_flow_yx = log_pi_y + log_r_yx + yx.log_acceptance
    log_symmetric_kernel_xy = log_r_xy + xy.log_symmetrization
    log_symmetric_kernel_yx = log_r_yx + yx.log_symmetrization
    assert torch.allclose(log_symmetric_kernel_xy, log_symmetric_kernel_yx, atol=1e-6)
    assert torch.allclose(log_flow_xy, log_flow_yx, atol=1e-6)


def test_metropolis_fast_acceptance_matches_full_direct_mh_ratio() -> None:
    log_r_xy = torch.tensor([-1.2, -4.0, -0.7])
    log_r_yx = torch.tensor([-2.0, -1.5, -3.1])
    phi_x = torch.tensor([1.0, 5.0, 3.0])
    phi_y = torch.tensor([3.0, 2.0, 3.5])
    target_lambda = -0.4
    fast = metropolis_log_acceptance_fast(
        phi_x,
        phi_y,
        target_lambda=target_lambda,
    )
    full = metropolis_log_acceptance_full(
        log_r_xy,
        log_r_yx,
        phi_x,
        phi_y,
        target_lambda=target_lambda,
    )
    expected = torch.minimum(
        torch.zeros_like(phi_x),
        -target_lambda * (phi_y - phi_x),
    )
    log_common_prefix = torch.tensor([-3.2, -1.1, -5.0])
    log_suffix_selection = math.log(0.2)
    log_pi_x = log_common_prefix + log_r_yx - target_lambda * phi_x
    log_pi_y = log_common_prefix + log_r_xy - target_lambda * phi_y
    log_q_xy = log_suffix_selection + log_r_xy
    log_q_yx = log_suffix_selection + log_r_yx
    explicit_full_mh = torch.minimum(
        torch.zeros_like(phi_x),
        log_pi_y + log_q_yx - log_pi_x - log_q_xy,
    )
    assert torch.allclose(fast, full, atol=1e-7)
    assert torch.allclose(fast, expected, atol=1e-7)
    assert torch.allclose(fast, explicit_full_mh, atol=1e-6)


def test_metropolis_fast_path_does_not_score_reverse_probability(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("reverse suffix probability must not be evaluated")

    monkeypatch.setattr(suffix_sampler, "paired_suffix_log_probabilities", fail_if_called)
    result = run_suffix_balanced_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=_config(
            balancing_function="metropolis",
            lambdas=(0.0, -0.5, -1.0),
            seed=41,
        ),
    )
    assert result.summary["metropolis_fast_path"] is True
    assert result.diagnostics["metropolis_fast_path"] is True
    assert result.diagnostics["level_diagnostics"][1][
        "mean_symmetrization_probability"
    ] is None


def test_acceptance_components_are_finite_probabilities() -> None:
    generator = _generator(29)
    log_forward = -8.0 * torch.rand(100, generator=generator)
    log_reverse = -8.0 * torch.rand(100, generator=generator)
    current_phi = 15.0 * torch.rand(100, generator=generator)
    proposal_phi = 15.0 * torch.rand(100, generator=generator)
    for balancing in ("barker", "sqrt"):
        kwargs = {"phi_min": 0.0, "phi_max": 15.0} if balancing == "sqrt" else {}
        components = kernel_components(
            log_forward,
            log_reverse,
            current_phi,
            proposal_phi,
            target_lambda=-2.0,
            balancing_function=balancing,
            **kwargs,
        )
        probabilities = torch.exp(components.log_acceptance)
        assert torch.isfinite(probabilities).all()
        assert torch.all(probabilities >= 0.0)
        assert torch.all(probabilities <= 1.0)
    metropolis = torch.exp(
        metropolis_log_acceptance_fast(
            current_phi,
            proposal_phi,
            target_lambda=-2.0,
        )
    )
    assert torch.isfinite(metropolis).all()
    assert torch.all((metropolis >= 0.0) & (metropolis <= 1.0))
    assert safe_sqrt_scale(-2.0, 0.0, 15.0) == pytest.approx(math.exp(-15.0))


def _next_probability(prefix: tuple[int, ...], token: int) -> float:
    total = sum((1,) + prefix)
    logits = np.asarray([0.2 * total, -0.1 * total], dtype=np.float64)
    probabilities = np.exp(logits - np.max(logits))
    return float(probabilities[token] / probabilities.sum())


def _sequence_probability(sequence: tuple[int, int]) -> float:
    return _next_probability((), sequence[0]) * _next_probability((sequence[0],), sequence[1])


@pytest.mark.parametrize("balancing", ["barker", "metropolis", "sqrt"])
def test_tilted_distribution_is_stationary_on_enumerable_space(balancing: str) -> None:
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
            probability = 0.0
            for suffix_size, rho in ((1, 0.5), (2, 0.5)):
                prefix_size = 2 - suffix_size
                if source[:prefix_size] != target_state[:prefix_size]:
                    continue
                if suffix_size == 1:
                    r_forward = _next_probability(source[:1], target_state[1])
                    r_reverse = _next_probability(source[:1], source[1])
                else:
                    r_forward = _sequence_probability(target_state)
                    r_reverse = _sequence_probability(source)
                if balancing == "metropolis":
                    log_acceptance = metropolis_log_acceptance_fast(
                        torch.tensor([phi[source_index]]),
                        torch.tensor([phi[target_index]]),
                        target_lambda=target_lambda,
                    )
                else:
                    kwargs = {"phi_min": 0.0, "phi_max": 2.0} if balancing == "sqrt" else {}
                    components = kernel_components(
                        torch.tensor([math.log(r_forward)]),
                        torch.tensor([math.log(r_reverse)]),
                        torch.tensor([phi[source_index]]),
                        torch.tensor([phi[target_index]]),
                        target_lambda=target_lambda,
                        balancing_function=balancing,
                        **kwargs,
                    )
                    log_acceptance = components.log_acceptance
                probability += rho * r_forward * float(torch.exp(log_acceptance))
            transition[source_index, target_index] = probability
        transition[source_index, source_index] = 1.0 - transition[source_index].sum()

    assert np.all(transition >= -1e-12)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(target @ transition, target, atol=1e-7)


def test_incremental_weight_update_matches_gamma_ratio() -> None:
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


def test_lambda_zero_keeps_uniform_weights_and_full_ess() -> None:
    result = run_suffix_balanced_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=_config(lambdas=(0.0, 0.0), seed=31),
    )
    assert result.summary["final_ess_over_n"] == pytest.approx(1.0)
    assert result.summary["resampling_steps"] == 0
    assert all(
        row["normalized_weight"] == pytest.approx(1.0 / 8)
        for row in result.samples
    )


def test_resampling_uses_threshold_but_never_resamples_final_level() -> None:
    result = run_suffix_balanced_smc(
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
    assert levels[1]["ess_over_n_before_resampling"] < 0.99
    assert not levels[2]["resampled"]
    assert levels[2]["ess_over_n_before_resampling"] < 0.99
    assert result.summary["resampling_steps"] == 1
    assert result.summary["final_ess_over_n"] < 0.99


def test_cpu_smoke_is_reproducible_with_fixed_seed() -> None:
    config = _config(
        num_particles=6,
        max_new_tokens=3,
        lambdas=(0.0, -0.5),
        suffix_sizes=(1, 3),
        suffix_mode="stochastic",
        suffix_size_schedule=None,
        suffix_rhos=(0.5, 0.5),
        batch_size=3,
        seed=37,
    )
    first = run_suffix_balanced_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=config,
    )
    second = run_suffix_balanced_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=config,
    )
    first_rows = [
        (row["completion_token_ids"], row["phi"], row["normalized_weight"])
        for row in first.samples
    ]
    second_rows = [
        (row["completion_token_ids"], row["phi"], row["normalized_weight"])
        for row in second.samples
    ]
    assert first_rows == second_rows
    assert first.summary["resample_final"] is False


def test_cli_exposes_suffix_balanced_configuration() -> None:
    args = parse_args(
        [
            "--device",
            "cuda",
            "--suffix-sizes",
            "1,2,4,8",
            "--suffix-rhos",
            "0.4,0.3,0.2,0.1",
            "--balancing-function",
            "metropolis",
            "--disable-metropolis-fast-path",
            "--batch-size",
            "16",
        ]
    )
    assert args.device == "cuda"
    assert args.suffix_sizes == "1,2,4,8"
    assert args.balancing_function == "metropolis"
    assert args.disable_metropolis_fast_path
    assert args.batch_size == 16
