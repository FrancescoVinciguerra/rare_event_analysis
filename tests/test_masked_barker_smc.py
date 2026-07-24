from __future__ import annotations

import numpy as np
import pytest
import torch

from masked_barker_smc_experiment.run_experiment import parse_args
from masked_barker_smc_experiment.smc_sampler import (
    MaskBatch,
    MaskedBarkerSMCConfig,
    barker_log_acceptance,
    completion_log_probabilities,
    generate_masked_proposal,
    masked_transition_log_probabilities,
    render_report,
    run_masked_barker_smc,
    sample_masks,
    symmetrization_log_acceptance,
    systematic_resample,
)
from rea_llms.observables.base import ObservableScore


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
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        batch, length = input_ids.shape
        prefix_sum = torch.cumsum(input_ids.to(torch.float32), dim=1)
        logits = torch.zeros((batch, length, 4), dtype=torch.float32, device=input_ids.device)
        logits[..., 0] = 0.2 * prefix_sum
        logits[..., 1] = -0.1 * prefix_sum
        logits[..., 2] = 0.3
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


def test_mask_validation_rejects_bad_schedules() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        MaskedBarkerSMCConfig(lambdas=(-0.5,))
    with pytest.raises(ValueError, match="sum to one"):
        MaskedBarkerSMCConfig(mask_rho_schedule=((0.5, 0.5, 0.5, 0.5),))
    with pytest.raises(ValueError, match="requires mask_mode"):
        MaskedBarkerSMCConfig(mask_policy="explicit", explicit_masks=(((0,),),))
    with pytest.raises(ValueError, match="zero-based"):
        MaskedBarkerSMCConfig(
            max_new_tokens=4,
            mask_mode="deterministic",
            mask_policy="explicit",
            explicit_masks=(((4,),),),
        )


def test_stochastic_and_contiguous_masks_have_requested_sizes() -> None:
    config = MaskedBarkerSMCConfig(
        num_particles=12,
        max_new_tokens=8,
        lambdas=(0.0, -0.5),
        mcmc_steps_per_level=1,
        mask_sizes=(2,),
        mask_rho_schedule=((1.0,),),
        mask_policy="contiguous",
    )
    masks = sample_masks(
        config,
        transition_index=0,
        step_index=0,
        device=torch.device("cpu"),
        generator=_generator(),
    )
    assert masks.positions.shape == (12, 2)
    assert torch.all(masks.sizes == 2)
    assert torch.all(masks.positions[:, 1] - masks.positions[:, 0] == 1)


def test_explicit_mask_schedule_is_deterministic() -> None:
    config = MaskedBarkerSMCConfig(
        num_particles=3,
        max_new_tokens=5,
        lambdas=(0.0, -0.5),
        mcmc_steps_per_level=2,
        mask_mode="deterministic",
        mask_policy="explicit",
        explicit_masks=(((3, 1), (4,)),),
    )
    first = sample_masks(
        config,
        transition_index=0,
        step_index=0,
        device=torch.device("cpu"),
        generator=_generator(),
    )
    second = sample_masks(
        config,
        transition_index=0,
        step_index=1,
        device=torch.device("cpu"),
        generator=_generator(),
    )
    assert first.positions.tolist() == [[1, 3], [1, 3], [1, 3]]
    assert second.positions.tolist() == [[4], [4], [4]]


def test_symmetrized_proposal_and_barker_satisfy_detailed_balance() -> None:
    model = PrefixDependentTinyModel()
    prompt_ids = torch.tensor([1], dtype=torch.long)
    x = torch.tensor([[0, 1, 2]], dtype=torch.long)
    y = torch.tensor([[3, 1, 0]], dtype=torch.long)
    masks = MaskBatch(
        positions=torch.tensor([[0, 2]], dtype=torch.long),
        valid=torch.tensor([[True, True]]),
        sizes=torch.tensor([2], dtype=torch.long),
    )
    log_r_xy = masked_transition_log_probabilities(
        model,
        prompt_ids,
        x,
        y,
        masks,
        temperature=0.7,
        batch_size=1,
    )
    log_r_yx = masked_transition_log_probabilities(
        model,
        prompt_ids,
        y,
        x,
        masks,
        temperature=0.7,
        batch_size=1,
    )
    log_k_xy = log_r_xy + symmetrization_log_acceptance(log_r_xy, log_r_yx)
    log_k_yx = log_r_yx + symmetrization_log_acceptance(log_r_yx, log_r_xy)
    assert torch.allclose(log_k_xy, log_k_yx)

    log_p_x = completion_log_probabilities(
        model, prompt_ids, x, batch_size=1, token_chunk_size=2
    )
    log_p_y = completion_log_probabilities(
        model, prompt_ids, y, batch_size=1, token_chunk_size=2
    )
    phi_x = torch.tensor([3.0])
    phi_y = torch.tensor([5.0])
    target_lambda = -0.5
    log_pi_x = log_p_x - target_lambda * phi_x
    log_pi_y = log_p_y - target_lambda * phi_y
    delta = log_pi_y - log_pi_x
    log_flow_xy = log_pi_x + log_k_xy + barker_log_acceptance(delta)
    log_flow_yx = log_pi_y + log_k_yx + barker_log_acceptance(-delta)
    assert torch.allclose(log_flow_xy, log_flow_yx, atol=1e-6)


def test_sampled_forward_probability_matches_exact_re_evaluation() -> None:
    model = PrefixDependentTinyModel()
    prompt_ids = torch.tensor([1], dtype=torch.long)
    source = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)
    masks = MaskBatch(
        positions=torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        valid=torch.tensor([[True, True], [True, True]]),
        sizes=torch.tensor([2, 2], dtype=torch.long),
    )
    proposal, sampled_log_r = generate_masked_proposal(
        model,
        prompt_ids,
        source,
        masks,
        temperature=0.8,
        batch_size=2,
        generator=_generator(12),
    )
    evaluated_log_r = masked_transition_log_probabilities(
        model,
        prompt_ids,
        source,
        proposal,
        masks,
        temperature=0.8,
        batch_size=2,
    )
    assert torch.allclose(sampled_log_r, evaluated_log_r, atol=1e-6)
    unmasked = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.equal(source[unmasked], proposal[unmasked])


def test_systematic_resampling_returns_valid_ancestors() -> None:
    log_weights = torch.log(torch.tensor([0.05, 0.15, 0.3, 0.5]))
    ancestors = systematic_resample(log_weights, generator=_generator())
    assert ancestors.shape == (4,)
    assert torch.all(ancestors[:-1] <= ancestors[1:])
    assert int(ancestors.min()) >= 0
    assert int(ancestors.max()) < 4


def test_lambda_zero_keeps_equal_weights_and_never_resamples() -> None:
    config = MaskedBarkerSMCConfig(
        num_particles=8,
        max_new_tokens=4,
        lambdas=(0.0, 0.0, 0.0),
        mcmc_steps_per_level=1,
        ess_threshold=0.99,
        mask_mode="deterministic",
        mask_sizes=(1,),
        deterministic_mask_size_schedule=((1,),),
        batch_size=8,
        seed=9,
    )
    result = run_masked_barker_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=config,
    )
    assert np.isclose(result.summary["final_ess_over_n"], 1.0)
    assert np.isclose(result.summary["min_ess_over_n"], 1.0)
    assert result.summary["resampling_steps"] == 0
    assert all(np.isclose(row["normalized_weight"], 1.0 / 8) for row in result.samples)


def test_incremental_weight_matches_lambda_update_without_resampling() -> None:
    config = MaskedBarkerSMCConfig(
        num_particles=6,
        max_new_tokens=3,
        lambdas=(0.0, -0.5),
        mcmc_steps_per_level=0,
        resampling=False,
        mask_mode="deterministic",
        mask_sizes=(1,),
        batch_size=6,
        seed=11,
        tail_thresholds=(3.0, 5.0),
    )
    result = run_masked_barker_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=config,
    )
    phi = np.asarray([row["phi"] for row in result.samples])
    expected = np.exp(0.5 * phi - np.max(0.5 * phi))
    expected /= expected.sum()
    actual = np.asarray([row["normalized_weight"] for row in result.samples])
    assert np.allclose(actual, expected)
    assert "mean_ari" not in result.summary
    assert np.isclose(result.summary["unweighted_mean_ari"], phi.mean())
    assert np.isclose(result.summary["weighted_mean_ari"], np.sum(expected * phi))
    assert result.summary["tail_counts"]["ari_ge_5"] == int(np.sum(phi >= 5.0))
    assert np.isclose(
        result.summary["tail_proportions"]["ari_ge_5"],
        np.mean(phi >= 5.0),
    )
    assert np.isclose(
        result.summary["weighted_tail_probabilities"]["ari_ge_5"],
        np.sum(expected[phi >= 5.0]),
    )

    final_level = result.diagnostics["level_diagnostics"][-1]
    assert np.isclose(
        final_level["weighted_mean_ari_before_resampling"],
        result.summary["weighted_mean_ari"],
    )
    assert np.isclose(
        final_level["unweighted_mean_ari_after_resampling"],
        result.summary["unweighted_mean_ari"],
    )


def test_low_ess_triggers_systematic_resampling_and_resets_weights() -> None:
    config = MaskedBarkerSMCConfig(
        num_particles=12,
        max_new_tokens=3,
        lambdas=(0.0, -20.0),
        mcmc_steps_per_level=0,
        ess_threshold=0.99,
        mask_mode="deterministic",
        mask_sizes=(1,),
        batch_size=12,
        seed=21,
    )
    result = run_masked_barker_smc(
        PrefixDependentTinyModel(),
        TinyTokenizer(),
        CompletionTokenSumObservable(),
        prompt="Once",
        config=config,
    )
    assert result.summary["resampling_steps"] == 1
    assert np.isclose(result.summary["final_ess_over_n"], 1.0)
    assert result.summary["min_ess_over_n"] < config.ess_threshold
    assert all(np.isclose(row["normalized_weight"], 1.0 / 12) for row in result.samples)
    assert np.isclose(
        result.summary["weighted_mean_ari"],
        result.summary["unweighted_mean_ari"],
    )
    final_level = result.diagnostics["level_diagnostics"][-1]
    assert final_level["resampled"]
    assert np.isclose(
        final_level["unweighted_mean_ari_after_resampling"],
        result.summary["unweighted_mean_ari"],
    )


def test_report_labels_weighted_and_unweighted_statistics() -> None:
    summary = {
        "runtime_total_seconds": 1.0,
        "runtime_seconds_per_final_particle": 0.25,
        "final_ess_over_n": 0.8,
        "min_ess_over_n": 0.5,
        "mean_mcmc_acceptance_rate": 0.3,
        "resampling_steps": 1,
        "unweighted_mean_ari": 7.0,
        "weighted_mean_ari": 8.0,
        "min_ari": 2.0,
        "max_ari": 12.0,
        "tail_counts": {"ari_ge_8": 2},
        "tail_proportions": {"ari_ge_8": 0.5},
        "weighted_tail_probabilities": {"ari_ge_8": 0.75},
    }
    diagnostics = {
        "level_diagnostics": [
            {
                "level": 0,
                "lambda": 0.0,
                "mutation_target_lambda": None,
                "ess_over_n_before_resampling": 1.0,
                "ess_over_n_after_resampling": 1.0,
                "resampled": False,
                "acceptance_rate": None,
                "weighted_mean_ari_before_resampling": 7.0,
                "unweighted_mean_ari_after_resampling": 7.0,
            }
        ]
    }
    report = render_report(summary, diagnostics)
    assert "Unweighted mean raw ARI" in report
    assert "Weighted mean raw ARI under final target" in report
    assert "weighted probability" in report
    assert "weighted mean ARI pre" in report


def test_cli_exposes_masked_barker_configuration() -> None:
    args = parse_args(
        [
            "--device",
            "cuda",
            "--mask-mode",
            "deterministic",
            "--deterministic-mask-size-schedule",
            "1,2;4,8",
            "--disable-resampling",
            "--run-name",
            "masked_unit",
        ]
    )
    assert args.device == "cuda"
    assert args.mask_mode == "deterministic"
    assert args.disable_resampling
    assert args.run_name == "masked_unit"
