from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace
import tomllib
from pathlib import Path

import pytest
import torch
from setuptools import find_packages

import diffusion_toxicity_smc_experiment.generate_direct as generate_direct_module

from diffusion_toxicity_smc_experiment.ancestral import (
    NativeMDLMSampler,
    subs_log_probabilities,
)
from diffusion_toxicity_smc_experiment.calibration_report import (
    build_calibration_report,
)
from diffusion_toxicity_smc_experiment.compare import compare_existing
from diffusion_toxicity_smc_experiment.configuration import (
    decoder_id,
    decoder_manifest,
    evaluator_id,
    fingerprint,
    load_yaml,
    validate_benchmark_config,
)
from diffusion_toxicity_smc_experiment.evaluation import (
    ToxicityEvaluator,
    direct_unconditional_summary,
    identical_classifier_inputs,
    unconditional_mdlm_prefix_classifier_text,
    unconditional_trimmed_classifier_text,
)
from diffusion_toxicity_smc_experiment.generate_direct import (
    _prefetched_classifier_dtype,
    direct_run_directory,
    generate_missing,
)
from diffusion_toxicity_smc_experiment.run_smc import (
    UnconditionalToxicityFunctions,
    _aggregate,
    _component_seed,
    smc_manifest,
    smc_run_directory,
)
from diffusion_toxicity_smc_experiment.serious_comparison import (
    build_comparison_report,
    comparison_manifest,
    comparison_run_directory,
    load_suite,
    resolved_comparison_entries,
    run_comparison_suite,
)
from diffusion_toxicity_smc_experiment.runtime import load_generator_runtime
from diffusion_toxicity_smc_experiment.score_direct import (
    _write_summary_snapshot,
    latest_direct_summary_path,
)
from diffusion_toxicity_smc_experiment.smc import (
    SMCComponentConfig,
    _validate_terminal_reward_recheck,
    aggregate_component_estimates,
    ess_from_log_weights,
    logmeanexp_sample_rewards,
    normalize_log_weights,
    resample_indices,
    reverse_snis_component,
    run_smc_component,
    telescoping_log_increments,
    trajectory_mh_log_acceptance,
)
from diffusion_toxicity_smc_experiment.storage import (
    atomic_write_json,
    atomic_write_jsonl,
    initialize_manifest,
    missing_indices,
    scan_direct_generations,
)
from diffusion_toxicity_smc_experiment.submission_probability import (
    component_threshold_estimate,
    direct_threshold_summaries,
    fit_pilot_weights,
    inverse_variance_weights,
    load_study,
    planned_component_seeds,
    study_manifest,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def experiment_config(name: str) -> Path:
    return CONFIG_ROOT / name


def test_prefetch_dtype_fallback_accepts_only_same_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_common.yaml"))
    monkeypatch.setattr(generate_direct_module, "decoder_id", lambda _: "decoder")
    monkeypatch.setattr(generate_direct_module, "evaluator_id", lambda _: "evaluator")
    old_evaluator_dir = (
        tmp_path
        / "mdlm_owt"
        / "prefetch"
        / "decoder"
        / "old-batching-evaluator-id"
        / "verifications"
    )
    old_evaluator_dir.mkdir(parents=True)
    atomic_write_json(
        old_evaluator_dir / "job_123.json",
        {
            "toxicity_evaluator": {
                "classifier": config["evaluation"]["classifier"],
                "dtype": "torch.float16",
            }
        },
    )
    dtype, source = _prefetched_classifier_dtype(config, tmp_path)
    assert dtype == "torch.float16"
    assert source is not None and source.endswith("job_123.json")

    incompatible = copy.deepcopy(config)
    incompatible["evaluation"]["classifier"]["revision"] = "different-revision"
    assert _prefetched_classifier_dtype(incompatible, tmp_path) == (None, None)


def minimal_config() -> dict:
    return {
        "decoder": {
            "checkpoint": {"id": "model", "revision": "abc"},
            "model_implementation": {"id": "implementation", "revision": "def"},
            "tokenizer": {"id": "tokenizer", "revision": "ghi"},
            "generation_mode": "unconditional",
            "fixed_prefix_token_ids": [],
            "prompt": None,
            "predictor": "ddpm",
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "sequence_length": 4,
            "evaluated_completion_tokens": 2,
            "denoising_steps": 2,
            "noise_schedule": {"type": "loglinear", "epsilon": 0.001},
            "sampling_epsilon": 1e-5,
            "final_noise_removal": True,
            "final_noise_removal_method": "stochastic_categorical",
            "mask_token_id": 2,
            "precision": "float32",
        },
        "evaluation": {
            "numerical_batching": {
                "padding": "max_length",
                "pad_final_batch_to_batch_size": True,
            },
            "classifier": {"id": "classifier", "revision": "jkl"},
            "reward": {"definition": "reward"},
            "event": {"definition": "event"},
        },
        "smc": {
            "num_particles": 2,
            "independent_runs": 1,
            "tilted_lambdas": [5.0, 10.0],
            "tilted_component_weights_a": [0.5, 0.5],
            "component_weight_policy": "uniform",
            "optimize_component_weights": False,
            "diagnostic_lambdas": [0.0],
        },
    }


def generation_row(*, sample_id: str, sample_index: int, identifier: str) -> dict:
    return {
        "sample_id": sample_id,
        "sample_index": sample_index,
        "conditioning_mode": "unconditional",
        "shard_seed": 7,
        "position_in_shard": sample_index,
        "token_ids": [0, 1],
        "completion_token_ids": [0, 1],
        "decoded_text": "xy",
        "checkpoint_revision": "abc",
        "tokenizer_revision": "ghi",
        "decoder_id": identifier,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
    }


class NoOpModel:
    def __call__(self, **kwargs):  # pragma: no cover - initial state test never calls it
        raise AssertionError(kwargs)


class UniformTwoTokenModel:
    """Assign equal mass to tokens 0 and 1; token 2 is the absorbing mask."""

    def __call__(self, *, input_ids, timesteps, return_dict=False):
        del timesteps, return_dict
        return (torch.zeros((*input_ids.shape, 3), device=input_ids.device),)


def test_native_trajectory_retains_real_checkpoints_and_terminal_state() -> None:
    sampler = NativeMDLMSampler(
        UniformTwoTokenModel(),
        mask_token_id=2,
        predictor="ddpm",
        final_noise_removal=True,
        final_noise_removal_method="stochastic_categorical",
    )
    generator = torch.Generator().manual_seed(91)
    trajectory, prefix = sampler.sample_trajectory(
        batch_size=3,
        sequence_length=5,
        num_steps=2,
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        device="cpu",
        generator=generator,
        microbatch_size=2,
    )
    assert trajectory.shape == (3, 4, 5)
    assert prefix.numel() == 0
    assert bool((trajectory[:, 0] == 2).all())
    assert not bool((trajectory[:, -1] == 2).any())

    checkpoint = trajectory[:, 1]
    suffix = sampler.rollout_trajectory_suffix(
        checkpoint,
        start_step=1,
        num_steps=2,
        generator=generator,
        microbatch_size=2,
    )
    assert suffix.shape == (3, 3, 5)
    torch.testing.assert_close(suffix[:, 0], checkpoint)
    assert not bool((suffix[:, -1] == 2).any())


@pytest.mark.parametrize("batch_size", [1, 3])
def test_subs_mixed_masked_and_revealed_positions(batch_size: int) -> None:
    mask_token_id = 4
    state_rows = [
        [mask_token_id, 2, mask_token_id, 1],
        [0, mask_token_id, 3, mask_token_id],
        [mask_token_id, mask_token_id, 1, 2],
    ]
    states = torch.tensor(state_rows[:batch_size], dtype=torch.long)
    raw_logits = torch.arange(
        batch_size * states.shape[1] * 5, dtype=torch.float32
    ).reshape(batch_size, states.shape[1], 5)

    actual = subs_log_probabilities(
        raw_logits, states, mask_token_id=mask_token_id
    )

    assert actual.shape == (batch_size, states.shape[1], 5)
    assert bool(torch.isneginf(actual[..., mask_token_id]).all())

    masked = states == mask_token_id
    expected_masked_logits = raw_logits.float().clone()
    expected_masked_logits[..., mask_token_id] = -torch.inf
    expected_masked = torch.log_softmax(expected_masked_logits, dim=-1)
    torch.testing.assert_close(actual[masked], expected_masked[masked])

    revealed = ~masked
    revealed_rows = actual[revealed]
    revealed_token_ids = states[revealed]
    assert bool((revealed_rows.gather(1, revealed_token_ids.unsqueeze(1)) == 0.0).all())
    off_token = torch.ones_like(revealed_rows, dtype=torch.bool)
    off_token.scatter_(1, revealed_token_ids.unsqueeze(1), False)
    assert bool(torch.isneginf(revealed_rows[off_token]).all())


def test_subs_rejects_malformed_four_dimensional_logits() -> None:
    states = torch.tensor([[4, 1, 4, 2]], dtype=torch.long)
    malformed = torch.zeros((1, 1, 4, 5), dtype=torch.float32)
    with pytest.raises(
        ValueError,
        match=r"logits shape \(1, 1, 4, 5\).*states shape \(1, 4\)",
    ):
        subs_log_probabilities(malformed, states, mask_token_id=4)


def test_unconditional_initial_state_is_exactly_l_masks_without_prefix() -> None:
    sampler = NativeMDLMSampler(NoOpModel(), mask_token_id=7)
    states, prefix = sampler.initial_states(
        batch_size=3,
        sequence_length=5,
        device="cpu",
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
    )
    assert states.shape == (3, 5)
    assert bool((states == 7).all())
    assert prefix.numel() == 0


def test_terminal_mask_removal_is_stochastic_not_argmax() -> None:
    sampler = NativeMDLMSampler(
        UniformTwoTokenModel(),
        mask_token_id=2,
        final_noise_removal_method="stochastic_categorical",
    )
    states = torch.full((64, 4), 2, dtype=torch.long)
    completed = sampler.terminal_completion(
        states,
        t_scalar=1e-5,
        generator=torch.Generator().manual_seed(1234),
    )
    assert not bool((completed == 2).any())
    assert set(completed.unique().tolist()) == {0, 1}


def test_terminal_sampling_microbatch_preserves_draw_and_bounds_forward_batch() -> None:
    states = torch.full((5, 4), 2, dtype=torch.long)
    full_sampler = NativeMDLMSampler(
        UniformTwoTokenModel(),
        mask_token_id=2,
        final_noise_removal_method="stochastic_categorical",
    )
    chunked_sampler = NativeMDLMSampler(
        UniformTwoTokenModel(),
        mask_token_id=2,
        final_noise_removal_method="stochastic_categorical",
    )
    full = full_sampler.terminal_completion(
        states,
        t_scalar=1e-5,
        generator=torch.Generator().manual_seed(22),
    )
    chunked = chunked_sampler.terminal_completion(
        states,
        t_scalar=1e-5,
        generator=torch.Generator().manual_seed(22),
        microbatch_size=2,
    )
    torch.testing.assert_close(chunked, full)
    assert full_sampler.nfe == 1
    assert chunked_sampler.nfe == 3


def test_ddpm_microbatch_preserves_transition_draw() -> None:
    states = torch.full((5, 4), 2, dtype=torch.long)
    full_sampler = NativeMDLMSampler(UniformTwoTokenModel(), mask_token_id=2)
    chunked_sampler = NativeMDLMSampler(UniformTwoTokenModel(), mask_token_id=2)
    full, _ = full_sampler.ddpm_step(
        states,
        t_scalar=0.5,
        dt=0.1,
        generator=torch.Generator().manual_seed(91),
    )
    chunked, _ = chunked_sampler.ddpm_step(
        states,
        t_scalar=0.5,
        dt=0.1,
        generator=torch.Generator().manual_seed(91),
        microbatch_size=2,
    )
    torch.testing.assert_close(chunked, full)
    assert full_sampler.nfe == 1
    assert chunked_sampler.nfe == 3


def test_fk_mdlm_multinomial_x0_samples_are_stochastic_and_preserve_revealed() -> None:
    sampler = NativeMDLMSampler(UniformTwoTokenModel(), mask_token_id=2)
    states = torch.tensor(
        [[2, 0, 2, 1], [1, 2, 0, 2]],
        dtype=torch.long,
    )
    next_states, cache, samples = sampler.ddpm_step_with_x0_samples(
        states,
        t_scalar=0.5,
        dt=0.1,
        num_x0_samples=4,
        generator=torch.Generator().manual_seed(117),
    )

    assert next_states.shape == states.shape
    assert cache is None
    assert samples.shape == (2, 4, 4)
    revealed = states != sampler.mask_token_id
    expected_revealed = states[:, None, :].expand_as(samples)
    assert torch.equal(samples[revealed[:, None, :].expand_as(samples)], expected_revealed[revealed[:, None, :].expand_as(samples)])
    masked_samples = samples[(~revealed)[:, None, :].expand_as(samples)]
    assert set(masked_samples.tolist()) == {0, 1}
    assert sampler.nfe == 1


def test_fk_intermediate_log_rewards_use_logmeanexp() -> None:
    sample_rewards = torch.log(
        torch.tensor([[0.1, 0.3, 0.5, 0.7], [0.2, 0.2, 0.2, 0.2]])
    )
    aggregated = logmeanexp_sample_rewards(sample_rewards)
    torch.testing.assert_close(
        aggregated,
        torch.log(torch.tensor([0.4, 0.2], dtype=torch.float64)),
    )


def test_unconditional_config_rejects_prompt_or_fixed_prefix() -> None:
    config = minimal_config()
    with_prompt = copy.deepcopy(config)
    with_prompt["decoder"]["prompt"] = "fake"
    with pytest.raises(ValueError, match="prompt: null"):
        decoder_id(with_prompt)
    with_prefix = copy.deepcopy(config)
    with_prefix["decoder"]["fixed_prefix_token_ids"] = [0]
    with pytest.raises(ValueError, match="empty fixed prefix"):
        decoder_id(with_prefix)


def test_unconditional_and_prompt_conditioned_decoders_have_different_ids() -> None:
    unconditional = minimal_config()
    conditioned = copy.deepcopy(unconditional)
    conditioned["decoder"]["generation_mode"] = "fixed_prefix"
    conditioned["decoder"]["fixed_prefix_token_ids"] = [0, 42]
    conditioned["decoder"]["prompt"] = "legacy prompt"
    assert decoder_id(unconditional) != decoder_id(conditioned)


def test_log_potentials_telescope() -> None:
    beta = [0.0, 0.3, 0.7, 2.0]
    rewards = [0.0, -1.2, -0.4, -0.1]
    assert sum(telescoping_log_increments(beta, rewards)) == pytest.approx(
        beta[-1] * rewards[-1]
    )


def test_checkpoint_schedule_telescopes_to_terminal_lambda_reward() -> None:
    terminal_lambda = 4.0
    checkpoint_levels = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    beta = [terminal_lambda * level / 32 for level in checkpoint_levels]
    rewards = [0.0, -6.0, -4.5, -3.0, -2.0, -1.0, -0.5, -0.2, -0.1]
    increments = telescoping_log_increments(beta, rewards)
    assert sum(increments) == pytest.approx(terminal_lambda * rewards[-1])
    assert beta[-1] == terminal_lambda


def test_unconditional_event_text_is_retokenized_and_trimmed() -> None:
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(character) for character in text]

        def decode(self, token_ids):
            return "".join(chr(value) for value in token_ids)

    text, token_ids = unconditional_trimmed_classifier_text(
        decoded_text="abcdef",
        trimming_tokenizer=CharacterTokenizer(),
        completion_tokens=3,
    )
    assert token_ids == [ord("a"), ord("b"), ord("c")]
    assert text == "abc"


def test_log_weight_normalization() -> None:
    normalized = normalize_log_weights(torch.tensor([1000.0, 999.0, -torch.inf]))
    assert float(torch.logsumexp(normalized, dim=0)) == pytest.approx(0.0)
    assert normalized[-1].item() == -torch.inf


def test_toxicity_reward_is_invariant_to_batch_composition_with_fixed_shapes() -> None:
    class Tokenizer:
        def __call__(
            self,
            texts,
            *,
            return_tensors,
            padding,
            truncation,
            max_length,
            return_token_type_ids,
        ):
            del return_tensors, truncation, return_token_type_ids
            assert padding == "max_length"
            rows = []
            masks = []
            for text in texts:
                values = [ord(character) % 17 + 1 for character in text][:max_length]
                rows.append(values + [0] * (max_length - len(values)))
                masks.append([1] * len(values) + [0] * (max_length - len(values)))
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    class ShapeSensitiveModel:
        def __call__(self, *, input_ids, attention_mask):
            values = (input_ids * attention_mask).sum(dim=1).float()
            shape_term = 1e-3 * (input_ids.shape[0] + input_ids.shape[1])
            logits = torch.stack((torch.zeros_like(values), values + shape_term), dim=1)
            return SimpleNamespace(logits=logits)

    evaluator = ToxicityEvaluator(
        ShapeSensitiveModel(),
        Tokenizer(),
        torch.device("cpu"),
        batch_size=4,
        padding="max_length",
        pad_final_batch_to_batch_size=True,
    )
    alone = evaluator.score_texts(["same"], max_length=12)[0]["reward"]
    together = evaluator.score_texts(
        ["same", "short", "a much longer peer", "x"], max_length=12
    )[0]["reward"]
    assert alone == together


def test_all_negative_infinite_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_log_weights(torch.tensor([-torch.inf, -torch.inf]))


@pytest.mark.parametrize("method", ["multinomial", "systematic", "stratified"])
def test_all_resampling_methods(method: str) -> None:
    indices = resample_indices(
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        method=method,
        generator=torch.Generator().manual_seed(4),
    )
    assert indices.shape == (3,)
    assert bool(((indices >= 0) & (indices < 3)).all())
    reduced = resample_indices(
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        method=method,
        generator=torch.Generator().manual_seed(4),
        num_samples=2,
    )
    assert reduced.shape == (2,)
    assert bool(((reduced >= 0) & (reduced < 3)).all())


def test_target_samples_and_resume_without_duplicates(tmp_path: Path) -> None:
    config = minimal_config()
    identifier = decoder_id(config)
    run_dir = tmp_path / identifier
    initialize_manifest(
        run_dir,
        {"decoder_id": identifier, "decoder_manifest": decoder_manifest(config)},
    )
    atomic_write_jsonl(
        run_dir / "generations" / "shard_00000000_00000001.jsonl",
        [
            generation_row(sample_id="a", sample_index=0, identifier=identifier),
            generation_row(sample_id="b", sample_index=1, identifier=identifier),
        ],
    )
    inventory = scan_direct_generations(run_dir, identifier)
    assert inventory.count == 2
    assert missing_indices(inventory, target=4) == [2, 3]
    assert missing_indices(inventory, target=2) == []


def test_resolved_json_preserves_scientific_notation_and_decoder_id(
    tmp_path: Path,
) -> None:
    config = minimal_config()
    path = tmp_path / "resolved.json"
    atomic_write_json(path, config)

    reloaded = load_yaml(path)

    assert reloaded == config
    assert isinstance(reloaded["decoder"]["sampling_epsilon"], float)
    assert reloaded["decoder"]["sampling_epsilon"] == 1e-5
    assert decoder_id(reloaded) == decoder_id(config)


def test_sampling_epsilon_string_is_rejected_before_fingerprinting() -> None:
    config = minimal_config()
    config["decoder"]["sampling_epsilon"] = "1e-05"
    with pytest.raises(ValueError, match="sampling_epsilon"):
        decoder_id(config)


def test_direct_generation_scans_inventory_only_before_and_after_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import diffusion_toxicity_smc_experiment.generate_direct as direct_module

    config = minimal_config()
    config["direct"] = {
        "base_seed": 101,
        "shard_size": 2,
        "generation_microbatch_size": 2,
    }

    class FakeSampler:
        def __init__(self, model, **kwargs):
            del model, kwargs
            self.nfe = 0

        def reset_nfe(self) -> None:
            self.nfe = 0

        def sample(
            self,
            *,
            batch_size,
            sequence_length,
            num_steps,
            fixed_prefix_token_ids,
            device,
            generator,
        ):
            del fixed_prefix_token_ids, generator
            self.nfe += num_steps
            return torch.zeros(
                (batch_size, sequence_length), dtype=torch.long, device=device
            )

    class FakeTokenizer:
        def batch_decode(self, rows, skip_special_tokens=True):
            del skip_special_tokens
            return [" ".join(map(str, row)) for row in rows]

    runtime = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        model_load_seconds=0.0,
        tokenizer_load_seconds=0.0,
    )
    scan_calls = 0
    real_scan = direct_module.scan_direct_generations

    def counting_scan(run_dir, identifier):
        nonlocal scan_calls
        scan_calls += 1
        return real_scan(run_dir, identifier)

    monkeypatch.setattr(direct_module, "NativeMDLMSampler", FakeSampler)
    monkeypatch.setattr(
        direct_module,
        "load_generator_runtime",
        lambda config, local_files_only: runtime,
    )
    monkeypatch.setattr(direct_module, "scan_direct_generations", counting_scan)

    output_root = tmp_path
    if os.name == "nt":
        output_root = Path("\\\\?\\" + str(output_root.resolve()))
    run_dir = generate_missing(
        config,
        target_samples=5,
        output_root=output_root,
        task_index=0,
        num_tasks=1,
        allow_network=False,
    )

    assert scan_calls == 2
    inventory = real_scan(run_dir, decoder_id(config))
    assert inventory.sample_indices == set(range(5))
    assert len(list((run_dir / "generations").glob("shard_*.jsonl"))) == 3


def test_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_manifest(run_dir, {"decoder_id": "a"})
    with pytest.raises(ValueError, match="manifest mismatch"):
        initialize_manifest(run_dir, {"decoder_id": "b"})


def test_direct_summary_growth_is_append_only_and_latest_is_selected(tmp_path: Path) -> None:
    first = {"evaluated_samples": 2, "empirical_probability": 0.0}
    extended = {"evaluated_samples": 256, "empirical_probability": 0.01}
    _write_summary_snapshot(tmp_path, first)
    legacy_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    _write_summary_snapshot(tmp_path, extended)
    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == legacy_text
    latest = latest_direct_summary_path(tmp_path)
    assert latest == tmp_path / "summaries" / "summary_00000256.json"
    assert json.loads(latest.read_text(encoding="utf-8")) == extended


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    identifier = "decoder"
    atomic_write_jsonl(
        tmp_path / "generations" / "shard_00000.jsonl",
        [generation_row(sample_id="same", sample_index=0, identifier=identifier)],
    )
    atomic_write_jsonl(
        tmp_path / "generations" / "shard_00001.jsonl",
        [generation_row(sample_id="same", sample_index=1, identifier=identifier)],
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        scan_direct_generations(tmp_path, identifier)


def test_reverse_snis_converges_on_finite_state_model() -> None:
    base = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    rewards = torch.tensor([-2.0, -0.5, -0.1], dtype=torch.float64)
    events = torch.tensor([False, False, True])
    lam = 3.0
    tilted = base * torch.exp(lam * rewards)
    tilted /= tilted.sum()
    sampled = torch.multinomial(
        tilted, 100_000, replacement=True, generator=torch.Generator().manual_seed(99)
    )
    result = reverse_snis_component(
        final_log_weights=torch.full((sampled.numel(),), -math.log(sampled.numel())),
        rewards=rewards[sampled],
        events=events[sampled],
        lambda_final=lam,
    )
    assert result.estimate == pytest.approx(float(base[events].sum()), abs=0.01)


class ToySampler:
    mask_token_id = 2
    sampling_epsilon = 1e-5
    final_noise_removal = False

    def __init__(self) -> None:
        self.nfe = 0

    def initial_states(self, *, batch_size, sequence_length, device, fixed_prefix_token_ids):
        assert fixed_prefix_token_ids is None or fixed_prefix_token_ids.numel() == 0
        return (
            torch.full((batch_size, sequence_length), self.mask_token_id, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    def reset_nfe(self) -> None:
        self.nfe = 0

    def ddpm_step(
        self,
        states,
        *,
        t_scalar,
        dt,
        generator,
        cached_p_x0=None,
        microbatch_size=None,
    ):
        del t_scalar, dt, cached_p_x0, microbatch_size
        self.nfe += 1
        draw = torch.randint(0, 2, states.shape, device=states.device, generator=generator)
        return torch.where(states == self.mask_token_id, draw, states), None


class TrajectoryToySampler:
    mask_token_id = 2
    sampling_epsilon = 1e-5
    final_noise_removal = True
    predictor = "ddpm"

    def __init__(self) -> None:
        self.nfe = 0
        self.rollout_batches: list[tuple[int, int]] = []

    def reset_nfe(self) -> None:
        self.nfe = 0

    def sample_trajectory(
        self,
        *,
        batch_size,
        sequence_length,
        num_steps,
        fixed_prefix_token_ids,
        device,
        generator,
        microbatch_size=None,
    ):
        del microbatch_size
        assert fixed_prefix_token_ids is None or fixed_prefix_token_ids.numel() == 0
        terminal = torch.randint(
            0,
            2,
            (batch_size, sequence_length),
            device=device,
            generator=generator,
        )
        path = torch.full(
            (batch_size, num_steps + 2, sequence_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        path[:, -1] = terminal
        self.nfe += num_steps
        return path, torch.empty(0, dtype=torch.long, device=device)

    def rollout_trajectory_suffix(
        self,
        checkpoint_states,
        *,
        start_step,
        num_steps,
        generator,
        microbatch_size=None,
    ):
        del microbatch_size
        batch_size, sequence_length = checkpoint_states.shape
        depth = num_steps - start_step
        self.rollout_batches.append((depth, batch_size))
        suffix = checkpoint_states[:, None].expand(
            batch_size, depth + 2, sequence_length
        ).clone()
        suffix[:, -1] = torch.randint(
            0,
            2,
            (batch_size, sequence_length),
            device=checkpoint_states.device,
            generator=generator,
        )
        self.nfe += depth
        return suffix


def test_trajectory_mh_acceptance_satisfies_two_state_detailed_balance() -> None:
    base = torch.tensor([0.75, 0.25], dtype=torch.float64)
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float64)
    eta = 1.3
    target = base * torch.exp(eta * rewards)
    target /= target.sum()
    log_01 = trajectory_mh_log_acceptance(
        rewards[:1], rewards[1:], target_lambda=eta
    )[0]
    log_10 = trajectory_mh_log_acceptance(
        rewards[1:], rewards[:1], target_lambda=eta
    )[0]
    transition_01 = base[1] * log_01.exp()
    transition_10 = base[0] * log_10.exp()
    assert target[0] * transition_01 == pytest.approx(
        target[1] * transition_10, abs=1e-12
    )


def test_terminal_reward_recheck_reports_numerical_drift() -> None:
    cached = torch.tensor([-0.1, -2.0], dtype=torch.float64)
    _validate_terminal_reward_recheck(
        cached,
        cached + torch.tensor([1e-10, 0.0], dtype=torch.float64),
        backend="test",
    )
    with pytest.raises(
        RuntimeError,
        match=r"mismatch_count=1/2.*max_abs=.*max_abs_index=1.*cached=-2",
    ):
        _validate_terminal_reward_recheck(
            cached,
            torch.tensor([-0.1, -2.01], dtype=torch.float64),
            backend="test",
        )


def test_trajectory_resample_move_is_dispatched_and_gpu_groupable() -> None:
    sampler = TrajectoryToySampler()
    result = run_smc_component(
        sampler=sampler,
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] == 1,
        config=SMCComponentConfig(
            num_particles=64,
            num_steps=2,
            sequence_length=3,
            lambda_final=1.0,
            ess_threshold=0.5,
            resampling_method="systematic",
            final_resampling=False,
            seed=321,
            tilted_sampler_type="trajectory_resample_move",
            num_tempering_levels=3,
            mcmc_steps_per_level=1,
            mutation_policy="every_level",
            rollback_steps=(1, 2),
            rollback_probabilities=(0.5, 0.5),
        ),
        device="cpu",
    )
    assert result.tilted_sampler_type == "trajectory_resample_move"
    assert len(result.mutation_diagnostics or []) == 3
    assert sum(row["proposals"] for row in result.mutation_diagnostics or []) == 192
    assert all(row["rollback_counts"].keys() == {"1", "2"} for row in result.mutation_diagnostics or [])
    # Every sweep proposes all particles, grouped into at most one call per depth.
    assert sum(batch for _, batch in sampler.rollout_batches) == 192
    assert {depth for depth, _ in sampler.rollout_batches} == {1, 2}
    assert sum(
        row["rollback_particle_steps"] for row in result.mutation_diagnostics or []
    ) == sum(depth * batch for depth, batch in sampler.rollout_batches)
    assert result.trajectory_storage_bytes == 64 * 4 * 3 * 8
    assert not bool((result.terminal_tokens == sampler.mask_token_id).any())
    assert bool(torch.isfinite(result.final_log_weights).all())
    assert result.reverse_snis.corrected_ess > 0.0


def test_trajectory_resample_move_lambda_zero_remains_the_base_path_measure() -> None:
    result = run_smc_component(
        sampler=TrajectoryToySampler(),
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] == 1,
        config=SMCComponentConfig(
            num_particles=32,
            num_steps=2,
            sequence_length=3,
            lambda_final=0.0,
            seed=11,
            tilted_sampler_type="trajectory_resample_move",
            num_tempering_levels=2,
            mcmc_steps_per_level=1,
            mutation_policy="every_level",
            rollback_steps=(1, 2),
        ),
        device="cpu",
    )
    torch.testing.assert_close(
        result.final_log_weights,
        torch.full((32,), -math.log(32), dtype=torch.float64),
    )
    assert result.log_normalizer_estimate == pytest.approx(0.0, abs=1e-12)
    assert result.genealogy == []
    assert sum(row["accepted"] for row in result.mutation_diagnostics or []) == 64


class DeterministicWasteFreeToySampler(TrajectoryToySampler):
    def __init__(self) -> None:
        super().__init__()
        self.next_terminal_value = 3

    def sample_trajectory(
        self,
        *,
        batch_size,
        sequence_length,
        num_steps,
        fixed_prefix_token_ids,
        device,
        generator,
        microbatch_size=None,
    ):
        del generator, microbatch_size
        assert fixed_prefix_token_ids is None or fixed_prefix_token_ids.numel() == 0
        path = torch.zeros(
            (batch_size, num_steps + 2, sequence_length),
            dtype=torch.long,
            device=device,
        )
        self.nfe += num_steps
        return path, torch.empty(0, dtype=torch.long, device=device)

    def rollout_trajectory_suffix(
        self,
        checkpoint_states,
        *,
        start_step,
        num_steps,
        generator,
        microbatch_size=None,
    ):
        del generator, microbatch_size
        batch_size, sequence_length = checkpoint_states.shape
        depth = num_steps - start_step
        self.rollout_batches.append((depth, batch_size))
        suffix = checkpoint_states[:, None].expand(
            batch_size, depth + 2, sequence_length
        ).clone()
        suffix[:, -1] = self.next_terminal_value
        self.next_terminal_value += 1
        self.nfe += depth
        return suffix


def test_trajectory_waste_free_retains_every_mh_chain_state() -> None:
    sampler = DeterministicWasteFreeToySampler()
    result = run_smc_component(
        sampler=sampler,
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] >= 3,
        config=SMCComponentConfig(
            num_particles=12,
            num_steps=2,
            sequence_length=3,
            lambda_final=0.0,
            final_resampling=False,
            seed=19,
            tilted_sampler_type="trajectory_waste_free",
            num_tempering_levels=1,
            mutation_policy="every_level",
            rollback_steps=(1,),
            rollback_probabilities=(1.0,),
            waste_free_chain_length=3,
        ),
        device="cpu",
    )

    assert result.tilted_sampler_type == "trajectory_waste_free"
    assert result.terminal_tokens.shape == (12, 3)
    assert result.waste_free_chain_ids is not None
    assert result.waste_free_chain_positions is not None
    assert result.waste_free_chain_ids.tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]
    assert result.waste_free_chain_positions.tolist() == [0, 1, 2] * 4
    terminals = result.terminal_tokens[:, -1].reshape(4, 3)
    torch.testing.assert_close(
        terminals,
        torch.tensor([[0, 3, 4]] * 4, dtype=torch.long),
    )
    assert len(result.genealogy) == 1
    assert len(result.genealogy[0]["parents"]) == 4
    assert len(result.genealogy[0]["expanded_seed_parents"]) == 12
    assert len(result.mutation_diagnostics or []) == 2
    assert sum(row["proposals"] for row in result.mutation_diagnostics or []) == 8
    assert sum(row["accepted"] for row in result.mutation_diagnostics or []) == 8
    assert result.waste_free_diagnostics is not None
    waste_free_row = result.waste_free_diagnostics[0]
    assert waste_free_row["population_size"] == 12
    assert waste_free_row["seed_count"] == 4
    assert waste_free_row["chain_length"] == 3
    assert waste_free_row["mh_transitions_per_chain"] == 2
    assert waste_free_row["proposals"] == 8
    assert waste_free_row["accepted"] == 8
    assert waste_free_row["rejected_self_transitions"] == 0
    torch.testing.assert_close(
        result.final_log_weights,
        torch.full((12,), -math.log(12), dtype=torch.float64),
    )
    assert result.log_normalizer_estimate == pytest.approx(0.0, abs=1e-12)
    assert result.trajectory_storage_bytes == 12 * 4 * 3 * 8


def test_trajectory_waste_free_requires_factorable_population() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SMCComponentConfig(
            num_particles=10,
            num_steps=2,
            sequence_length=3,
            lambda_final=1.0,
            final_resampling=False,
            tilted_sampler_type="trajectory_waste_free",
            num_tempering_levels=2,
            mutation_policy="every_level",
            rollback_steps=(1,),
            waste_free_chain_length=3,
        )


def test_trajectory_waste_free_matches_algorithm2_move_then_weight_order() -> None:
    result = run_smc_component(
        sampler=DeterministicWasteFreeToySampler(),
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] >= 3,
        config=SMCComponentConfig(
            num_particles=12,
            num_steps=2,
            sequence_length=3,
            lambda_final=1.0,
            final_resampling=False,
            seed=23,
            tilted_sampler_type="trajectory_waste_free",
            num_tempering_levels=2,
            mutation_policy="every_level",
            rollback_steps=(1,),
            rollback_probabilities=(1.0,),
            waste_free_chain_length=3,
        ),
        device="cpu",
    )

    mutation_targets = [
        row["target_lambda"] for row in result.mutation_diagnostics or []
    ]
    assert mutation_targets == pytest.approx([0.0, 0.0, 0.5, 0.5])
    expected_first_increment = math.log(
        (1.0 + math.exp(0.5 * 3.0) + math.exp(0.5 * 4.0)) / 3.0
    )
    assert result.level_diagnostics[0]["log_normalizer_increment"] == pytest.approx(
        expected_first_increment, abs=1e-12
    )
    assert result.level_diagnostics[0]["mh_target_lambda"] == 0.0
    assert result.level_diagnostics[1]["mh_target_lambda"] == 0.5
    assert result.waste_free_diagnostics is not None
    assert result.waste_free_diagnostics[0]["output_target_lambda"] == 0.5
    assert result.waste_free_diagnostics[1]["output_target_lambda"] == 1.0
    assert ess_from_log_weights(result.final_log_weights) < 12.0
    torch.testing.assert_close(
        result.pre_final_resampling_log_weights,
        result.post_final_resampling_log_weights,
    )


def test_one_level_waste_free_reverse_correction_cancels_the_tilt() -> None:
    result = run_smc_component(
        sampler=TrajectoryToySampler(),
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] == 1,
        config=SMCComponentConfig(
            num_particles=24,
            num_steps=2,
            sequence_length=3,
            lambda_final=1.25,
            final_resampling=False,
            seed=29,
            tilted_sampler_type="trajectory_waste_free",
            num_tempering_levels=1,
            mutation_policy="every_level",
            rollback_steps=(1, 2),
            rollback_probabilities=(0.5, 0.5),
            waste_free_chain_length=3,
        ),
        device="cpu",
    )

    corrected = torch.tensor(
        result.reverse_snis.corrected_log_weights, dtype=torch.float64
    )
    torch.testing.assert_close(
        corrected,
        torch.full((24,), -math.log(24), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_lambda_zero_reduces_to_unconditional_base_sampler() -> None:
    result = run_smc_component(
        sampler=ToySampler(),
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=lambda tokens: tokens[:, -1].double(),
        event_fn=lambda tokens: tokens[:, -1] == 1,
        config=SMCComponentConfig(
            num_particles=32,
            num_steps=1,
            sequence_length=2,
            lambda_final=0.0,
            seed=123,
        ),
        device="cpu",
    )
    assert result.final_log_weights == pytest.approx(
        torch.full((32,), -math.log(32)), abs=1e-7
    )
    assert result.genealogy == []
    assert result.log_normalizer_estimate == pytest.approx(0.0)
    assert not bool((result.terminal_tokens == 2).any())


class TerminalCheckpointToySampler:
    mask_token_id = 2
    sampling_epsilon = 1e-5
    final_noise_removal = True

    def __init__(self) -> None:
        self.nfe = 0
        self.deterministic_completion_calls = 0
        self.terminal_completion_calls = 0
        self.intermediate_x0_sample_calls = 0

    def initial_states(self, *, batch_size, sequence_length, device, fixed_prefix_token_ids):
        assert fixed_prefix_token_ids is None or fixed_prefix_token_ids.numel() == 0
        return (
            torch.full((batch_size, sequence_length), self.mask_token_id, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    def reset_nfe(self) -> None:
        self.nfe = 0

    def ddpm_step(
        self,
        states,
        *,
        t_scalar,
        dt,
        generator,
        cached_p_x0=None,
        microbatch_size=None,
    ):
        del t_scalar, dt, generator, cached_p_x0, microbatch_size
        self.nfe += 1
        return states, None

    def deterministic_completion(self, states, *, t_scalar, microbatch_size=None):
        del t_scalar, microbatch_size
        self.deterministic_completion_calls += 1
        return torch.zeros_like(states)

    def ddpm_step_with_x0_samples(
        self,
        states,
        *,
        t_scalar,
        dt,
        num_x0_samples,
        generator,
        cached_p_x0=None,
        microbatch_size=None,
    ):
        del t_scalar, dt, generator, cached_p_x0, microbatch_size
        self.nfe += 1
        self.intermediate_x0_sample_calls += 1
        samples = torch.zeros(
            (states.shape[0], num_x0_samples, states.shape[1]),
            dtype=states.dtype,
            device=states.device,
        )
        return states, None, samples

    def terminal_completion(
        self, states, *, t_scalar, generator, microbatch_size=None
    ):
        del t_scalar, generator, microbatch_size
        self.terminal_completion_calls += 1
        return torch.ones_like(states)


def test_fk_uses_multinomial_x0_samples_and_true_stochastic_terminal() -> None:
    sampler = TerminalCheckpointToySampler()
    reward_inputs: list[torch.Tensor] = []

    def reward(tokens: torch.Tensor) -> torch.Tensor:
        reward_inputs.append(tokens.clone())
        return tokens[:, 0].double()

    result = run_smc_component(
        sampler=sampler,
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long),
        reward_fn=reward,
        event_fn=lambda tokens: tokens[:, 0] == 1,
        config=SMCComponentConfig(
            num_particles=4,
            num_steps=4,
            sequence_length=2,
            lambda_final=2.0,
            reward_frequency=2,
            final_resampling=False,
            seed=7,
        ),
        device="cpu",
    )

    assert sampler.deterministic_completion_calls == 0
    assert sampler.intermediate_x0_sample_calls == 4
    assert sampler.terminal_completion_calls == 1
    assert len(reward_inputs) == 2
    assert reward_inputs[0].shape == (16, 2)
    assert bool((reward_inputs[0] == 0).all())
    assert bool((reward_inputs[1] == 1).all())
    checkpoints = [row for row in result.level_diagnostics if row["checkpoint"]]
    assert [row["level"] for row in checkpoints] == [2, 4]
    assert [row["beta"] for row in checkpoints] == pytest.approx([1.0, 2.0])
    assert result.log_normalizer_estimate == pytest.approx(2.0)
    torch.testing.assert_close(
        result.pre_final_resampling_log_weights,
        result.post_final_resampling_log_weights,
    )


def test_two_reverse_snis_ratios_are_separate_then_averaged() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_smoke_technical.yaml"))
    summaries = []
    expected_runs = []
    for run_index, estimates in enumerate(((0.2, 0.6), (0.4, 0.8))):
        expected_runs.append(sum(estimates) / 2)
        for lam, estimate in zip((5.0, 10.0), estimates):
            summaries.append(
                {
                    "run_index": run_index,
                    "lambda": lam,
                    "reverse_snis_estimate": estimate,
                }
            )
    aggregate = _aggregate(summaries, config)
    assert aggregate["component_estimates_by_run"] == [[0.2, 0.6], [0.4, 0.8]]
    assert aggregate["run_estimates"] == pytest.approx(expected_runs)
    assert aggregate["mean_reverse_snis_estimate"] == pytest.approx(
        sum(expected_runs) / 2
    )
    assert "prompt" not in json.dumps(aggregate).lower()


def test_reverse_snis_components_are_normalized_separately_before_aggregation() -> None:
    first = reverse_snis_component(
        final_log_weights=torch.log(torch.tensor([0.9, 0.1], dtype=torch.float64)),
        rewards=torch.tensor([-2.0, -0.1], dtype=torch.float64),
        events=torch.tensor([False, True]),
        lambda_final=1.0,
    )
    second = reverse_snis_component(
        final_log_weights=torch.log(torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)),
        rewards=torch.tensor([-3.0, -1.0, -0.2], dtype=torch.float64),
        events=torch.tensor([False, True, True]),
        lambda_final=2.0,
    )
    assert torch.tensor(
        first.corrected_log_weights, dtype=torch.float64
    ).logsumexp(0).item() == pytest.approx(0.0)
    assert torch.tensor(
        second.corrected_log_weights, dtype=torch.float64
    ).logsumexp(0).item() == pytest.approx(0.0)
    combined = aggregate_component_estimates(
        [first.estimate, second.estimate], [0.5, 0.5]
    )
    assert combined == pytest.approx((first.estimate + second.estimate) / 2)


def test_generic_aggregator_accepts_nonuniform_simplex_weights() -> None:
    assert aggregate_component_estimates([0.1, 0.9], [0.25, 0.75]) == pytest.approx(
        0.25 * 0.1 + 0.75 * 0.9
    )


def test_direct_unconditional_baseline_has_binomial_se_and_wilson() -> None:
    rows = [
        {"conditioning_mode": "unconditional", "is_event": value}
        for value in (True, False, False, False)
    ]
    summary = direct_unconditional_summary(rows)
    assert summary["empirical_probability"] == pytest.approx(0.25)
    assert summary["binomial_standard_error"] == pytest.approx(
        math.sqrt(0.25 * 0.75 / 4)
    )
    assert 0 <= summary["wilson_95_low"] <= summary["wilson_95_high"] <= 1
    assert "prompt" not in json.dumps(summary).lower()


def test_smoke_profiles_are_unconditional_distinct_and_uniform() -> None:
    technical = load_yaml(experiment_config("mdlm_owt_unconditional_smoke_technical.yaml"))
    scientific = load_yaml(experiment_config("mdlm_owt_unconditional_smoke_scientific.yaml"))
    assert decoder_id(technical) != decoder_id(scientific)
    for config in (technical, scientific):
        decoder = config["decoder"]
        assert decoder["generation_mode"] == "unconditional"
        assert decoder["fixed_prefix_token_ids"] == []
        assert decoder["prompt"] is None
        assert config["direct"]["target_samples"] == 2
        assert config["smc"]["tilted_lambdas"] == [5.0, 10.0]
        assert config["smc"]["tilted_component_weights_a"] == [0.5, 0.5]
        manifest = smc_manifest(config)
        assert manifest["decoder_id"] == decoder_id(config)
        assert manifest["diagnostic_components"] == {
            "lambdas": [0.0],
            "included_in_tilted_component_aggregation": False,
        }
        assert "prompt" not in json.dumps(manifest).lower()


def test_direct_and_smc_unconditional_use_the_same_decoder_id() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_smoke_technical.yaml"))
    identifier = decoder_id(config)
    assert direct_run_directory(config).name == identifier
    assert smc_manifest(config)["decoder_id"] == identifier


def test_current_uniform_policy_and_future_explicit_policy() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_common.yaml"))
    nonuniform = copy.deepcopy(config)
    nonuniform["smc"]["tilted_component_weights_a"] = [0.25, 0.75]
    with pytest.raises(ValueError, match="1/J"):
        smc_manifest(nonuniform)
    nonuniform["smc"]["component_weight_policy"] = "explicit"
    assert smc_manifest(nonuniform)["tilted_component_aggregation"]["weights_a"] == [
        0.25,
        0.75,
    ]


def test_lambda_zero_is_diagnostic_not_aggregated() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_smoke_technical.yaml"))
    manifest = smc_manifest(config)
    assert 0.0 not in manifest["tilted_component_aggregation"]["tilted_lambdas"]
    assert manifest["diagnostic_components"]["lambdas"] == [0.0]
    assert manifest["diagnostic_components"][
        "included_in_tilted_component_aggregation"
    ] is False


def test_calibration_components_are_never_aggregated() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_lambda_calibration.yaml"))
    validate_benchmark_config(config)
    manifest = smc_manifest(config)
    assert manifest["experiment_phase"] == "calibration"
    assert manifest["calibration_components"] == {
        "lambdas": [0.5, 1.0, 2.0, 4.0],
        "aggregation_performed": False,
        "purpose": "terminal-lambda selection only",
    }
    assert "tilted_component_aggregation" not in manifest
    assert manifest["diagnostic_components"]["lambdas"] == [0.0]
    assert config["smc"]["num_particles"] == 256
    assert config["smc"]["num_independent_runs"] == 1


def test_progressive_lambda_calibration_is_dense_bounded_and_not_aggregated() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_lambda_progressive.yaml"))
    validate_benchmark_config(config)
    manifest = smc_manifest(config)
    assert config["smc"]["calibration_lambdas"] == [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ]
    assert max(config["smc"]["calibration_lambdas"]) == 2.0
    assert config["smc"]["num_particles"] == 256
    assert config["smc"]["num_independent_runs"] == 1
    assert manifest["experiment_phase"] == "calibration"
    assert manifest["calibration_components"]["aggregation_performed"] is False
    assert manifest["diagnostic_components"]["lambdas"] == [0.0]
    assert "tilted_component_aggregation" not in manifest


def test_100_token_estimation_uses_independent_components_and_uniform_average() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml"))
    validate_benchmark_config(config)
    smc = config["smc"]
    assert smc["experiment_phase"] == "estimation"
    assert smc["tilted_lambdas"] == [0.25, 0.5, 0.75, 1.0]
    assert smc["tilted_component_weights_a"] == [0.25] * 4
    assert smc["num_particles"] == 512
    assert smc["component_rng_policy"] == "independent_seed_per_run_and_component"
    seeds = [
        _component_seed(smc, run_index=0, component_index=index)
        for index in range(4)
    ]
    assert seeds == [67000, 67001, 67002, 67003]
    assert len(set(seeds)) == 4
    assert config["evaluation"]["reward"] == config["evaluation"]["event"] | {
        "definition": "log_softmax_probability_of_positive_toxicity_class"
    }
    manifest = smc_manifest(config)
    aggregation = manifest["tilted_component_aggregation"]
    assert aggregation["aggregation"] == "p_hat = sum_j a_j * p_hat_j"
    assert aggregation["weights_a"] == [0.25] * 4
    assert manifest["diagnostic_components"]["lambdas"] == []
    assert manifest["smc"]["tilted_sampler"] == {
        "type": "fk_bootstrap",
        "implementation_version": "fk_mdlm_multinomial_x0_k4_v1",
        "intermediate_x0_sampling": "mdlm_multinomial",
        "num_x0_samples": 4,
        "intermediate_reward_aggregation": "logmeanexp",
    }


def test_fk_config_rejects_nonpaper_intermediate_reward_policy() -> None:
    config = load_yaml(
        experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml")
    )
    config["smc"]["tilted_sampler"]["intermediate_x0_sampling"] = "argmax"
    with pytest.raises(ValueError, match="mdlm_multinomial"):
        validate_benchmark_config(config)


def test_tilted_sampler_is_interchangeable_without_changing_decoder_or_evaluator() -> None:
    baseline = load_yaml(
        experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml")
    )
    resample_move = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
        )
    )
    validate_benchmark_config(baseline)
    validate_benchmark_config(resample_move)
    assert baseline["smc"]["tilted_sampler"] == {"type": "fk_bootstrap"}
    assert resample_move["smc"]["tilted_sampler"]["type"] == (
        "trajectory_resample_move"
    )
    assert decoder_id(baseline) == decoder_id(resample_move)
    assert evaluator_id(baseline) == evaluator_id(resample_move)
    assert fingerprint(smc_manifest(baseline)) != fingerprint(smc_manifest(resample_move))
    for key in (
        "tilted_lambdas",
        "tilted_component_weights_a",
        "component_weight_policy",
        "num_particles",
    ):
        assert baseline["smc"][key] == resample_move["smc"][key]


def test_waste_free_profile_is_an_interchangeable_smc_backend() -> None:
    baseline = load_yaml(
        experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml")
    )
    waste_free = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_waste_free.yaml"
        )
    )
    validate_benchmark_config(waste_free)
    sampler = waste_free["smc"]["tilted_sampler"]
    assert sampler == {
        "type": "trajectory_waste_free",
        "implementation_version": "waste_free_trajectory_rollback_mh_v1",
        "num_tempering_levels": 8,
        "mutation_policy": "every_level",
        "chain_length": 4,
        "rollback_steps": [4, 8, 16, 24, 32],
        "rollback_probabilities": [0.2] * 5,
    }
    assert waste_free["smc"]["final_resampling"] is False
    assert waste_free["smc"]["num_particles"] == 512
    assert decoder_id(waste_free) == decoder_id(baseline)
    assert evaluator_id(waste_free) == evaluator_id(baseline)
    assert fingerprint(smc_manifest(waste_free)) != fingerprint(smc_manifest(baseline))


def test_waste_free_config_rejects_nonfactorable_chain_length() -> None:
    config = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_waste_free.yaml"
        )
    )
    config["smc"]["tilted_sampler"]["chain_length"] = 3
    with pytest.raises(ValueError, match="divisible"):
        validate_benchmark_config(config)


def test_waste_free_config_rejects_terminal_resampling() -> None:
    config = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_waste_free.yaml"
        )
    )
    config["smc"]["final_resampling"] = True
    with pytest.raises(ValueError, match="final_resampling: false"):
        validate_benchmark_config(config)


def test_matched_fk_profile_changes_only_sampler_and_keeps_512_particles() -> None:
    baseline = load_yaml(
        experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml")
    )
    matched = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_mh_matched_fk.yaml"
        )
    )
    validate_benchmark_config(matched)
    assert decoder_id(matched) == decoder_id(baseline)
    assert evaluator_id(matched) == evaluator_id(baseline)
    for key in (
        "num_particles",
        "microbatch_size",
        "tilted_lambdas",
        "tilted_component_weights_a",
        "num_independent_runs",
        "base_seed",
    ):
        assert matched["smc"][key] == baseline["smc"][key]
    sampler = matched["smc"]["tilted_sampler"]
    assert sampler == {
        "type": "trajectory_resample_move",
        "implementation_version": "exact_trajectory_rollback_mh_v1",
        "num_tempering_levels": 8,
        "mcmc_steps_per_level": 1,
        "mutation_policy": "after_resampling",
        "rollback_steps": [4, 8],
        "rollback_probabilities": [0.5, 0.5],
    }


def test_trajectory_sampler_requires_implementation_version() -> None:
    config = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
        )
    )
    del config["smc"]["tilted_sampler"]["implementation_version"]
    with pytest.raises(ValueError, match="implementation_version"):
        validate_benchmark_config(config)


def test_trajectory_sampler_config_rejects_nonexistent_rollback_checkpoint() -> None:
    config = load_yaml(
        experiment_config(
            "mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
        )
    )
    config["smc"]["tilted_sampler"]["rollback_steps"] = [33]
    config["smc"]["tilted_sampler"]["rollback_probabilities"] = [1.0]
    with pytest.raises(ValueError, match="within denoising_steps"):
        validate_benchmark_config(config)


class _PrefixTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(str(value) for value in token_ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def score_texts(self, texts, *, max_length):
        self.calls.append((list(texts), max_length))
        return [
            {
                "reward": -0.5,
                "logits": [0.5, -0.5],
                "toxic_probability": 0.25,
                "is_event": False,
            }
            for _ in texts
        ]


def test_reward_and_event_use_the_identical_100_mdlm_token_text() -> None:
    tokenizer = _PrefixTokenizer()
    evaluator = _RecordingEvaluator()
    functions = UnconditionalToxicityFunctions(
        generator_tokenizer=tokenizer,
        evaluator=evaluator,
        reward_completion_tokens=100,
        event_completion_tokens=100,
        reward_max_length=512,
        event_max_length=512,
        event_text_scope="first_generated_mdlm_tokens_only",
        completion_trimming_tokenizer=object(),
    )
    tokens = torch.arange(128).repeat(2, 1)
    functions.reward(tokens)
    functions.event(tokens)
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][1] == 512
    assert len(evaluator.calls[0][0][0].split()) == 100
    assert functions.last_event_rows is functions.last_reward_rows
    assert functions.last_event_token_ids == [list(range(100)), list(range(100))]


def test_classifier_reuse_requires_identical_texts_and_max_length() -> None:
    assert identical_classifier_inputs(
        ["a", "b"],
        first_max_length=512,
        second_texts=["a", "b"],
        second_max_length=512,
    )
    assert not identical_classifier_inputs(
        ["a", "b"],
        first_max_length=512,
        second_texts=["a", "different"],
        second_max_length=512,
    )
    assert not identical_classifier_inputs(
        ["a", "b"],
        first_max_length=50,
        second_texts=["a", "b"],
        second_max_length=512,
    )


def test_mdlm_prefix_classifier_text_does_not_retokenize() -> None:
    text, token_ids = unconditional_mdlm_prefix_classifier_text(
        token_ids=list(range(128)),
        decoder_tokenizer=_PrefixTokenizer(),
        completion_tokens=100,
    )
    assert token_ids == list(range(100))
    assert len(text.split()) == 100


def test_estimation_template_is_provisional_uniform_and_excludes_lambda_zero() -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_estimation_template.yaml"))
    validate_benchmark_config(config)
    manifest = smc_manifest(config)
    aggregation = manifest["tilted_component_aggregation"]
    assert aggregation["tilted_lambdas"] == [0.5, 1.0, 2.0]
    assert aggregation["weights_a"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert 0.0 not in aggregation["tilted_lambdas"]
    assert manifest["diagnostic_components"]["lambdas"] == [0.0]
    assert config["smc"]["num_particles"] == 256
    assert config["smc"]["microbatch_size"] == 16


def test_calibration_and_direct_32_use_identical_decoder() -> None:
    calibration = load_yaml(experiment_config("mdlm_owt_unconditional_lambda_calibration.yaml"))
    direct = load_yaml(experiment_config("mdlm_owt_unconditional_direct_32.yaml"))
    assert decoder_id(calibration) == decoder_id(direct)


def test_calibration_report_keeps_components_separate(tmp_path: Path) -> None:
    config = load_yaml(experiment_config("mdlm_owt_unconditional_lambda_calibration.yaml"))
    decoder = decoder_id(config)
    evaluator = evaluator_id(config)
    direct_run = tmp_path / "direct"
    eval_dir = direct_run / "evaluations" / evaluator
    _write_summary_snapshot(
        eval_dir,
        {
            "decoder_id": decoder,
            "evaluator_id": evaluator,
            "conditioning_mode": "unconditional",
            "evaluated_samples": 100,
            "empirical_probability": 0.1,
        },
    )

    def component(role: str, lam: float, fraction: float) -> dict:
        return {
            "component_role": role,
            "lambda": lam,
            "num_particles": 64,
            "tilted_event_fraction_not_p0_estimate": fraction,
            "tilted_event_discoveries": int(64 * fraction),
            "reverse_snis_estimate": 0.1,
            "reverse_weight_ess": 32.0,
            "reverse_weight_ess_over_n": 0.5,
            "resampling_count": 1,
            "distinct_root_ancestors": 40,
            "distinct_root_ancestor_fraction": 0.625,
            "pre_final_resampling_max_normalized_weight": 0.1,
            "reverse_max_normalized_weight": 0.2,
            "reward_quantiles": {
                "q00": -5.0,
                "q10": -4.0,
                "q25": -3.0,
                "q50": -2.0,
                "q75": -1.0,
                "q90": -0.5,
                "q100": -0.1,
            },
            "runtime_seconds": 1.0,
            "nfe": 40,
            "classifier_model_forward_calls": 10,
            "classifier_texts_scored": 128,
            "peak_cuda_memory_allocated_bytes": 123,
            "nonfinite_detected": False,
        }

    calibration_run = tmp_path / "calibration"
    calibration_run.mkdir()
    (calibration_run / "summary.json").write_text(
        json.dumps(
            {
                "decoder_id": decoder,
                "evaluator_id": evaluator,
                "conditioning_mode": "unconditional",
                "experiment_phase": "calibration",
                "aggregation_performed": False,
                "diagnostic_components": [component("diagnostic", 0.0, 0.125)],
                "calibration_components": [component("calibration", 1.0, 0.25)],
            }
        ),
        encoding="utf-8",
    )
    report, destination = build_calibration_report(
        config,
        direct_run=direct_run,
        calibration_run=calibration_run,
    )
    assert report["aggregation_performed"] is False
    assert report["selection_performed"] is False
    assert [row["component_role"] for row in report["components"]] == [
        "diagnostic",
        "calibration",
    ]
    assert "aggregate" not in report
    assert (destination / "direct_00000100.csv").exists()


def test_setuptools_discovers_only_explicit_python_packages() -> None:
    repository_root = REPOSITORY_ROOT
    with (repository_root / "pyproject.toml").open("rb") as handle:
        discovery = tomllib.load(handle)["tool"]["setuptools"]["packages"]["find"]
    discovered = set()
    for location in discovery["where"]:
        discovered.update(
            find_packages(
                where=str(repository_root / location),
                include=tuple(discovery["include"]),
                exclude=tuple(discovery["exclude"]),
            )
        )
    assert discovered == {
        "rea_llms",
        "rea_llms.observables",
        "suffix_direct_mh_smc_experiment",
        "official_annealed_tps_experiment",
        "diffusion_toxicity_smc_experiment",
    }


def test_comparison_rejects_different_decoder_ids(tmp_path: Path) -> None:
    config = minimal_config()
    evaluator = evaluator_id(config)
    direct_run = tmp_path / "direct"
    evaluation_dir = direct_run / "evaluations" / evaluator
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "decoder_id": "prompt-conditioned-old-id",
                "evaluator_id": evaluator,
                "conditioning_mode": "unconditional",
            }
        ),
        encoding="utf-8",
    )
    smc_run = tmp_path / "smc"
    smc_run.mkdir()
    (smc_run / "summary.json").write_text(
        json.dumps(
            {
                "decoder_id": decoder_id(config),
                "evaluator_id": evaluator,
                "conditioning_mode": "unconditional",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="decoder_id"):
        compare_existing(config, direct_run=direct_run, smc_run=smc_run)


def test_serious_suite_is_three_method_multi_budget_multi_seed() -> None:
    suite_path = experiment_config(
        "mdlm_owt_unconditional_serious_comparison.yaml"
    ).resolve()
    with suite_path.open(encoding="utf-8") as handle:
        import yaml

        suite = yaml.safe_load(handle)
    entries = resolved_comparison_entries(suite, suite_path)
    manifest = comparison_manifest(suite, suite_path)
    assert suite["direct"]["target_samples"] == 200_000
    assert suite["smc"]["particle_budgets"] == [64, 128, 256, 512]
    assert suite["smc"]["independent_runs"] == 8
    assert len(entries) == 12
    assert {entry["method_key"] for entry in entries} == {
        "fk_bootstrap",
        "trajectory_resample_move",
        "trajectory_waste_free",
    }
    assert len({entry["smc_id"] for entry in entries}) == 12
    assert manifest["decoder_id"] == "cb78b724cdaa4a7ec1c91955be80de5e169f77ab776a98ec3ae1dc6aa0cfb609"
    assert manifest["evaluator_id"] == "91e750abf0a456123b42a8aec27719a35e80950dd4c7387e6a3f47a62084691d"
    base_seeds = {
        (entry["method_key"], entry["particle_budget"]): entry["config"]["smc"][
            "base_seed"
        ]
        for entry in entries
    }
    assert base_seeds == {
        (method, budget): 91_000 + method_offset + 10_000_000 * budget_index
        for method, method_offset in {
            "fk_bootstrap": 0,
            "trajectory_resample_move": 100_000_000,
            "trajectory_waste_free": 200_000_000,
        }.items()
        for budget_index, budget in enumerate([64, 128, 256, 512])
    }
    component_seeds = {
        _component_seed(
            entry["config"]["smc"],
            run_index=run_index,
            component_index=component_index,
        )
        for entry in entries
        for run_index in range(8)
        for component_index in range(4)
    }
    assert len(component_seeds) == 12 * 8 * 4
    for entry in entries:
        validate_benchmark_config(entry["config"])
        assert entry["config"]["smc"]["tilted_lambdas"] == [0.25, 0.5, 0.75, 1.0]
        assert entry["config"]["smc"]["tilted_component_weights_a"] == [
            0.25,
            0.25,
            0.25,
            0.25,
        ]


def test_final_confirmatory_suite_uses_disjoint_multi_budget_seeds() -> None:
    suite_path = experiment_config(
        "mdlm_owt_unconditional_final_confirmatory.yaml"
    ).resolve()
    suite = load_suite(suite_path)
    entries = resolved_comparison_entries(suite, suite_path)

    assert suite["smc"]["particle_budgets"] == [192, 512]
    assert suite["smc"]["independent_runs"] == 32
    assert suite["smc"]["budget_seed_stride"] == 50_000_000
    assert {
        (entry["method_key"], entry["particle_budget"]) for entry in entries
    } == {
        ("fk_bootstrap", 192),
        ("fk_bootstrap", 512),
        ("trajectory_resample_move", 192),
        ("trajectory_resample_move", 512),
    }
    seeds = planned_component_seeds(entries)
    assert len(seeds) == 4 * 32 * 4

    prior_paths = [
        experiment_config("mdlm_owt_unconditional_serious_comparison.yaml").resolve(),
        experiment_config("mdlm_owt_unconditional_submission_long.yaml").resolve(),
        experiment_config(
            "mdlm_owt_unconditional_submission_matched_runtime.yaml"
        ).resolve(),
    ]
    prior_seeds: set[int] = set()
    for prior_path in prior_paths:
        prior_entries = resolved_comparison_entries(load_suite(prior_path), prior_path)
        prior_seeds.update(planned_component_seeds(prior_entries))
    assert not seeds.intersection(prior_seeds)


def test_all_serious_configs_survive_child_process_json_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_path = experiment_config(
        "mdlm_owt_unconditional_serious_comparison.yaml"
    ).resolve()
    with suite_path.open(encoding="utf-8") as handle:
        import yaml

        suite = yaml.safe_load(handle)
    invocations: list[Path] = []

    def fake_subprocess_run(command, check):
        assert check is True
        config_path = Path(command[command.index("--config") + 1])
        output_root = Path(command[command.index("--output-root") + 1])
        child_config = load_yaml(config_path)
        child_run_dir = smc_run_directory(child_config, output_root)
        child_run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            child_run_dir / "summary.json",
            {
                "decoder_id": decoder_id(child_config),
                "evaluator_id": evaluator_id(child_config),
                "smc_id": fingerprint(smc_manifest(child_config)),
            },
        )
        invocations.append(config_path)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "diffusion_toxicity_smc_experiment.serious_comparison.subprocess.run",
        fake_subprocess_run,
    )

    output_root = tmp_path / "outputs"
    if os.name == "nt":
        output_root = Path("\\\\?\\" + str(output_root.resolve()))
    suite_dir = run_comparison_suite(
        suite, suite_path=suite_path, output_root=output_root
    )

    assert len(invocations) == 12
    completion = json.loads((suite_dir / "completion.json").read_text(encoding="utf-8"))
    assert completion["completed_grid_items"] == 12


def test_real_gpu_validation_suite_covers_every_method_and_two_budgets() -> None:
    suite_path = experiment_config(
        "mdlm_owt_unconditional_three_smc_gpu_validation.yaml"
    ).resolve()
    with suite_path.open(encoding="utf-8") as handle:
        import yaml

        suite = yaml.safe_load(handle)
    entries = resolved_comparison_entries(suite, suite_path)

    assert suite["smc"]["particle_budgets"] == [16, 64]
    assert suite["smc"]["independent_runs"] == 2
    assert len(entries) == 6
    assert {
        (entry["method_key"], entry["particle_budget"]) for entry in entries
    } == {
        (method, budget)
        for method in {
            "fk_bootstrap",
            "trajectory_resample_move",
            "trajectory_waste_free",
        }
        for budget in {16, 64}
    }
    assert len({entry["smc_id"] for entry in entries}) == 6
    for entry in entries:
        validate_benchmark_config(entry["config"])


def test_serious_report_builds_csv_json_and_boxplots(tmp_path: Path) -> None:
    direct_generation_path = experiment_config(
        "mdlm_owt_unconditional_direct_32.yaml"
    ).resolve()
    evaluation_path = experiment_config(
        "mdlm_owt_unconditional_estimation_100tok.yaml"
    ).resolve()
    method_specs = [
        (
            "fk_bootstrap",
            "FK Steering",
            experiment_config("mdlm_owt_unconditional_estimation_100tok.yaml").resolve(),
            0,
        ),
        (
            "trajectory_resample_move",
            "Trajectory MH",
            experiment_config(
                "mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
            ).resolve(),
            100_000,
        ),
        (
            "trajectory_waste_free",
            "Waste-Free SMC",
            experiment_config(
                "mdlm_owt_unconditional_estimation_100tok_trajectory_waste_free.yaml"
            ).resolve(),
            200_000,
        ),
    ]
    suite = {
        "schema_version": 1,
        "suite_name": "synthetic-serious-report",
        "direct": {
            "generation_config": str(direct_generation_path),
            "evaluation_config": str(evaluation_path),
            "target_samples": 8,
            "reference_block_size": 4,
        },
        "smc": {
            "particle_budgets": [4],
            "independent_runs": 2,
            "base_seed": 900,
            "methods": [
                {
                    "key": key,
                    "label": label,
                    "config": str(config),
                    "seed_offset": offset,
                }
                for key, label, config, offset in method_specs
            ],
        },
        "report": {},
    }
    suite_path = tmp_path / "suite.yaml"
    output_root = tmp_path / "outputs"
    if os.name == "nt":
        output_root = Path("\\\\?\\" + str(output_root.resolve()))
    generation_config = load_yaml(direct_generation_path)
    evaluation_config = load_yaml(evaluation_path)
    direct_dir = direct_run_directory(generation_config, output_root)
    identifier = decoder_id(generation_config)
    initialize_manifest(
        direct_dir,
        {
            "decoder_id": identifier,
            "decoder_manifest": decoder_manifest(generation_config),
            "conditioning_mode": "unconditional",
            "storage_schema": "unconditional_direct_generation_shards_v1",
        },
    )
    generation_rows = []
    score_rows = []
    for index in range(8):
        sample_id = f"sample-{index}"
        generation_rows.append(
            {
                "sample_id": sample_id,
                "sample_index": index,
                "conditioning_mode": "unconditional",
                "shard_seed": 1000 + index,
                "position_in_shard": index,
                "token_ids": [index + 1] * 128,
                "completion_token_ids": [index + 1] * 128,
                "decoded_text": f"sample {index}",
                "checkpoint_revision": generation_config["decoder"]["checkpoint"]["revision"],
                "tokenizer_revision": generation_config["decoder"]["tokenizer"]["revision"],
                "decoder_id": identifier,
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
            }
        )
        score_rows.append(
            {
                "sample_id": sample_id,
                "conditioning_mode": "unconditional",
                "decoder_id": identifier,
                "evaluator_id": evaluator_id(evaluation_config),
                "is_event": index in {1, 6},
            }
        )
    (direct_dir / "generations").mkdir(parents=True)
    atomic_write_jsonl(direct_dir / "generations" / "shard_00000000_00000007.jsonl", generation_rows)
    eval_dir = direct_dir / "evaluations" / evaluator_id(evaluation_config)
    eval_dir.mkdir(parents=True)
    atomic_write_jsonl(eval_dir / "scores_00000000_00000007.jsonl", score_rows)

    entries = resolved_comparison_entries(suite, suite_path)
    for method_index, entry in enumerate(entries):
        components = []
        run_estimates = []
        for run_index in range(2):
            run_estimate = 0.21 + 0.02 * method_index + 0.08 * run_index
            run_estimates.append(run_estimate)
            for component_index, value in enumerate((0.25, 0.5, 0.75, 1.0)):
                proposals = 0 if method_index == 0 else 20 + component_index
                components.append(
                    {
                        "run_index": run_index,
                        "lambda": value,
                        "seed": entry["config"]["smc"]["base_seed"] + 1_000_000 * run_index + component_index,
                        "reverse_snis_estimate": run_estimate,
                        "runtime_seconds": 2.0 + method_index + 0.1 * run_index,
                        "total_decoder_particle_steps": 128 + 32 * method_index,
                        "classifier_texts_scored": 32 + 8 * method_index,
                        "reverse_weight_ess_over_n": 0.8 - 0.1 * method_index,
                        "distinct_root_ancestor_fraction": 0.9 - 0.1 * method_index,
                        "normalized_root_ancestor_entropy": 0.95 - 0.1 * method_index,
                        "tilted_event_discoveries": 1 + method_index,
                        "resampling_count": method_index,
                        "mcmc_proposals": proposals,
                        "mcmc_accepted": int(0.8 * proposals),
                        "peak_cuda_memory_allocated_bytes": 1_000_000 + method_index,
                        "nfe": 8 + method_index,
                    }
                )
        run_dir = smc_run_directory(entry["config"], output_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_dir / "summary.json",
            {
                "decoder_id": identifier,
                "evaluator_id": evaluator_id(evaluation_config),
                "smc_id": entry["smc_id"],
                "tilted_components": components,
                "aggregate": {"run_estimates": run_estimates},
            },
        )
    suite_dir = comparison_run_directory(suite, suite_path, output_root)
    manifest = comparison_manifest(suite, suite_path)
    initialize_manifest(suite_dir, {"suite_id": fingerprint(manifest), **manifest})
    atomic_write_json(suite_dir / "completion.json", {"complete": True})

    report_dir = build_comparison_report(
        suite, suite_path=suite_path, output_root=output_root
    )
    report = json.loads((report_dir / "serious_report.json").read_text(encoding="utf-8"))
    assert report["direct_reference"]["empirical_probability"] == 0.25
    assert len(report["group_summary"]) == 3
    assert len(list(report_dir.glob("*.csv"))) == 5
    assert len(list(report_dir.glob("*.svg"))) == 8
    assert "<svg" in (report_dir / "estimate_boxplot.svg").read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.gpu
def test_optional_real_unconditional_model_smoke_is_explicitly_opt_in() -> None:
    if os.environ.get("RUN_MDLM_GPU_INTEGRATION") != "1":
        pytest.skip("set RUN_MDLM_GPU_INTEGRATION=1 with pinned artifacts to run")
    config = load_yaml(experiment_config("mdlm_owt_unconditional_smoke.yaml"))
    runtime = load_generator_runtime(config, local_files_only=True)
    decoder = config["decoder"]
    sampler = NativeMDLMSampler(
        runtime.model,
        mask_token_id=int(decoder["mask_token_id"]),
        predictor="ddpm",
        final_noise_removal=True,
    )
    tokens = sampler.sample(
        batch_size=1,
        sequence_length=int(decoder["sequence_length"]),
        num_steps=2,
        fixed_prefix_token_ids=torch.empty(0, dtype=torch.long, device=runtime.device),
        device=runtime.device,
        generator=torch.Generator(device="cuda").manual_seed(1),
    )
    assert tokens.shape == (1, int(decoder["sequence_length"]))
    assert not bool((tokens == int(decoder["mask_token_id"])).any())


def test_stored_particles_can_be_rethresholded_without_resampling() -> None:
    particles = [
        {
            "reverse_corrected_log_weight": math.log(0.25),
            "event_toxic_probability": 0.80,
        },
        {
            "reverse_corrected_log_weight": math.log(0.75),
            "event_toxic_probability": 0.95,
        },
    ]
    estimate, discoveries, ess = component_threshold_estimate(particles, 0.90)
    assert estimate == pytest.approx(0.75)
    assert discoveries == 1
    assert ess == pytest.approx(1.0 / (0.25**2 + 0.75**2))


def test_rethresholding_rejects_nonfinite_continuous_scores() -> None:
    particles = [
        {
            "reverse_corrected_log_weight": 0.0,
            "event_toxic_probability": math.nan,
        }
    ]
    with pytest.raises(FloatingPointError, match="toxic probabilities"):
        component_threshold_estimate(particles, 0.90)


def test_direct_threshold_summary_uses_continuous_toxicity_score() -> None:
    rows = [
        {"event_toxic_probability": value}
        for value in (0.10, 0.50, 0.90, 0.99)
    ]
    summaries = direct_threshold_summaries(rows, (0.50, 0.95))
    assert summaries[0]["event_discoveries"] == 3
    assert summaries[0]["empirical_probability"] == pytest.approx(0.75)
    assert summaries[1]["event_discoveries"] == 1
    assert summaries[1]["empirical_probability"] == pytest.approx(0.25)


def test_inverse_variance_pilot_weights_are_valid_and_favour_stability() -> None:
    fitted = inverse_variance_weights(
        (
            (0.010, 0.011, 0.009, 0.010),
            (0.005, 0.020, 0.001, 0.025),
            (0.008, 0.015, 0.004, 0.018),
            (0.006, 0.017, 0.003, 0.022),
        ),
        shrinkage=0.25,
        floor_fraction=0.05,
    )
    weights = fitted["weights"]
    assert all(value >= 0.0 for value in weights)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == max(weights)


def test_pilot_weight_fit_rejects_misaligned_component_runs() -> None:
    rows = [
        {
            "method_key": "fk_bootstrap",
            "particle_budget": 512,
            "threshold": 0.9,
            "lambda": lambda_value,
            "run_index": run_index,
            "estimate": 0.01 + 0.001 * run_index,
        }
        for lambda_value, run_index in ((0.25, 0), (0.25, 1), (0.5, 0), (0.5, 2))
    ]
    with pytest.raises(ValueError, match="same run set"):
        fit_pilot_weights(rows, shrinkage=0.25, floor_fraction=0.05)


def test_submission_suite_excludes_waste_free_and_uses_disjoint_test_seeds() -> None:
    study_path = experiment_config(
        "mdlm_owt_unconditional_submission_probability.yaml"
    ).resolve()
    study = load_study(study_path)
    manifest = study_manifest(study, study_path)
    assert manifest["methods"] == ["fk_bootstrap", "trajectory_resample_move"]

    pilot_path = experiment_config(
        "mdlm_owt_unconditional_serious_comparison.yaml"
    ).resolve()
    test_path = experiment_config(
        "mdlm_owt_unconditional_submission_long.yaml"
    ).resolve()
    pilot_entries = [
        entry
        for entry in resolved_comparison_entries(load_suite(pilot_path), pilot_path)
        if entry["method_key"] in set(manifest["methods"])
        and entry["particle_budget"] == 512
    ]
    test_entries = resolved_comparison_entries(load_suite(test_path), test_path)
    assert len(test_entries) == 2
    assert all(entry["particle_budget"] == 512 for entry in test_entries)
    assert all(entry["config"]["smc"]["independent_runs"] == 32 for entry in test_entries)
    assert not planned_component_seeds(pilot_entries).intersection(
        planned_component_seeds(test_entries)
    )


def test_legacy_serious_suite_keeps_its_avon_artifact_identity() -> None:
    suite_path = experiment_config(
        "mdlm_owt_unconditional_serious_comparison.yaml"
    ).resolve()
    manifest = comparison_manifest(load_suite(suite_path), suite_path)
    avon_config_root = (
        "/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/"
        "experiments/mdlm_toxicity_rare_event/configs/"
    )
    manifest["direct"]["generation_config_path"] = (
        avon_config_root + "mdlm_owt_unconditional_direct_32.yaml"
    )
    manifest["direct"]["evaluation_config_path"] = (
        avon_config_root + "mdlm_owt_unconditional_estimation_100tok.yaml"
    )
    assert fingerprint(manifest) == (
        "cfc61727355e356a4a961dbabf0fee7b5395b8992da506e40a42fdc6a307095d"
    )
