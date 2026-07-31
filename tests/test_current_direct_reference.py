from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import rea_llms.current_direct_reference as direct
import suffix_direct_mh_smc_experiment.smc_sampler as smc
from rea_llms.data_reuse_audit import prompt_sha256 as audit_prompt_sha256
from rea_llms.models import DEFAULT_MODEL_NAME, DEFAULT_PROMPT, DEFAULT_TOKENIZER_NAME
from suffix_direct_mh_smc_experiment.run_tail_validation import (
    ComponentSpec,
    _component_metadata,
    build_lambda_schedule,
    component_run_id,
    component_seed,
    parse_args as parse_validation_args,
)


class ParameterOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class PromptTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[4, 5]], dtype=torch.long)}


def _direct_config(**changes) -> direct.CurrentDirectConfig:
    values = {
        "num_samples": 5,
        "batch_size": 2,
        "max_new_tokens": 4,
        "seed": 17,
        "thresholds": (6.0, 7.0),
    }
    values.update(changes)
    return direct.CurrentDirectConfig(**values)


def test_current_direct_and_smc_share_generation_and_scoring_primitives() -> None:
    assert direct.generate_fixed_length_completions is smc.generate_fixed_length_completions
    assert direct.score_completion_tokens is smc.score_completion_tokens


def test_current_direct_and_smc_metadata_share_prompt_and_length_policy() -> None:
    config = _direct_config()
    direct_metadata = direct.requested_metadata(
        model_name=DEFAULT_MODEL_NAME,
        tokenizer_name=DEFAULT_TOKENIZER_NAME,
        prompt=DEFAULT_PROMPT,
        config=config,
        device_type="cpu",
    )
    args = parse_validation_args(
        [
            "--seed-groups",
            "200",
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--num-particles",
            "2",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            "1,4",
        ]
    )
    spec = ComponentSpec(cap=6.0, lambda_final=-0.5)
    smc_metadata = _component_metadata(
        args,
        spec,
        seed_group=200,
        component_index=0,
        seed=component_seed(200, 0),
        schedule=build_lambda_schedule(-0.5, 0.25),
    )
    assert direct_metadata["prompt_sha256"] == audit_prompt_sha256(
        smc_metadata["prompt"]
    )
    assert direct_metadata["prompt"] == smc_metadata["prompt"]
    assert direct_metadata["max_new_tokens"] == smc_metadata["max_new_tokens"]
    assert direct_metadata["fixed_completion_length"] is True
    assert smc_metadata["fixed_completion_length"] is True
    assert direct_metadata["eos_stopping"] is False


def test_official_public_direct_and_smc_share_scoring_metadata() -> None:
    config = _direct_config(ari_convention="official_public")
    direct_metadata = direct.requested_metadata(
        model_name=DEFAULT_MODEL_NAME,
        tokenizer_name=DEFAULT_TOKENIZER_NAME,
        prompt=DEFAULT_PROMPT,
        config=config,
        device_type="cuda",
    )
    args = parse_validation_args(
        [
            "--ari-convention",
            "official_public",
            "--seed-groups",
            "500",
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--num-particles",
            "2",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            f"1,{config.max_new_tokens}",
        ]
    )
    spec = ComponentSpec(cap=7.0, lambda_final=-1.5)
    smc_metadata = _component_metadata(
        args,
        spec,
        seed_group=500,
        component_index=0,
        seed=component_seed(500, 0),
        schedule=build_lambda_schedule(-1.5, 0.25),
    )

    assert direct.score_completion_tokens is smc.score_completion_tokens
    assert direct_metadata["ari_convention"] == "official_public"
    assert smc_metadata["ari_convention"] == "official_public"
    assert direct_metadata["prompt_sha256"] == smc_metadata["prompt_sha256"]
    assert direct_metadata["decoding_convention"] == smc_metadata[
        "decoding_convention"
    ]
    assert "decode the full prompt-token plus completion-token sequence" in smc_metadata[
        "decoding_convention"
    ]
    assert smc_metadata["ari_cap"] == 7.0
    assert smc_metadata["lambda_final"] == -1.5

    current_args = parse_validation_args(
        [
            "--seed-groups",
            "500",
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--num-particles",
            "2",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            f"1,{config.max_new_tokens}",
        ]
    )
    current_metadata = _component_metadata(
        current_args,
        spec,
        seed_group=500,
        component_index=0,
        seed=component_seed(500, 0),
        schedule=build_lambda_schedule(-1.5, 0.25),
    )
    assert current_metadata != smc_metadata
    assert component_run_id(current_metadata) != component_run_id(smc_metadata)


def test_current_direct_cache_rejects_different_metadata(tmp_path: Path) -> None:
    config = _direct_config()
    requested = direct.requested_metadata(
        model_name=DEFAULT_MODEL_NAME,
        tokenizer_name=DEFAULT_TOKENIZER_NAME,
        prompt=DEFAULT_PROMPT,
        config=config,
        device_type="cpu",
    )
    values = np.asarray([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    running = direct.RunningARIStatistics(config.thresholds)
    running.update(values)
    ari_path = tmp_path / "direct_ari.npz"
    direct._write_npz_atomic(ari_path, values)
    summary = direct.build_summary(
        values,
        running,
        requested,
        ari_file_sha256=direct._sha256_file(ari_path),
    )
    direct._write_json_atomic(tmp_path / "summary.json", summary)
    assert direct.load_reusable_summary(tmp_path, requested) is not None

    changed = dict(requested)
    changed["prompt"] = DEFAULT_PROMPT + " "
    changed["prompt_sha256"] = direct.prompt_sha256(changed["prompt"])
    assert direct.load_reusable_summary(tmp_path, changed) is None

    official = direct.requested_metadata(
        model_name=DEFAULT_MODEL_NAME,
        tokenizer_name=DEFAULT_TOKENIZER_NAME,
        prompt=DEFAULT_PROMPT,
        config=_direct_config(ari_convention="official_public"),
        device_type="cpu",
    )
    assert direct.load_reusable_summary(tmp_path, official) is None


def test_current_direct_tail_counts_and_wilson_intervals_on_toy_data() -> None:
    values = np.asarray([5.0, 6.0, 6.5, 8.0], dtype=np.float32)
    running = direct.RunningARIStatistics((6.0, 9.0))
    running.update(values[:2])
    running.update(values[2:])
    summary = direct.build_summary(
        values,
        running,
        {"toy": True},
        ari_file_sha256="toy",
        runtime_seconds=12.5,
        generated_token_count=400,
    )
    tails = summary["statistics"]["tails"]
    assert tails[0]["count"] == 3
    assert tails[0]["empirical_probability"] == pytest.approx(0.75)
    assert tails[0]["wilson_95_lower"] < 0.75 < tails[0]["wilson_95_upper"]
    assert tails[1]["count"] == 0
    assert tails[1]["wilson_95_lower"] == pytest.approx(0.0)
    assert summary["runtime_seconds"] == pytest.approx(12.5)
    assert summary["generated_token_count"] == 400
    assert summary["statistics"]["quantiles"]["0.5"] == pytest.approx(6.25)


def test_current_direct_sampler_batches_to_exact_requested_count(monkeypatch) -> None:
    generated_batch_sizes: list[int] = []

    def fake_generate(
        model,
        prompt_ids,
        *,
        num_sequences,
        max_new_tokens,
        batch_size,
        generator,
    ):
        generated_batch_sizes.append(num_sequences)
        return torch.ones((num_sequences, max_new_tokens), dtype=torch.long)

    next_value = 0

    def fake_score(tokenizer, observable, *, prompt, prompt_ids, completions):
        nonlocal next_value
        count = completions.shape[0]
        raw = torch.arange(next_value, next_value + count, dtype=torch.float32)
        next_value += count
        return raw, raw, [None] * count

    monkeypatch.setattr(direct, "generate_fixed_length_completions", fake_generate)
    monkeypatch.setattr(direct, "score_completion_tokens", fake_score)
    progress: list[int] = []
    values, running = direct.sample_current_direct_reference(
        ParameterOnlyModel(),
        PromptTokenizer(),
        object(),
        prompt=DEFAULT_PROMPT,
        config=_direct_config(),
        progress_callback=lambda done, total: progress.append(done),
    )
    assert generated_batch_sizes == [2, 2, 1]
    assert values.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert running.count == 5
    assert progress == [2, 4, 5]


def test_current_direct_cli_defaults_do_not_execute_sampling() -> None:
    args = direct.parse_args(["--output-dir", "unused"])
    assert args.num_samples == 200_000
    assert args.skip_completed is True
    assert callable(direct.main)
