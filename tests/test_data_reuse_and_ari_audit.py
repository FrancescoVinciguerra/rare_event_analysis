from __future__ import annotations

import json
from pathlib import Path

from rea_llms.current_direct_reference import CurrentDirectConfig, requested_metadata
from rea_llms.data_reuse_audit import (
    assess_data_reuse,
    audit_run_directory,
    prompt_sha256,
)
from rea_llms.observables.ari import ARIObservable
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTION_OFFICIAL_PUBLIC,
    text_for_ari_from_token_ids,
)


PROMPT = "Once upon a time, in a big forest, there lived a rhinoc"


class BoundaryTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        ids = tuple(int(value) for value in token_ids)
        self.calls.append((ids, skip_special_tokens))
        return {
            (1, 2): "Hello world",
            (2,): "world",
        }[ids]


class LeadingSpaceTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        return {
            (1, 2): "Hello world",
            (2,): " world",
        }[tuple(int(value) for value in token_ids)]


def test_full_sequence_and_completion_only_decoding_can_differ() -> None:
    tokenizer = BoundaryTokenizer()
    current = text_for_ari_from_token_ids(
        tokenizer,
        prompt="Hello",
        token_ids=[1, 2],
        prompt_token_count=1,
        convention=ARI_CONVENTION_CURRENT,
    )
    official = text_for_ari_from_token_ids(
        tokenizer,
        prompt="Hello",
        token_ids=[1, 2],
        prompt_token_count=1,
        convention=ARI_CONVENTION_OFFICIAL_PUBLIC,
    )
    assert current == "Helloworld"
    assert official == "Hello world"
    assert all(skip_special_tokens for _, skip_special_tokens in tokenizer.calls)


def test_prompt_completion_boundary_preserves_tokenizer_leading_space() -> None:
    tokenizer = LeadingSpaceTokenizer()
    current = text_for_ari_from_token_ids(
        tokenizer,
        prompt="Hello",
        token_ids=[1, 2],
        prompt_token_count=1,
        convention=ARI_CONVENTION_CURRENT,
    )
    official = text_for_ari_from_token_ids(
        tokenizer,
        prompt="Hello",
        token_ids=[1, 2],
        prompt_token_count=1,
        convention=ARI_CONVENTION_OFFICIAL_PUBLIC,
    )
    assert current == official == "Hello world"


def test_exact_paper_prompt_hash() -> None:
    assert repr(PROMPT) == "'Once upon a time, in a big forest, there lived a rhinoc'"
    assert prompt_sha256(PROMPT) == (
        "b4b7f60343cd3d1e24d8f2e312810afdaaa28034b9a194c66865b47483f70f59"
    )


def test_same_text_produces_same_ari_across_conventions(monkeypatch) -> None:
    monkeypatch.setattr(
        "rea_llms.observables.ari.automated_readability_index",
        lambda text: len(text) / 10,
    )
    current = ARIObservable(mode="raw", convention=ARI_CONVENTION_CURRENT)
    official = ARIObservable(mode="raw", convention=ARI_CONVENTION_OFFICIAL_PUBLIC)
    assert current.score_text("identical text").value == official.score_text(
        "identical text"
    ).value


def test_ari_conventions_are_separate_in_scores_and_direct_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "rea_llms.observables.ari.automated_readability_index",
        lambda text: 3.0,
    )
    current_score = ARIObservable(convention=ARI_CONVENTION_CURRENT).score_text("x")
    official_score = ARIObservable(
        convention=ARI_CONVENTION_OFFICIAL_PUBLIC
    ).score_text("x")
    assert current_score.metadata["ari_convention"] == ARI_CONVENTION_CURRENT
    assert official_score.metadata["ari_convention"] == ARI_CONVENTION_OFFICIAL_PUBLIC

    current = requested_metadata(
        model_name="model",
        tokenizer_name="tokenizer",
        prompt="prompt",
        config=CurrentDirectConfig(num_samples=1, batch_size=1, max_new_tokens=1),
        device_type="cpu",
    )
    official = requested_metadata(
        model_name="model",
        tokenizer_name="tokenizer",
        prompt="prompt",
        config=CurrentDirectConfig(
            num_samples=1,
            batch_size=1,
            max_new_tokens=1,
            ari_convention=ARI_CONVENTION_OFFICIAL_PUBLIC,
        ),
        device_type="cpu",
    )
    assert "ari_convention" not in current  # Legacy current cache stays reusable.
    assert official["ari_convention"] == ARI_CONVENTION_OFFICIAL_PUBLIC
    assert current["decoding_convention"] != official["decoding_convention"]


def test_scalar_ari_without_text_or_tokens_cannot_be_rescored() -> None:
    result = assess_data_reuse(
        {
            "artifact_type": "direct",
            "ari_convention": ARI_CONVENTION_CURRENT,
            "has_raw_ari": True,
        },
        target_ari_convention=ARI_CONVENTION_CURRENT,
    )
    assert not result["reusable_for_rescoring_phi"]


def test_old_current_smc_is_not_reusable_for_official_public_target() -> None:
    result = assess_data_reuse(
        {
            "artifact_type": "smc",
            "ari_convention": ARI_CONVENTION_CURRENT,
            "has_phi": True,
            "has_final_smc_weights": True,
            "has_log_z": True,
        },
        target_ari_convention=ARI_CONVENTION_OFFICIAL_PUBLIC,
    )
    assert not result["ari_target_match"]
    assert not result["reusable_for_probability_estimation"]


def test_tps_without_chain_order_is_not_reusable_for_temporal_ess() -> None:
    result = assess_data_reuse(
        {
            "artifact_type": "tps",
            "ari_convention": ARI_CONVENTION_CURRENT,
            "has_phi": True,
            "has_lambda_schedule": True,
            "has_level_labels": True,
            "has_chain_order": False,
        },
        target_ari_convention=ARI_CONVENTION_CURRENT,
    )
    assert not result["reusable_for_temporal_ess"]


def test_tps_phi_lambda_and_level_labels_are_reusable_for_mbar() -> None:
    result = assess_data_reuse(
        {
            "artifact_type": "tps",
            "ari_convention": ARI_CONVENTION_CURRENT,
            "has_phi": True,
            "has_lambda_schedule": True,
            "has_level_labels": True,
            "has_chain_order": True,
        },
        target_ari_convention=ARI_CONVENTION_CURRENT,
    )
    assert result["reusable_for_mbar"]
    assert result["reusable_for_temporal_ess"]


def test_runtime_without_hardware_or_configuration_is_not_comparable() -> None:
    incomplete = assess_data_reuse(
        {
            "artifact_type": "tps",
            "has_runtime": True,
            "has_hardware_metadata": False,
            "has_configuration_metadata": True,
        }
    )
    complete = assess_data_reuse(
        {
            "artifact_type": "tps",
            "has_runtime": True,
            "has_hardware_metadata": True,
            "has_configuration_metadata": True,
        }
    )
    assert not incomplete["reusable_for_runtime"]
    assert complete["reusable_for_runtime"]


def test_drive_audit_detects_adjacent_sample_tokens_without_writing(tmp_path: Path) -> None:
    run_root = tmp_path / "smc" / "seed_1"
    run_root.mkdir(parents=True)
    summary = {
        "algorithm": "suffix_direct_mh_smc",
        "mean_phi": 6.0,
        "log_normalizing_constant_estimate": 1.2,
        "config": {
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "prompt": PROMPT,
            "max_new_tokens": 100,
            "ari_convention": ARI_CONVENTION_CURRENT,
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sample = {
        "completion": " example",
        "completion_token_ids": [1, 2],
        "normalized_weight": 1.0,
    }
    (run_root / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    files_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    rows = audit_run_directory(tmp_path)

    files_after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert files_after == files_before
    assert len(rows) == 1
    assert rows[0]["reusable_for_rescoring_phi"]
    assert rows[0]["has_token_ids"]
