from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from rea_llms.observables import create_observable
from rea_llms.observables.base import ObservableScore
from rea_llms.observables.qwen_guard import QwenGuardHarmfulnessObservable
from qwen_safety_smc_experiment.in_memory import (
    PrefixedCompletionObservable,
    SafetyPromptContext,
    build_qwen_fixed_context,
    evaluate_paired_prefix_robustness,
    run_direct_in_memory,
)
from suffix_direct_mh_smc_experiment.smc_sampler import run_suffix_direct_mh_smc


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: value.to(device) for key, value in self.items()})

    __getattr__ = dict.__getitem__


class FakeGuardTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        text = "".join(
            f"<{row['role']}>{row['content']}<eos>" for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return self._encode(text) if tokenize else text

    @staticmethod
    def _encode(text):
        return [3 + (ord(character) % 13) for character in text]

    def __call__(self, texts, *, return_tensors, padding, add_special_tokens):
        rows = [self._encode(text) for text in texts]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return FakeBatch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(97 + int(token) % 26) for token in token_ids)


class FakeGuardModel(torch.nn.Module):
    response_risk_level_map = {0: "Safe", 1: "Unsafe", 2: "Controversial"}

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            response_risk_level_map={"0": "Safe", "1": "Unsafe", "2": "Controversial"}
        )
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 3), device=input_ids.device)
        logits[..., 0] = 0.5
        logits[..., 1] = (input_ids.float() % 5) / 2
        logits[..., 2] = -0.25
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(risk_level_logits=logits)


def fake_observable(phi_definition="unsafe_probability"):
    observable = QwenGuardHarmfulnessObservable(
        device="cpu",
        dtype="float32",
        phi_definition=phi_definition,
        batch_size=8,
    )
    observable._model = FakeGuardModel()
    observable._tokenizer = FakeGuardTokenizer()
    return observable


def test_qwen_guard_factory_is_lazy() -> None:
    observable = create_observable(
        "qwen_guard",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(observable, QwenGuardHarmfulnessObservable)
    assert observable._model is None
    assert observable._tokenizer is None


def test_qwen_guard_uses_full_softmax_unsafe_probability_and_batches() -> None:
    observable = fake_observable()
    scores = observable.score_batch("prompt", ["alpha", "beta"])
    assert len(scores) == 2
    assert observable.model.forward_calls == 1
    for score in scores:
        probabilities = [
            score.metadata["safe_probability"],
            score.metadata["unsafe_probability"],
            score.metadata["controversial_probability"],
        ]
        assert sum(probabilities) == pytest.approx(1.0)
        assert score.value == pytest.approx(score.metadata["unsafe_probability"])
        assert 0.0 <= score.value <= 1.0
    assert observable.evaluation_batches == 1
    assert observable.evaluated_completions == 2


def test_qwen_guard_coalesces_duplicate_uncached_completions() -> None:
    observable = fake_observable()
    scores = observable.score_batch("prompt", ["same", "same", "different"])
    assert len(scores) == 3
    assert scores[0] == scores[1]
    assert observable.model.forward_calls == 1
    assert observable.evaluated_completions == 2


def test_weighted_phi_and_exact_cache_are_explicit() -> None:
    observable = fake_observable("unsafe_plus_half_controversial")
    first = observable.score_completion("prompt", "completion")
    calls = observable.model.forward_calls
    second = observable.score_completion("prompt", "completion")
    assert first == second
    assert observable.model.forward_calls == calls
    assert observable.cache_hits == 1
    assert first.value == pytest.approx(
        first.metadata["unsafe_probability"]
        + 0.5 * first.metadata["controversial_probability"]
    )


def test_prefix_trajectory_returns_finite_three_class_probabilities() -> None:
    observable = fake_observable()
    rows = observable.prefix_trajectory("prompt", "safe abstract response")
    assert rows
    for row in rows:
        assert row["Safe"] + row["Unsafe"] + row["Controversial"] == pytest.approx(1.0)
        assert row["phi"] == pytest.approx(row["Unsafe"])


def test_fixed_assistant_prefix_is_frozen_in_context_and_observable() -> None:
    base = fake_observable()
    wrapped = PrefixedCompletionObservable(base, completion_prefix="I understand. ")
    context = build_qwen_fixed_context(
        base.tokenizer,
        system_prompt="Follow safety policy.",
        user_prompt="Discuss an abstract risky request safely.",
        assistant_prefix="I understand. ",
        enable_thinking=False,
    )
    assert context.generator_prompt.endswith("I understand. ")
    assert context.full_completion("I will refuse.") == "I understand. I will refuse."
    score = wrapped.score_completion(context.user_prompt, "I will refuse.")
    assert (context.user_prompt, context.full_completion("I will refuse.")) in base._cache
    assert math.isfinite(score.value)


def test_direct_helper_is_reproducible_in_memory_and_preserves_prefix(tmp_path, monkeypatch) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "tests" / "test_suffix_direct_mh_smc.py")
    )
    monkeypatch.chdir(tmp_path)
    context = SafetyPromptContext("", "Once", "Fixed: ", False, "OnceFixed: ", "OnceFixed: ")

    class SumObservable(namespace["CompletionTokenSumObservable"]):
        completion_prefix = "Fixed: "

    kwargs = dict(
        model=namespace["PrefixDependentTinyModel"](),
        tokenizer=namespace["TinyTokenizer"](),
        observable=SumObservable(),
        context=context,
        num_samples=4,
        max_new_tokens=2,
        batch_size=2,
        seed=53,
        event_threshold=2.0,
    )
    first = run_direct_in_memory(**kwargs)
    second = run_direct_in_memory(**{**kwargs, "model": namespace["PrefixDependentTinyModel"]()})
    assert first.rows == second.rows
    assert all(row["completion"].startswith("Fixed: ") for row in first.rows)
    assert first.summary["persisted"] is False
    assert list(tmp_path.iterdir()) == []


def test_paired_prefix_robustness_reuses_fixed_tokens_without_generation() -> None:
    class TextTokenizer:
        @staticmethod
        def batch_decode(rows, skip_special_tokens=True):
            return ["".join(str(token) for token in row) for row in rows]

    class PrefixSensitiveObservable:
        def __init__(self):
            self.requests = []

        def score_batch(self, prompt, completions):
            self.requests.append((prompt, list(completions)))
            scores = []
            for text in completions:
                digit_count = sum(character.isdigit() for character in text)
                base = {1: 0.6, 2: 0.4, 4: 0.7}[digit_count]
                value = base if text.startswith("Yes ") else max(0.0, base - 0.5)
                scores.append(ObservableScore(value=value, metadata={}))
            return scores

    token_rows = [[1, 2, 3, 4], [4, 5, 6, 7]]
    original_rows = [row.copy() for row in token_rows]
    observable = PrefixSensitiveObservable()
    result = evaluate_paired_prefix_robustness(
        TextTokenizer(),
        observable,
        prompt="fixed prompt",
        token_sequences=token_rows,
        affirmative_prefix="Yes ",
        event_threshold=0.5,
        trajectory_checkpoints=(1, 2, 4),
        trajectory_sample_limit=1,
        existing_affirmative_terminal_phi=(0.8, 0.7),
    )

    assert token_rows == original_rows
    assert result.evaluated_sample_count == 2
    assert result.trajectory_sample_count == 1
    assert result.trajectory_checkpoints == (1, 2, 4)
    assert len(result.terminal_rows) == 2
    assert len(result.trajectory_rows) == 6
    assert len(result.path_rows) == 2
    assert result.terminal_rows[0]["affirmative_prefix_phi"] == pytest.approx(0.8)
    assert result.terminal_rows[0]["no_prefix_phi"] == pytest.approx(0.2)
    assert result.terminal_rows[0]["paired_delta_phi"] == pytest.approx(0.6)
    affirmative_path = next(
        row for row in result.path_rows if row["condition"] == "affirmative_prefix"
    )
    assert affirmative_path["first_observed_crossing_checkpoint"] == 1
    assert affirmative_path["forward_crossings"] == 1
    assert affirmative_path["reverse_crossings"] == 1
    assert affirmative_path["has_reverse_crossing"] is True
    assert affirmative_path["terminal_malicious"] is True
    assert affirmative_path["normalized_auc_phi"] == pytest.approx(0.5333333333)
    assert len(observable.requests) == 2
    assert all(prompt == "fixed prompt" for prompt, _ in observable.requests)


def test_paired_prefix_robustness_validates_checkpoint_support() -> None:
    class Tokenizer:
        @staticmethod
        def batch_decode(rows, skip_special_tokens=True):
            return [str(row) for row in rows]

    class Observable:
        @staticmethod
        def score_batch(prompt, completions):
            return [ObservableScore(value=0.0, metadata={}) for _ in completions]

    with pytest.raises(ValueError, match="fit every selected sequence"):
        evaluate_paired_prefix_robustness(
            Tokenizer(),
            Observable(),
            prompt="prompt",
            token_sequences=[[1, 2]],
            affirmative_prefix="Yes ",
            event_threshold=0.5,
            trajectory_checkpoints=(1, 3),
            trajectory_sample_limit=1,
        )


def test_current_and_proposal_phi_order_favors_higher_phi_for_negative_lambda() -> None:
    from suffix_direct_mh_smc_experiment.smc_sampler import metropolis_log_acceptance

    current = torch.tensor([0.1, 0.9])
    proposal = torch.tensor([0.9, 0.1])
    log_acceptance = metropolis_log_acceptance(
        current,
        proposal,
        target_lambda=-1.0,
    )
    assert float(log_acceptance[0]) == pytest.approx(0.0)
    assert float(log_acceptance[1]) == pytest.approx(-0.8)


def test_safety_experiment_notebook_is_valid_in_memory_and_uses_shared_apis() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "safety_smc_experiment.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = []
    markdown = []
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("outputs", []) == []
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            ast.parse(source, filename=f"{path.name}:cell-{index}")
            code_cells.append(source)
        else:
            markdown.append(source)
    joined = "\n".join(code_cells)
    headings = "\n".join(markdown)
    assert joined.count("# EDIT PARAMETERS HERE") == 1
    for name in (
        "GENERATOR_MODEL",
        "GUARD_MODEL",
        "NUM_PARTICLES",
        "MCMC_STEPS_PER_LEVEL",
        "LAMBDAS",
        "SUFFIX_SIZES",
        "SUFFIX_RHOS",
        "EVENT_THRESHOLD",
        "FAST_MODE",
        "SHOW_FULL_COMPLETIONS",
    ):
        assert name in joined
    assert "run_direct_in_memory(" in joined
    assert joined.count("run_direct_in_memory(") == 2
    assert "run_suffix_direct_mh_smc(" in joined
    assert "evaluate_paired_prefix_robustness(" in joined
    assert "prefix_robustness_results" in joined
    assert "first_observed_crossing_checkpoint" in joined
    assert "recovery_probability_given_ever_malicious" in joined
    assert "completion_catalog_df" in joined
    assert "selected_completion_samples_df" in joined
    assert "'lm_generated_token_transitions': 0" in joined
    assert "scenario_specs" not in joined
    assert "SCENARIO_SAMPLES" not in joined
    assert "estimate_base_tails_from_final_samples(" in joined
    assert "capture_level_particles=True" in joined
    assert "profile_runtime=True" in joined
    assert "batch_single_atol = 2e-3" in joined
    assert "batch_single_probability_atol" in joined
    assert "batched_equals_individual" not in joined
    assert "math.isclose(batch_score.value, single_score.value" not in joined
    assert "drive.mount" not in joined
    for forbidden in ("to_csv(", "to_json(", "savefig(", "write_text("):
        assert forbidden not in joined
    for section in range(1, 13):
        assert f"## {section}." in headings


def test_ari_seeded_snapshot_is_unchanged() -> None:
    namespace = runpy.run_path("tests/test_suffix_direct_mh_smc.py")
    result = run_suffix_direct_mh_smc(
        namespace["PrefixDependentTinyModel"](),
        namespace["TinyTokenizer"](),
        namespace["CompletionTokenSumObservable"](),
        prompt="Once",
        config=namespace["_config"](
            num_particles=6,
            suffix_mode="stochastic",
            suffix_sizes=(1, 3),
            suffix_size_schedule=None,
            suffix_rhos=(0.5, 0.5),
            batch_size=3,
            seed=37,
        ),
    )
    assert [row["completion_token_ids"] for row in result.samples] == [
        [2, 3, 0],
        [1, 1, 2],
        [1, 1, 2],
        [1, 2, 0],
        [0, 2, 0],
        [2, 0, 0],
    ]
    assert result.summary["unweighted_mean_ari"] == pytest.approx(3.3333333333333335)
    assert result.summary["weighted_mean_ari"] == pytest.approx(3.9139799191376166)
