from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rea_llms.observables import create_observable
from rea_llms.observables.qwen_guard_gen import QwenGuardGenObservable


class FakeBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeGenTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    _labels = {
        " Safe": [10],
        " Unsafe": [11],
        " Controversial": [12, 13],
    }

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return (
            f"USER: {messages[0]['content']}\n"
            f"ASSISTANT: {messages[1]['content']}\nASSESSMENT:\n"
        )

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._labels[text]

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ):
        rows = [[2 + ord(character) % 7 for character in text] for text in texts]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        input_ids = torch.tensor(padded, dtype=torch.long)
        return FakeBatch(
            input_ids=input_ids,
            attention_mask=input_ids.ne(0).long(),
        )

    @staticmethod
    def batch_decode(rows, skip_special_tokens=True):
        return [
            "Safety: Unsafe\nCategories: Violent\nRefusal: No"
            for _ in rows
        ]


class FakeGenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 20), -20.0, device=input_ids.device)
        logits[:, -1, 10] = 0.0
        logits[:, -1, 11] = 2.0
        logits[:, -1, 12] = 1.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        suffix = torch.tensor([[7, 8]], device=input_ids.device).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, suffix], dim=1)


def fake_observable(**kwargs) -> QwenGuardGenObservable:
    observable = QwenGuardGenObservable(
        device="cpu",
        dtype="float32",
        batch_size=8,
        **kwargs,
    )
    observable._model = FakeGenModel()
    observable._tokenizer = FakeGenTokenizer()
    return observable


def test_factory_constructs_lazy_qwen_guard_gen() -> None:
    observable = create_observable(
        "qwen_guard_gen",
        device="cpu",
        dtype="float32",
    )
    assert isinstance(observable, QwenGuardGenObservable)
    assert observable._model is None
    assert observable._tokenizer is None


def test_log_odds_use_all_three_severity_labels() -> None:
    observable = fake_observable(log_odds_clip=None)
    scores = observable.score_batch("prompt", ["first", "second"])

    expected = 2.0 - torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0).item()
    assert observable.model.forward_calls == 1
    for score in scores:
        assert score.value == pytest.approx(expected)
        assert score.metadata["unsafe_log_odds_raw"] == pytest.approx(expected)
        assert score.metadata["predicted_class"] == "Unsafe"
        assert (
            score.metadata["safe_probability"]
            + score.metadata["unsafe_probability"]
            + score.metadata["controversial_probability"]
        ) == pytest.approx(1.0)


def test_log_odds_are_clipped_and_duplicate_scores_are_cached() -> None:
    observable = fake_observable(log_odds_clip=0.25)
    scores = observable.score_batch("prompt", ["same", "same"])
    assert scores[0].value == pytest.approx(0.25)
    assert scores[0] == scores[1]
    assert observable.evaluated_completions == 1

    observable.score_completion("prompt", "same")
    assert observable.model.forward_calls == 1
    assert observable.cache_hits == 1


def test_structured_terminal_classification_parses_category_and_refusal() -> None:
    observable = fake_observable()
    rows = observable.classify_batch("prompt", ["completion"])
    assert rows == [{
        "severity": "Unsafe",
        "categories": ["Violent"],
        "refusal": "No",
        "raw_assessment": "Safety: Unsafe\nCategories: Violent\nRefusal: No",
        "parse_ok": True,
    }]


def test_response_template_contains_both_conversation_turns() -> None:
    observable = fake_observable()
    text = observable._moderation_text("fixed query", "full response")
    assert "USER: fixed query" in text
    assert "ASSISTANT: full response" in text


def test_violent_category_notebook_is_complete_and_readable_in_place() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "qwen_safety_smc_experiment"
        / "notebooks"
        / "procedural_leakage_direct_experiment.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = []
    markdown_cells = []
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("outputs", []) == []
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            ast.parse(source, filename=f"{path.name}:cell-{index}")
            code_cells.append(source)
        else:
            markdown_cells.append(source)

    joined = "\n".join(code_cells)
    headings = "\n".join(markdown_cells)
    assert notebook["nbformat"] == 4
    assert joined.count("# EDIT PARAMETERS HERE") == 1
    assert "Qwen/Qwen3Guard-Gen-0.6B" in joined
    assert "GUARD_PHI_DEFINITION = 'unsafe_log_odds'" in joined
    assert "GUARD_LOG_ODDS_CLIP = 20.0" in joined
    assert "TARGET_CATEGORY = 'Violent'" in joined
    assert "RUN_PROMPT_PILOT = True" in joined
    assert "RUN_DIRECT = True" in joined
    assert "RUN_SMC = True" in joined
    assert joined.count("run_direct_in_memory(") == 2
    assert joined.count("run_suffix_direct_mh_smc(") == 1
    assert "PROMPT_VARIANTS" in joined
    for prompt_variant in (
        "direct",
        "instruction_override",
        "fictional_fable",
        "syllable_reconstruction",
        "affirmative_document",
        "quoted_injection",
    ):
        assert f"'{prompt_variant}'" in joined
    assert "selection_used_confirmatory_samples': False" in joined
    assert "PILOT_SEED = SEED + 500" in joined
    assert "DIRECT_SEED = SEED + 1000" in joined
    assert "SMC_SEED = SEED + 2000" in joined
    assert "estimate_base_expectation(" in joined
    assert "attach_terminal_assessments(" in joined
    assert "clip_floor_fraction" in joined
    assert "Observable saturation" in joined
    assert "display_completion_catalog(" in joined
    assert "html_lib.escape(str(row.full_completion))" in joined
    assert "off_category_samples_discarded': 0" in joined
    assert "completion_text_truncated_in_catalog': False" in joined
    assert "GITHUB_TOKEN" in joined
    assert "drive.mount" not in joined
    assert ".to_csv(" not in joined
    assert "files.download(" not in joined
    assert "procedural leakage" not in headings.lower()
    for section in range(1, 12):
        assert f"## {section}." in headings
