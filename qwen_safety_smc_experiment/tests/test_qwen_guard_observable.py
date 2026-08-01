from __future__ import annotations

import runpy
from types import SimpleNamespace

import pytest
import torch

from rea_llms.observables import create_observable
from rea_llms.observables.qwen_guard import QwenGuardHarmfulnessObservable
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
