from __future__ import annotations

import ast
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from official_annealed_tps_experiment.annealed_tps import (
    OfficialAnnealedTPSConfig,
    generate_official_tps_proposal,
    official_choice_indices,
    official_tps_log_acceptance,
    run_official_annealed_tps,
    sample_official_cut_index,
    suffix_size_from_official_cut,
)
from rea_llms.benchmarking import (
    completed_run_action,
    configuration_fingerprint,
    temporal_ess_initial_positive,
)
from rea_llms.observables.ari import ARIObservable
from rea_llms.observables.base import ObservableScore
from smc_tps_comparison.run_comparison import parse_args


ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(f" {int(token)}" for token in token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        batch, length = input_ids.shape
        prefix_sum = torch.cumsum(input_ids.to(torch.float32), dim=1)
        logits = torch.zeros((batch, length, 3), dtype=torch.float32)
        logits[..., 0] = 0.2 * prefix_sum
        logits[..., 1] = -0.1 * prefix_sum
        logits[..., 2] = 0.3
        return type("Output", (), {"logits": logits})


class TokenSumObservable:
    mode = "raw"

    def score_token_sequences(self, *, tokenizer, prompt, token_sequences, prompt_token_count):
        return [
            ObservableScore(
                value=float(sum(ids[prompt_token_count:])),
                metadata={
                    "raw_ari": float(sum(ids[prompt_token_count:])),
                    "capped_ari": min(15.0, float(sum(ids[prompt_token_count:]))),
                },
            )
            for ids in token_sequences
        ]


def _run(seed: int = 11):
    return run_official_annealed_tps(
        TinyModel(),
        TinyTokenizer(),
        TokenSumObservable(),
        prompt="Once",
        config=OfficialAnnealedTPSConfig(
            max_new_tokens=3,
            lambdas=(0.0, -0.5),
            steps_per_bias=4,
            seed=seed,
            tail_thresholds=(2.0, 4.0),
        ),
    )


def test_raw_ari_mode_does_not_apply_cap(monkeypatch) -> None:
    monkeypatch.setattr("rea_llms.observables.ari.automated_readability_index", lambda text: 22.5)
    raw = ARIObservable(mode="raw", cap=15.0).score_text("text")
    capped = ARIObservable(mode="capped", cap=15.0).score_text("text")
    assert raw.value == 22.5
    assert raw.metadata["raw_ari"] == 22.5
    assert capped.value == 15.0


def test_official_tau_domain_matches_notebook_choice_indices() -> None:
    prompt_tokens = 16
    completion_tokens = 100
    choices = official_choice_indices(prompt_tokens, completion_tokens)
    assert choices == tuple(range(16, 116))
    assert suffix_size_from_official_cut(16, 16, 100) == 100
    assert suffix_size_from_official_cut(115, 16, 100) == 1
    assert 116 not in choices
    rng = random.Random(3)
    assert all(sample_official_cut_index(rng, 16, 100) in choices for _ in range(100))


def test_official_suffix_proposal_preserves_exact_cut_prefix() -> None:
    model = TinyModel()
    prompt_ids = torch.tensor([1], dtype=torch.long)
    current = torch.tensor([[0, 1, 2]], dtype=torch.long)
    proposal, suffix_size = generate_official_tps_proposal(
        model,
        prompt_ids,
        current,
        cut_index=2,
        batch_size=1,
        generator=torch.Generator().manual_seed(4),
    )
    assert suffix_size == 2
    assert torch.equal(proposal[:, :1], current[:, :1])


def test_tps_has_one_chain_without_weights_particles_or_resampling() -> None:
    result = _run()
    summary = result.summary
    serialized = json.dumps(summary)
    assert summary["num_chains"] == 1
    assert summary["resampling"] is False
    assert summary["importance_weights"] is False
    assert "normalized_weight" not in serialized
    assert "num_particles" not in serialized


def test_tps_records_exact_full_matrix_and_accounting() -> None:
    result = _run()
    expected = 2 * 4
    assert result.summary["completed_full_schedule"]
    assert result.summary["trajectory_values"] == expected
    assert all(values.shape == (expected,) for values in result.trajectory.values())
    assert result.summary["total_proposals"] == expected
    assert result.summary["total_regenerated_tokens"] == int(
        result.trajectory["suffix_size"].sum()
    )
    assert result.trajectory["cumulative_proposals"][-1] == expected
    final = result.trajectory["raw_ari"][-4:]
    assert result.summary["mean_raw_ari_final_target"] == pytest.approx(final.mean())
    assert result.summary["tail_probabilities_final_target"]["ari_ge_2"] == pytest.approx(
        np.mean(final >= 2.0)
    )


def test_tps_is_reproducible_at_fixed_seed() -> None:
    first = _run(seed=19)
    second = _run(seed=19)
    for key in ("lambda_index", "phi", "accepted", "suffix_size"):
        assert np.array_equal(first.trajectory[key], second.trajectory[key])


def test_tps_acceptance_and_enumerable_stationarity() -> None:
    states = list(itertools.product(range(2), repeat=2))
    target_lambda = -0.5

    def next_probability(prefix: tuple[int, ...], token: int) -> float:
        total = sum((1,) + prefix)
        logits = np.asarray([0.2 * total, -0.1 * total])
        probabilities = np.exp(logits - logits.max())
        return float(probabilities[token] / probabilities.sum())

    def sequence_probability(sequence: tuple[int, int]) -> float:
        return next_probability((), sequence[0]) * next_probability((sequence[0],), sequence[1])

    phi = np.asarray([sum(state) for state in states], dtype=np.float64)
    target = np.asarray([sequence_probability(state) for state in states]) * np.exp(
        -target_lambda * phi
    )
    target /= target.sum()
    transition = np.zeros((4, 4), dtype=np.float64)
    for source_index, source in enumerate(states):
        for target_index, target_state in enumerate(states):
            if source_index == target_index:
                continue
            for suffix_size, probability_size in ((1, 0.5), (2, 0.5)):
                prefix_size = 2 - suffix_size
                if source[:prefix_size] != target_state[:prefix_size]:
                    continue
                proposal_probability = (
                    next_probability(source[:1], target_state[1])
                    if suffix_size == 1
                    else sequence_probability(target_state)
                )
                log_acceptance = official_tps_log_acceptance(
                    torch.tensor([phi[source_index]]),
                    torch.tensor([phi[target_index]]),
                    target_lambda=target_lambda,
                )
                transition[source_index, target_index] += (
                    probability_size * proposal_probability * float(torch.exp(log_acceptance))
                )
        transition[source_index, source_index] = 1.0 - transition[source_index].sum()
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(target @ transition, target, atol=1e-7)


def test_temporal_ess_on_synthetic_chains() -> None:
    alternating = np.tile([1.0, -1.0], 50)
    persistent = np.repeat(np.arange(10, dtype=float), 10)
    assert temporal_ess_initial_positive(np.ones(20)) == 20.0
    assert temporal_ess_initial_positive(alternating) == 100.0
    assert 1.0 <= temporal_ess_initial_positive(persistent) < 20.0


def test_fingerprint_rejects_incompatible_completed_run(tmp_path: Path) -> None:
    first = configuration_fingerprint({"algorithm": "tps", "seed": 1, "ari_mode": "raw"})
    second = configuration_fingerprint({"algorithm": "tps", "seed": 1, "ari_mode": "capped"})
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"configuration_fingerprint": first}), encoding="utf-8"
    )
    assert completed_run_action(output, first, skip_completed=True, overwrite=False) == "skip"
    with pytest.raises(FileExistsError, match="Incompatible"):
        completed_run_action(output, second, skip_completed=True, overwrite=False)


def test_comparison_algorithms_are_selectable_independently() -> None:
    smc = parse_args(["--algorithms", "suffix_direct_mh_smc"])
    tps = parse_args(["--algorithms", "official_annealed_tps"])
    assert smc.algorithms == "suffix_direct_mh_smc"
    assert tps.algorithms == "official_annealed_tps"


def test_comparison_notebook_is_valid_and_calls_shared_runner() -> None:
    path = ROOT / "notebooks" / "REA_SMC_vs_Official_TPS_Comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{path.name}:cell-{index}")
        code.append(source)
    joined = "\n".join(code)
    assert "EDIT PARAMETERS HERE" in joined
    assert "suffix_direct_mh_smc" in joined
    assert "official_annealed_tps" in joined
    assert "smc_tps_comparison/run_comparison.py" in joined
