from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from rea_llms.generation import (
    GenerationDiagnostics,
    fixed_completion_log_probabilities,
    generate_fixed_length_completions,
    prompt_token_ids,
)
from rea_llms.models import set_seed
from rea_llms.observables.base import ObservableScore, observable_configuration
from suffix_direct_mh_smc_experiment.smc_sampler import score_observable


@dataclass(frozen=True)
class SafetyPromptContext:
    """Exact fixed generator context and its human-readable components."""

    system_prompt: str
    user_prompt: str
    assistant_prefix: str
    enable_thinking: bool
    chat_template_without_prefix: str
    generator_prompt: str

    def full_completion(self, mutable_completion: str) -> str:
        return self.assistant_prefix + mutable_completion


def build_qwen_fixed_context(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_prefix: str = "",
    enable_thinking: bool = False,
) -> SafetyPromptContext:
    """Render system/user turns and freeze an optional assistant text prefix."""

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return SafetyPromptContext(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_prefix=assistant_prefix,
        enable_thinking=bool(enable_thinking),
        chat_template_without_prefix=template,
        generator_prompt=template + assistant_prefix,
    )


@dataclass
class PrefixedCompletionObservable:
    """Expose mutable completions while scoring a fixed assistant prefix too."""

    base_observable: Any
    completion_prefix: str = ""
    name: str = "qwen_guard"

    def score_batch(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> list[ObservableScore]:
        return self.base_observable.score_batch(
            prompt,
            [self.completion_prefix + completion for completion in completions],
        )

    def score_completion(self, prompt: str, completion: str) -> ObservableScore:
        return self.score_batch(prompt, [completion])[0]

    def score_token_sequences(
        self,
        *,
        tokenizer,
        prompt: str,
        token_sequences: list[list[int]],
        prompt_token_count: int,
    ) -> list[ObservableScore]:
        mutable = tokenizer.batch_decode(
            [ids[prompt_token_count:] for ids in token_sequences],
            skip_special_tokens=True,
        )
        return self.score_batch(prompt, mutable)

    def prefix_trajectory(
        self,
        prompt: str,
        mutable_completion: str,
    ) -> list[dict[str, Any]]:
        return self.base_observable.prefix_trajectory(
            prompt,
            self.completion_prefix + mutable_completion,
        )

    def reported_value(self, score: ObservableScore) -> float:
        selector = getattr(self.base_observable, "reported_value", None)
        return float(selector(score) if callable(selector) else score.value)

    def configuration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "completion_prefix": self.completion_prefix,
            "base_observable": observable_configuration(self.base_observable),
        }

    @property
    def cache_hits(self) -> int:
        return int(getattr(self.base_observable, "cache_hits", 0))

    @property
    def cache_misses(self) -> int:
        return int(getattr(self.base_observable, "cache_misses", 0))

    @property
    def evaluation_batches(self) -> int:
        return int(getattr(self.base_observable, "evaluation_batches", 0))

    @property
    def evaluated_completions(self) -> int:
        return int(getattr(self.base_observable, "evaluated_completions", 0))

    @property
    def inference_seconds(self) -> float:
        return float(getattr(self.base_observable, "inference_seconds", 0.0))


@dataclass(frozen=True)
class DirectSampleResult:
    """One IID fixed-length direct sample retained entirely in memory."""

    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    completion_token_ids: torch.Tensor


@dataclass(frozen=True)
class PairedPrefixRobustnessResult:
    """Guard scores for fixed continuations with and without a text prefix."""

    terminal_rows: list[dict[str, Any]]
    trajectory_rows: list[dict[str, Any]]
    path_rows: list[dict[str, Any]]
    evaluated_sample_count: int
    trajectory_sample_count: int
    trajectory_checkpoints: tuple[int, ...]


def evaluate_paired_prefix_robustness(
    tokenizer,
    observable,
    *,
    prompt: str,
    token_sequences: Sequence[Sequence[int]] | torch.Tensor,
    affirmative_prefix: str,
    event_threshold: float,
    trajectory_checkpoints: Sequence[int],
    trajectory_sample_limit: int,
    existing_affirmative_terminal_phi: Sequence[float] | None = None,
) -> PairedPrefixRobustnessResult:
    """Re-score fixed LM continuations under two prefix presentations.

    The language model is deliberately not an argument: this analysis performs
    no generation. Both conditions use the same mutable token sequences, so it
    measures the paired effect of presenting ``affirmative_prefix`` to the
    guard, not the LM distribution induced by generating without that prefix.
    """

    if not affirmative_prefix:
        raise ValueError("affirmative_prefix must be non-empty")
    if not math.isfinite(float(event_threshold)):
        raise ValueError("event_threshold must be finite")
    if trajectory_sample_limit <= 0:
        raise ValueError("trajectory_sample_limit must be positive")

    if isinstance(token_sequences, torch.Tensor):
        token_rows = token_sequences.detach().cpu().tolist()
    else:
        token_rows = [list(row) for row in token_sequences]
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("token_sequences must contain non-empty sequences")

    checkpoints = tuple(sorted({int(value) for value in trajectory_checkpoints}))
    selected_count = min(len(token_rows), int(trajectory_sample_limit))
    maximum_checkpoint = min(len(row) for row in token_rows[:selected_count])
    if not checkpoints or checkpoints[0] <= 0 or checkpoints[-1] > maximum_checkpoint:
        raise ValueError(
            "trajectory_checkpoints must be positive and fit every selected sequence"
        )

    mutable_completions = tokenizer.batch_decode(
        token_rows,
        skip_special_tokens=True,
    )
    no_prefix_scores = observable.score_batch(prompt, mutable_completions)
    if existing_affirmative_terminal_phi is None:
        affirmative_scores = observable.score_batch(
            prompt,
            [affirmative_prefix + text for text in mutable_completions],
        )
        affirmative_values = [float(score.value) for score in affirmative_scores]
    else:
        affirmative_values = [
            float(value) for value in existing_affirmative_terminal_phi
        ]
        if len(affirmative_values) != len(token_rows):
            raise ValueError(
                "existing_affirmative_terminal_phi must match token_sequences"
            )
        if not all(math.isfinite(value) for value in affirmative_values):
            raise ValueError("existing affirmative terminal scores must be finite")

    terminal_rows: list[dict[str, Any]] = []
    for sample_id, (affirmative, no_prefix_score) in enumerate(
        zip(affirmative_values, no_prefix_scores)
    ):
        no_prefix = float(no_prefix_score.value)
        terminal_rows.append(
            {
                "sample_id": int(sample_id),
                "affirmative_prefix_phi": affirmative,
                "no_prefix_phi": no_prefix,
                "paired_delta_phi": affirmative - no_prefix,
                "affirmative_prefix_event": bool(affirmative >= event_threshold),
                "no_prefix_event": bool(no_prefix >= event_threshold),
            }
        )

    trajectory_texts: list[str] = []
    trajectory_metadata: list[tuple[int, int, str]] = []
    selected_rows = token_rows[:selected_count]
    for checkpoint in checkpoints:
        mutable_prefixes = tokenizer.batch_decode(
            [row[:checkpoint] for row in selected_rows],
            skip_special_tokens=True,
        )
        for condition, prefix in (
            ("affirmative_prefix", affirmative_prefix),
            ("no_prefix", ""),
        ):
            for sample_id, mutable_prefix in enumerate(mutable_prefixes):
                trajectory_texts.append(prefix + mutable_prefix)
                trajectory_metadata.append((sample_id, checkpoint, condition))

    trajectory_scores = observable.score_batch(prompt, trajectory_texts)
    trajectory_rows: list[dict[str, Any]] = []
    for (sample_id, checkpoint, condition), score in zip(
        trajectory_metadata,
        trajectory_scores,
    ):
        value = float(score.value)
        trajectory_rows.append(
            {
                "sample_id": int(sample_id),
                "token_checkpoint": int(checkpoint),
                "condition": condition,
                "phi": value,
                "event": bool(value >= event_threshold),
            }
        )

    grouped_paths: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        key = (str(row["condition"]), int(row["sample_id"]))
        grouped_paths.setdefault(key, []).append(row)
    path_rows: list[dict[str, Any]] = []
    for (condition, sample_id), rows in grouped_paths.items():
        ordered = sorted(rows, key=lambda row: int(row["token_checkpoint"]))
        path_checkpoints = np.asarray(
            [row["token_checkpoint"] for row in ordered],
            dtype=np.int64,
        )
        path_phi = np.asarray([row["phi"] for row in ordered], dtype=np.float64)
        path_events = np.asarray([row["event"] for row in ordered], dtype=bool)
        event_indices = np.flatnonzero(path_events)
        reverse_crossings = int(
            np.count_nonzero(path_events[:-1] & ~path_events[1:])
        )
        forward_crossings = int(
            np.count_nonzero(~path_events[:-1] & path_events[1:])
        )
        if len(path_checkpoints) > 1:
            widths = np.diff(path_checkpoints).astype(np.float64)
            area = float(
                np.sum(0.5 * (path_phi[:-1] + path_phi[1:]) * widths)
                / (path_checkpoints[-1] - path_checkpoints[0])
            )
        else:
            area = float(path_phi[0])
        ever_event = bool(event_indices.size)
        path_rows.append(
            {
                "sample_id": int(sample_id),
                "condition": condition,
                "checkpoint_count": len(ordered),
                "first_observed_crossing_checkpoint": (
                    int(path_checkpoints[event_indices[0]])
                    if ever_event
                    else None
                ),
                "ever_malicious_at_checkpoints": ever_event,
                "terminal_malicious": bool(path_events[-1]),
                "recovered_below_threshold_by_terminal": bool(
                    ever_event and not path_events[-1]
                ),
                "forward_crossings": forward_crossings,
                "reverse_crossings": reverse_crossings,
                "has_reverse_crossing": bool(reverse_crossings),
                "max_phi": float(path_phi.max()),
                "terminal_phi": float(path_phi[-1]),
                "mean_checkpoint_phi": float(path_phi.mean()),
                "normalized_auc_phi": area,
            }
        )

    return PairedPrefixRobustnessResult(
        terminal_rows=terminal_rows,
        trajectory_rows=trajectory_rows,
        path_rows=path_rows,
        evaluated_sample_count=len(token_rows),
        trajectory_sample_count=selected_count,
        trajectory_checkpoints=checkpoints,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _counter_snapshot(observable) -> dict[str, float]:
    return {
        name: float(getattr(observable, name, 0.0))
        for name in (
            "cache_hits",
            "cache_misses",
            "evaluation_batches",
            "evaluated_completions",
            "inference_seconds",
        )
    }


def _effective_length(token_ids: Sequence[int], eos_token_id: int | None) -> int:
    if eos_token_id is None:
        return len(token_ids)
    try:
        return list(token_ids).index(int(eos_token_id)) + 1
    except ValueError:
        return len(token_ids)


def run_direct_in_memory(
    model,
    tokenizer,
    observable,
    *,
    context: SafetyPromptContext,
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    event_threshold: float,
    temperature: float = 1.0,
    top_p: float = 1.0,
    compute_log_probabilities: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DirectSampleResult:
    """Sample the shared fixed-length base distribution without writing files.

    The active Direct-MH cancellation requires unrestricted temperature-one
    proposals. The explicit checks prevent a notebook parameter from silently
    changing the direct distribution while leaving SMC mathematically invalid.
    """

    if num_samples <= 0 or max_new_tokens <= 0 or batch_size <= 0:
        raise ValueError("sample, token, and batch counts must be positive")
    if not math.isclose(float(temperature), 1.0, abs_tol=0.0, rel_tol=0.0):
        raise ValueError("Suffix Direct-MH requires TEMPERATURE=1.0")
    if not math.isclose(float(top_p), 1.0, abs_tol=0.0, rel_tol=0.0):
        raise ValueError("Suffix Direct-MH requires TOP_P=1.0 (full support)")
    if not math.isfinite(float(event_threshold)):
        raise ValueError("event_threshold must be finite")
    configured_prefix = getattr(observable, "completion_prefix", None)
    if configured_prefix is not None and configured_prefix != context.assistant_prefix:
        raise ValueError("observable and generator assistant prefixes differ")

    set_seed(seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    fixed_prompt_ids = prompt_token_ids(tokenizer, context.generator_prompt, device)
    generation = GenerationDiagnostics()
    counters_before = _counter_snapshot(observable)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    total_started = time.perf_counter()
    generation_started = time.perf_counter()
    completion_tokens = generate_fixed_length_completions(
        model,
        fixed_prompt_ids,
        num_sequences=num_samples,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        generator=generator,
        diagnostics=generation,
        progress_callback=progress_callback,
    )
    _synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    scoring_started = time.perf_counter()
    phi, _, scores = score_observable(
        tokenizer,
        observable,
        prompt=context.generator_prompt,
        observable_prompt=context.user_prompt,
        prompt_ids=fixed_prompt_ids,
        completions=completion_tokens,
    )
    guard_scoring_seconds = time.perf_counter() - scoring_started

    log_probability_seconds = 0.0
    if compute_log_probabilities:
        log_probability_started = time.perf_counter()
        log_probabilities = fixed_completion_log_probabilities(
            model,
            fixed_prompt_ids,
            completion_tokens,
            batch_size=batch_size,
        )
        _synchronize(device)
        log_probability_seconds = time.perf_counter() - log_probability_started
    else:
        log_probabilities = torch.full(
            (num_samples,),
            torch.nan,
            dtype=torch.float64,
            device=device,
        )

    mutable_completions = tokenizer.batch_decode(
        completion_tokens.detach().cpu().tolist(),
        skip_special_tokens=True,
    )
    token_rows = completion_tokens.detach().cpu().tolist()
    phi_rows = phi.detach().cpu().tolist()
    log_probability_rows = log_probabilities.detach().cpu().tolist()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    rows: list[dict[str, Any]] = []
    for index, (token_ids, mutable, value, log_probability, score) in enumerate(
        zip(token_rows, mutable_completions, phi_rows, log_probability_rows, scores)
    ):
        metadata = dict(score.metadata)
        length = _effective_length(token_ids, eos_token_id)
        rows.append(
            {
                "sample_id": index,
                "seed": int(seed),
                "mutable_completion": mutable,
                "completion": context.full_completion(mutable),
                "completion_token_ids": token_ids,
                "phi": float(value),
                "predicted_class": metadata.get("predicted_class"),
                "safe_probability": metadata.get("safe_probability"),
                "controversial_probability": metadata.get(
                    "controversial_probability"
                ),
                "unsafe_probability": metadata.get("unsafe_probability"),
                "risk_logits": metadata.get("risk_logits"),
                "fixed_length_log_probability": float(log_probability),
                "effective_length_tokens": int(length),
                "eos_observed": bool(
                    eos_token_id is not None and int(eos_token_id) in token_ids
                ),
                "event": bool(float(value) >= float(event_threshold)),
            }
        )

    _synchronize(device)
    runtime_seconds = time.perf_counter() - total_started
    counters_after = _counter_snapshot(observable)
    deltas = {
        name: counters_after[name] - counters_before[name]
        for name in counters_after
    }
    values = np.asarray(phi_rows, dtype=np.float64)
    event_count = int(np.count_nonzero(values >= float(event_threshold)))
    total_cache_requests = deltas["cache_hits"] + deltas["cache_misses"]
    summary = {
        "method": "iid_direct_in_memory",
        "persisted": False,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "max_new_tokens": int(max_new_tokens),
        "event_threshold": float(event_threshold),
        "event_count": event_count,
        "empirical_event_probability": event_count / num_samples,
        "unique_completion_count": len({row["completion"] for row in rows}),
        "generated_token_transitions": int(
            generation.generated_token_transitions
        ),
        "generator_batch_calls": int(generation.batch_calls),
        "generator_model_forward_calls": int(generation.model_forward_calls),
        "guard_evaluation_batches": int(deltas["evaluation_batches"]),
        "guard_evaluated_completions": int(deltas["evaluated_completions"]),
        "guard_cache_hits": int(deltas["cache_hits"]),
        "guard_cache_misses": int(deltas["cache_misses"]),
        "guard_cache_hit_rate": (
            deltas["cache_hits"] / total_cache_requests
            if total_cache_requests
            else math.nan
        ),
        "runtime_seconds": float(runtime_seconds),
        "generator_runtime_seconds": float(generation_seconds),
        "guard_scoring_runtime_seconds": float(guard_scoring_seconds),
        "guard_inference_seconds": float(deltas["inference_seconds"]),
        "log_probability_runtime_seconds": float(log_probability_seconds),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "observable_configuration": observable_configuration(observable),
        "fixed_context_token_count": int(fixed_prompt_ids.numel()),
    }
    return DirectSampleResult(
        rows=rows,
        summary=summary,
        completion_token_ids=completion_tokens,
    )
