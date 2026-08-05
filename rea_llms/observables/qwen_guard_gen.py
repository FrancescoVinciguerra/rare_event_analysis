from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from rea_llms.models import resolve_device, resolve_torch_dtype
from rea_llms.observables.base import ObservableScore


DEFAULT_QWEN_GUARD_GEN_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_QWEN_GUARD_GEN_REVISION = "fada3b2f655b89601929198343c94cd2f64d93cc"
QWEN_GUARD_GEN_PHI_DEFINITIONS = (
    "unsafe_probability",
    "unsafe_log_odds",
)
QWEN_GUARD_GEN_SEVERITIES = ("Safe", "Unsafe", "Controversial")


@dataclass
class QwenGuardGenObservable:
    """Terminal Qwen3Guard-Gen severity score for complete responses.

    Qwen3Guard-Gen is a causal language model whose first structured field is
    ``Safety: <label>``. This adapter compares the first generated token of the
    three official labels under a constrained softmax. ``unsafe_log_odds`` is
    the Unsafe logit minus logsumexp(Safe, Controversial); it is optionally
    clipped before being exposed as the SMC potential.
    """

    model_name: str = DEFAULT_QWEN_GUARD_GEN_MODEL
    revision: str | None = DEFAULT_QWEN_GUARD_GEN_REVISION
    device: str = "auto"
    dtype: str = "float16"
    phi_definition: str = "unsafe_log_odds"
    log_odds_clip: float | None = 20.0
    batch_size: int = 16
    cache_enabled: bool = True
    local_files_only: bool = False
    name: str = "qwen_guard_gen"
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _cache: dict[tuple[str, str], ObservableScore] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)
    evaluation_batches: int = field(default=0, init=False)
    evaluated_completions: int = field(default=0, init=False)
    inference_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.phi_definition not in QWEN_GUARD_GEN_PHI_DEFINITIONS:
            raise ValueError(
                "phi_definition must be one of "
                f"{QWEN_GUARD_GEN_PHI_DEFINITIONS}"
            )
        if self.log_odds_clip is not None and (
            not math.isfinite(float(self.log_odds_clip))
            or float(self.log_odds_clip) <= 0
        ):
            raise ValueError("log_odds_clip must be finite and positive or None")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolve_device(self.device)
        resolve_torch_dtype(self.dtype, resolve_device(self.device))

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = resolve_device(self.device)
        selected_dtype = resolve_torch_dtype(self.dtype, selected_device)
        load_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=selected_dtype,
            **load_kwargs,
        )
        model.to(selected_device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _moderation_text(self, prompt: str, completion: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
        )

    def _severity_first_token_ids(self) -> dict[str, int]:
        token_ids = {
            label: self.tokenizer.encode(
                f" {label}",
                add_special_tokens=False,
            )[0]
            for label in QWEN_GUARD_GEN_SEVERITIES
        }
        if len(set(token_ids.values())) != len(token_ids):
            raise RuntimeError(
                f"Qwen3Guard severity labels do not have distinct first tokens: {token_ids}"
            )
        return token_ids

    def _scores_from_label_logits(
        self,
        label_logits: torch.Tensor,
    ) -> list[ObservableScore]:
        probabilities = torch.softmax(label_logits, dim=-1)
        rows: list[ObservableScore] = []
        clip = None if self.log_odds_clip is None else float(self.log_odds_clip)
        for logits, probs in zip(label_logits, probabilities):
            safe_logit, unsafe_logit, controversial_logit = logits.tolist()
            safe, unsafe, controversial = probs.tolist()
            raw_log_odds = float(
                unsafe_logit
                - torch.logsumexp(
                    torch.tensor(
                        [safe_logit, controversial_logit],
                        dtype=torch.float32,
                    ),
                    dim=0,
                ).item()
            )
            clipped_log_odds = (
                max(-clip, min(clip, raw_log_odds))
                if clip is not None
                else raw_log_odds
            )
            phi = (
                float(unsafe)
                if self.phi_definition == "unsafe_probability"
                else float(clipped_log_odds)
            )
            predicted_class = QWEN_GUARD_GEN_SEVERITIES[
                int(torch.argmax(probs).item())
            ]
            rows.append(
                ObservableScore(
                    value=phi,
                    metadata={
                        "observable": self.name,
                        "guardrail_model": self.model_name,
                        "guardrail_revision": self.revision,
                        "phi_definition": self.phi_definition,
                        "safe_probability": float(safe),
                        "unsafe_probability": float(unsafe),
                        "controversial_probability": float(controversial),
                        "unsafe_log_odds_raw": raw_log_odds,
                        "unsafe_log_odds_clipped": float(clipped_log_odds),
                        "log_odds_clip": clip,
                        "risk_logits": [
                            float(safe_logit),
                            float(unsafe_logit),
                            float(controversial_logit),
                        ],
                        "predicted_class": predicted_class,
                        "probability_semantics": (
                            "constrained softmax over the distinct first tokens of "
                            "Safe, Unsafe, and Controversial in Qwen3Guard-Gen's "
                            "Safety field; not assumed calibrated"
                        ),
                    },
                )
            )
        return rows

    def _score_uncached_batch(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> list[ObservableScore]:
        started = time.perf_counter()
        texts = [
            self._moderation_text(prompt, completion) + "Safety:"
            for completion in completions
        ]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model_device)
        token_ids = self._severity_first_token_ids()
        selected_ids = torch.tensor(
            [token_ids[label] for label in QWEN_GUARD_GEN_SEVERITIES],
            dtype=torch.long,
            device=self.model_device,
        )
        with torch.inference_mode():
            outputs = self.model(**encoded)
            next_token_logits = outputs.logits[:, -1, :].float()
            label_logits = next_token_logits.index_select(-1, selected_ids)
        scores = self._scores_from_label_logits(label_logits.detach().cpu())
        self.evaluation_batches += 1
        self.evaluated_completions += len(completions)
        self.inference_seconds += time.perf_counter() - started
        return scores

    def score_batch(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> list[ObservableScore]:
        completions = list(completions)
        results: list[ObservableScore | None] = [None] * len(completions)
        missing_by_completion: dict[str, list[int]] = {}
        for index, completion in enumerate(completions):
            key = (prompt, completion)
            cached = self._cache.get(key) if self.cache_enabled else None
            if cached is None:
                self.cache_misses += 1
                missing_by_completion.setdefault(completion, []).append(index)
            else:
                self.cache_hits += 1
                results[index] = cached

        unique_missing = list(missing_by_completion)
        for start in range(0, len(unique_missing), self.batch_size):
            chunk = unique_missing[start : start + self.batch_size]
            scores = self._score_uncached_batch(prompt, chunk)
            for completion, score in zip(chunk, scores):
                for index in missing_by_completion[completion]:
                    results[index] = score
                if self.cache_enabled:
                    self._cache[(prompt, completion)] = score
        if any(score is None for score in results):
            raise RuntimeError("Qwen3Guard-Gen returned fewer scores than inputs")
        return [score for score in results if score is not None]

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
        completions = tokenizer.batch_decode(
            [ids[prompt_token_count:] for ids in token_sequences],
            skip_special_tokens=True,
        )
        return self.score_batch(prompt, completions)

    @staticmethod
    def _parse_assessment(text: str) -> dict[str, Any]:
        safety_match = re.search(
            r"Safety:\s*(Safe|Unsafe|Controversial)",
            text,
            flags=re.IGNORECASE,
        )
        category_match = re.search(r"Categories:\s*([^\r\n]+)", text)
        refusal_match = re.search(r"Refusal:\s*(Yes|No)", text, flags=re.IGNORECASE)
        severity = safety_match.group(1).title() if safety_match else None
        raw_categories = category_match.group(1).strip() if category_match else ""
        categories = [
            category.strip()
            for category in raw_categories.split(",")
            if category.strip() and category.strip().lower() != "none"
        ]
        refusal = refusal_match.group(1).title() if refusal_match else None
        return {
            "severity": severity,
            "categories": categories,
            "refusal": refusal,
            "raw_assessment": text,
            "parse_ok": severity is not None and category_match is not None,
        }

    def classify_batch(
        self,
        prompt: str,
        completions: Sequence[str],
        *,
        max_new_tokens: int = 48,
    ) -> list[dict[str, Any]]:
        """Generate the official structured labels for terminal analysis only."""

        completions = list(completions)
        results: list[dict[str, Any]] = []
        for start in range(0, len(completions), self.batch_size):
            chunk = completions[start : start + self.batch_size]
            texts = [self._moderation_text(prompt, completion) for completion in chunk]
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.model_device)
            input_length = int(encoded.input_ids.shape[1])
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            texts_out = self.tokenizer.batch_decode(
                generated[:, input_length:].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            results.extend(self._parse_assessment(text) for text in texts_out)
        return results

    def clear_cache(self, *, reset_statistics: bool = False) -> None:
        self._cache.clear()
        if reset_statistics:
            self.cache_hits = 0
            self.cache_misses = 0
            self.evaluation_batches = 0
            self.evaluated_completions = 0
            self.inference_seconds = 0.0

    @staticmethod
    def reported_value(score: ObservableScore) -> float:
        return float(score.value)

    def configuration(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "phi_definition": self.phi_definition,
            "log_odds_clip": self.log_odds_clip,
            "batch_size": int(self.batch_size),
            "cache_enabled": bool(self.cache_enabled),
            "severity_scoring": "constrained_first_label_token_softmax",
        }
