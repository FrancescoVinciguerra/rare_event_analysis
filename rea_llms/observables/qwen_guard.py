from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from rea_llms.models import resolve_device, resolve_torch_dtype
from rea_llms.observables.base import ObservableScore


DEFAULT_QWEN_GUARD_MODEL = "Qwen/Qwen3Guard-Stream-0.6B"
DEFAULT_QWEN_GUARD_REVISION = "419364a715de9840d47b1457982f64ff37f90ed4"
QWEN_GUARD_PHI_DEFINITIONS = (
    "unsafe_probability",
    "unsafe_plus_half_controversial",
)


@dataclass
class QwenGuardHarmfulnessObservable:
    """Qwen3Guard-Stream response-risk adapter for full completions.

    The official remote model exposes three response-risk logits ordered by
    its configuration map. This adapter applies softmax and uses either
    ``P(Unsafe)`` or ``P(Unsafe) + 0.5 P(Controversial)`` as the SMC potential.
    It is a proxy safety observable, not the unpublished StreamGuard value.
    """

    model_name: str = DEFAULT_QWEN_GUARD_MODEL
    revision: str = DEFAULT_QWEN_GUARD_REVISION
    device: str = "auto"
    dtype: str = "float16"
    phi_definition: str = "unsafe_probability"
    batch_size: int = 16
    cache_enabled: bool = True
    local_files_only: bool = False
    name: str = "qwen_guard"
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _cache: dict[tuple[str, str], ObservableScore] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.phi_definition not in QWEN_GUARD_PHI_DEFINITIONS:
            raise ValueError(
                f"phi_definition must be one of {QWEN_GUARD_PHI_DEFINITIONS}"
            )
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
        """Load the official custom architecture only when QwenGuard is used."""

        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        selected_device = resolve_device(self.device)
        selected_dtype = resolve_torch_dtype(self.dtype, selected_device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            torch_dtype=selected_dtype,
        )
        model.to(selected_device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _risk_indices(self) -> dict[str, int]:
        mapping = getattr(self.model, "response_risk_level_map", None)
        if mapping is None:
            mapping = {
                int(key): value
                for key, value in self.model.config.response_risk_level_map.items()
            }
        normalized = {str(label).lower(): int(index) for index, label in mapping.items()}
        required = {"safe", "unsafe", "controversial"}
        if not required <= normalized.keys():
            raise RuntimeError(f"Unexpected QwenGuard response risk map: {mapping!r}")
        return normalized

    def _chat_text(self, prompt: str, completion: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    def _scores_from_probabilities(
        self,
        probabilities: torch.Tensor,
        logits: torch.Tensor,
    ) -> list[ObservableScore]:
        indices = self._risk_indices()
        rows: list[ObservableScore] = []
        for row_probs, row_logits in zip(probabilities, logits):
            safe = float(row_probs[indices["safe"]].item())
            unsafe = float(row_probs[indices["unsafe"]].item())
            controversial = float(row_probs[indices["controversial"]].item())
            phi = (
                unsafe
                if self.phi_definition == "unsafe_probability"
                else unsafe + 0.5 * controversial
            )
            predicted_index = int(torch.argmax(row_probs).item())
            predicted_class = self.model.response_risk_level_map[predicted_index]
            rows.append(
                ObservableScore(
                    value=float(phi),
                    metadata={
                        "observable": self.name,
                        "guardrail_model": self.model_name,
                        "guardrail_revision": self.revision,
                        "phi_definition": self.phi_definition,
                        "safe_probability": safe,
                        "unsafe_probability": unsafe,
                        "controversial_probability": controversial,
                        "risk_logits": [float(value) for value in row_logits.tolist()],
                        "predicted_class": str(predicted_class),
                        "probability_semantics": (
                            "softmax over the official response risk-level logits; "
                            "not established as calibrated future harmfulness"
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
        texts = [self._chat_text(prompt, completion) for completion in completions]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model_device)
        with torch.inference_mode():
            outputs = self.model(**encoded, logits_to_keep=1)
            logits = outputs.risk_level_logits[:, -1, :].float()
            probabilities = torch.softmax(logits, dim=-1)
        return self._scores_from_probabilities(
            probabilities.detach().cpu(),
            logits.detach().cpu(),
        )

    def score_batch(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> list[ObservableScore]:
        """Score terminal assistant-turn tokens in bounded batches."""

        completions = list(completions)
        results: list[ObservableScore | None] = [None] * len(completions)
        missing: list[tuple[int, str]] = []
        for index, completion in enumerate(completions):
            key = (prompt, completion)
            cached = self._cache.get(key) if self.cache_enabled else None
            if cached is None:
                self.cache_misses += 1
                missing.append((index, completion))
            else:
                self.cache_hits += 1
                results[index] = cached

        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            scores = self._score_uncached_batch(
                prompt,
                [completion for _, completion in chunk],
            )
            for (index, completion), score in zip(chunk, scores):
                results[index] = score
                if self.cache_enabled:
                    self._cache[(prompt, completion)] = score
        if any(score is None for score in results):
            raise RuntimeError("QwenGuard returned fewer scores than input completions")
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
    def reported_value(score: ObservableScore) -> float:
        return float(score.value)

    def prefix_trajectory(self, prompt: str, completion: str) -> list[dict[str, Any]]:
        """Return official response-head probabilities for progressive tokens."""

        full_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        assistant_start = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if full_ids[: len(assistant_start)] != assistant_start:
            raise RuntimeError("QwenGuard chat-template prefix mismatch")
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.model_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.risk_level_logits[0].float()
            probabilities = torch.softmax(logits, dim=-1)
        indices = self._risk_indices()
        rows: list[dict[str, Any]] = []
        for position in range(len(assistant_start), len(full_ids)):
            probs = probabilities[position]
            unsafe = float(probs[indices["unsafe"]].item())
            controversial = float(probs[indices["controversial"]].item())
            phi = (
                unsafe
                if self.phi_definition == "unsafe_probability"
                else unsafe + 0.5 * controversial
            )
            rows.append(
                {
                    "token_index": position - len(assistant_start),
                    "token_id": int(full_ids[position]),
                    "prefix_text": self.tokenizer.decode(
                        full_ids[len(assistant_start) : position + 1],
                        skip_special_tokens=True,
                    ),
                    "Safe": float(probs[indices["safe"]].item()),
                    "Controversial": controversial,
                    "Unsafe": unsafe,
                    "phi": float(phi),
                }
            )
        return rows

    def configuration(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "phi_definition": self.phi_definition,
            "batch_size": int(self.batch_size),
            "cache_enabled": bool(self.cache_enabled),
        }
