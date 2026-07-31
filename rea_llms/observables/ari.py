from __future__ import annotations

from dataclasses import dataclass, field

from textstat import automated_readability_index

from rea_llms.observables.base import ObservableScore
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    text_for_ari_from_token_ids,
    validate_ari_convention,
)


@dataclass
class ARIObservable:
    """Exact textstat ARI evaluated on decoded prompt-plus-completion text."""

    mode: str = "capped"
    cap: float = 15.0
    backend: str = "exact_cached"
    convention: str = ARI_CONVENTION_CURRENT
    cache_enabled: bool = True
    name: str = "ari"
    _cache: dict[str, ObservableScore] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "capped"}:
            raise ValueError("mode must be one of {'raw', 'capped'}")
        if self.backend not in {"exact", "exact_cached", "textstat"}:
            raise ValueError("ARI backend must be one of {'exact', 'exact_cached', 'textstat'}")
        validate_ari_convention(self.convention)

    def score_text(self, text: str) -> ObservableScore:
        """Score one exact text, optionally reusing an exact-text cache entry."""

        if self.cache_enabled and self.backend == "exact_cached" and text in self._cache:
            return self._cache[text]
        raw = float(automated_readability_index(text))
        capped = min(raw, float(self.cap))
        value = raw if self.mode == "raw" else capped
        score = ObservableScore(
            value=float(value),
            metadata={
                "raw_ari": float(raw),
                "capped_ari": float(capped),
                "ari_mode": self.mode,
                "ari_cap": float(self.cap),
                "backend": self.backend,
                "ari_convention": self.convention,
            },
        )
        if self.cache_enabled and self.backend == "exact_cached":
            self._cache[text] = score
        return score

    def score_completion(self, prompt: str, completion: str) -> ObservableScore:
        """Score a completion with the repository's prompt concatenation rule."""

        return self.score_text(prompt + completion)

    def score_token_sequences(
        self,
        *,
        tokenizer,
        prompt: str,
        token_sequences: list[list[int]],
        prompt_token_count: int,
    ) -> list[ObservableScore]:
        """Decode token sequences using the explicitly selected ARI convention."""

        scores = []
        for ids in token_sequences:
            text = text_for_ari_from_token_ids(
                tokenizer,
                prompt=prompt,
                token_ids=ids,
                prompt_token_count=prompt_token_count,
                convention=self.convention,
            )
            scores.append(self.score_text(text))
        return scores
