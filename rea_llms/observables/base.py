from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ObservableScore:
    """One scalar potential value plus exact evaluation metadata."""

    value: float
    metadata: dict[str, Any]


class CompletionObservable(Protocol):
    """Minimal interface consumed by the full-completion samplers."""

    name: str

    def score_completion(self, prompt: str, completion: str) -> ObservableScore:
        """Return one finite scalar potential and serializable metadata."""


def observable_configuration(observable: Any) -> dict[str, Any]:
    """Return stable, serializable identity fields for caches and reports."""

    describe = getattr(observable, "configuration", None)
    if callable(describe):
        return dict(describe())
    return {"name": str(getattr(observable, "name", type(observable).__name__))}


def reported_value(observable: Any, score: ObservableScore) -> float:
    """Return the human-facing metric while keeping ``score.value`` as phi."""

    selector = getattr(observable, "reported_value", None)
    return float(selector(score) if callable(selector) else score.value)
