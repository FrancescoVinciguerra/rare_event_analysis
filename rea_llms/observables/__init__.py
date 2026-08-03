from __future__ import annotations

from typing import Any

from rea_llms.observables.ari import ARIObservable
from rea_llms.observables.canary import CanaryLeakageObservable
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTION_OFFICIAL_PUBLIC,
    ARI_CONVENTIONS,
)


OBSERVABLE_NAMES = ("ari", "canary_leakage", "qwen_guard")


def create_observable(name: str, **kwargs) -> Any:
    """Construct a registered full-completion observable by name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"ari", "automated_readability_index"}:
        return ARIObservable(**kwargs)
    if normalized in {"canary", "canary_leakage", "secret_leakage"}:
        return CanaryLeakageObservable(**kwargs)
    if normalized in {"qwen_guard", "qwen3guard_stream"}:
        from rea_llms.observables.qwen_guard import QwenGuardHarmfulnessObservable

        return QwenGuardHarmfulnessObservable(**kwargs)
    raise ValueError(f"Unknown observable: {name!r}")


__all__ = [
    "ARIObservable",
    "CanaryLeakageObservable",
    "ARI_CONVENTION_CURRENT",
    "ARI_CONVENTION_OFFICIAL_PUBLIC",
    "ARI_CONVENTIONS",
    "OBSERVABLE_NAMES",
    "create_observable",
]
