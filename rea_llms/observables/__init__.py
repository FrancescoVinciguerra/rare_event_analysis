from __future__ import annotations

from typing import Any


OBSERVABLE_NAMES = ("ari", "canary_leakage", "qwen_guard", "qwen_guard_gen")


def create_observable(name: str, **kwargs) -> Any:
    """Construct a registered full-completion observable by name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"ari", "automated_readability_index"}:
        from rea_llms.observables.ari import ARIObservable

        return ARIObservable(**kwargs)
    if normalized in {"canary", "canary_leakage", "secret_leakage"}:
        from rea_llms.observables.canary import CanaryLeakageObservable

        return CanaryLeakageObservable(**kwargs)
    if normalized in {"qwen_guard", "qwen3guard_stream"}:
        from rea_llms.observables.qwen_guard import QwenGuardHarmfulnessObservable

        return QwenGuardHarmfulnessObservable(**kwargs)
    if normalized in {"qwen_guard_gen", "qwen3guard_gen"}:
        from rea_llms.observables.qwen_guard_gen import QwenGuardGenObservable

        return QwenGuardGenObservable(**kwargs)
    raise ValueError(f"Unknown observable: {name!r}")


def __getattr__(name: str) -> Any:
    if name == "ARIObservable":
        from rea_llms.observables.ari import ARIObservable

        return ARIObservable
    if name == "CanaryLeakageObservable":
        from rea_llms.observables.canary import CanaryLeakageObservable

        return CanaryLeakageObservable
    if name == "QwenGuardGenObservable":
        from rea_llms.observables.qwen_guard_gen import QwenGuardGenObservable

        return QwenGuardGenObservable
    if name in {
        "ARI_CONVENTION_CURRENT",
        "ARI_CONVENTION_OFFICIAL_PUBLIC",
        "ARI_CONVENTIONS",
    }:
        from rea_llms import scoring

        return getattr(scoring, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARIObservable",
    "CanaryLeakageObservable",
    "QwenGuardGenObservable",
    "ARI_CONVENTION_CURRENT",
    "ARI_CONVENTION_OFFICIAL_PUBLIC",
    "ARI_CONVENTIONS",
    "OBSERVABLE_NAMES",
    "create_observable",
]
