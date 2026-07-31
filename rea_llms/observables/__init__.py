from __future__ import annotations

from rea_llms.observables.ari import ARIObservable
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTION_OFFICIAL_PUBLIC,
    ARI_CONVENTIONS,
)


def create_observable(name: str, **kwargs) -> ARIObservable:
    """Construct a registered full-completion observable by name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"ari", "automated_readability_index"}:
        return ARIObservable(**kwargs)
    raise ValueError(f"Unknown observable: {name!r}")


__all__ = [
    "ARIObservable",
    "ARI_CONVENTION_CURRENT",
    "ARI_CONVENTION_OFFICIAL_PUBLIC",
    "ARI_CONVENTIONS",
    "create_observable",
]
