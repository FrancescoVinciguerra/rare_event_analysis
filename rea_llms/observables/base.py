from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservableScore:
    """One scalar potential value plus exact evaluation metadata."""

    value: float
    metadata: dict[str, Any]
