"""Selectable SMC potentials, independent of the native model and event.

The same Phi is passed to tempering, native RollBack--MH, and both reverse
corrections. Classifier probabilities and event definitions never change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ToxicityPotential:
    name: str = "log_probability"

    def __post_init__(self) -> None:
        if self.name not in {"log_probability", "logit"}:
            raise ValueError("smc.phi must be log_probability or logit")

    def from_classifier_row(self, row: Mapping[str, Any], *, positive_label_index: int = 1) -> float:
        if self.name == "log_probability":
            # Preserve the historical float32 log-softmax value exactly.
            value = float(row["reward"])
        else:
            logits = [float(x) for x in row["logits"]]
            if len(logits) < 2 or not 0 <= positive_label_index < len(logits):
                raise ValueError("Logit potential requires a positive class and at least one other class")
            if not all(math.isfinite(x) for x in logits):
                raise FloatingPointError("Non-finite classifier logits")
            other = [x for i, x in enumerate(logits) if i != positive_label_index]
            peak = max(other)
            log_other_mass = peak + math.log(sum(math.exp(x - peak) for x in other))
            value = logits[positive_label_index] - log_other_mass
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite SMC potential")
        return value

    def probability(self, value: float) -> float:
        """Recover q for checkpoint event counts; no clipping or epsilon."""
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite saved SMC potential")
        if self.name == "log_probability":
            return math.exp(value)
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        mass = math.exp(value)
        return mass / (1.0 + mass)

    def threshold_value(self, probability: float) -> float:
        if not 0 < probability < 1:
            raise ValueError("Event threshold must lie strictly between zero and one")
        value = math.log(probability)
        return value if self.name == "log_probability" else value - math.log1p(-probability)

    def manifest(self) -> dict[str, Any]:
        return dict(name=self.name,
                    definition="log(q)" if self.name == "log_probability" else "log(q/(1-q))",
                    source="saved classifier log-softmax" if self.name == "log_probability" else "positive classifier logit minus logsumexp(other logits)",
                    probability_clipping=None)


def potential_from_config(config: Mapping[str, Any]) -> ToxicityPotential:
    return ToxicityPotential(config.get("smc", {}).get("phi", "log_probability"))
