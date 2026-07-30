from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class BaseTailEstimates:
    """SNIS correction of one final SMC population back to the base model."""

    thresholds: torch.Tensor
    raw_counts: torch.Tensor
    raw_tilted_fractions: torch.Tensor
    estimated_probabilities: torch.Tensor
    corrected_log_weights: torch.Tensor
    corrected_ess: torch.Tensor
    corrected_ess_over_n: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        thresholds = self.thresholds.detach().cpu().tolist()
        counts = self.raw_counts.detach().cpu().tolist()
        fractions = self.raw_tilted_fractions.detach().cpu().tolist()
        probabilities = self.estimated_probabilities.detach().cpu().tolist()
        return {
            "corrected_importance_ess": float(self.corrected_ess.detach().cpu()),
            "corrected_importance_ess_over_n": float(
                self.corrected_ess_over_n.detach().cpu()
            ),
            "tails": [
                {
                    "threshold": float(threshold),
                    "raw_count": int(count),
                    "raw_tilted_fraction": float(fraction),
                    "estimated_probability_under_p_m": float(probability),
                }
                for threshold, count, fraction, probability in zip(
                    thresholds,
                    counts,
                    fractions,
                    probabilities,
                )
            ],
        }


def _as_vector(values, *, name: str, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return tensor


def corrected_base_log_weights(
    phi,
    smc_weights,
    lambda_final: float,
    *,
    weights_are_log: bool = False,
) -> torch.Tensor:
    """Return normalized log weights that correct final SMC samples to p_M.

    If ``weights_are_log`` is false, ``smc_weights`` contains non-negative SMC
    weights. Otherwise it contains their logarithms. The calculation is always
    normalized with ``torch.logsumexp`` and remains on the input tensor device.
    """

    if not math.isfinite(float(lambda_final)):
        raise ValueError("lambda_final must be finite")
    phi_tensor = _as_vector(phi, name="phi")
    if not bool(torch.isfinite(phi_tensor).all().item()):
        raise ValueError("phi must contain only finite values")
    weights = _as_vector(
        smc_weights,
        name="smc_weights",
        device=phi_tensor.device,
    )
    if weights.shape != phi_tensor.shape:
        raise ValueError("phi and smc_weights must have the same length")

    if weights_are_log:
        valid = ~torch.isnan(weights) & ~torch.isposinf(weights)
        if not bool(valid.all().item()) or not bool(torch.isfinite(weights).any().item()):
            raise ValueError("log SMC weights must be finite or -inf, with one finite value")
        smc_log_weights = weights
    else:
        if not bool(torch.isfinite(weights).all().item()) or not bool(
            (weights >= 0.0).all().item()
        ):
            raise ValueError("SMC weights must be finite and non-negative")
        if not bool((weights > 0.0).any().item()):
            raise ValueError("at least one SMC weight must be positive")
        negative_infinity = torch.full_like(weights, -torch.inf)
        smc_log_weights = torch.where(weights > 0.0, torch.log(weights), negative_infinity)

    corrected = smc_log_weights + float(lambda_final) * phi_tensor
    log_normalizer = torch.logsumexp(corrected, dim=0)
    if not bool(torch.isfinite(log_normalizer).item()):
        raise ValueError("corrected importance weights have a non-finite normalizer")
    return corrected - log_normalizer


def estimate_base_expectation(
    phi,
    smc_weights,
    lambda_final: float,
    f_values,
    *,
    weights_are_log: bool = False,
) -> torch.Tensor:
    """Estimate E_pM[f(X)] by self-normalized importance sampling.

    The final SMC population approximates
    ``pi_lambda proportional to p_M * exp(-lambda * phi)``. Therefore its
    normalized correction is ``W_i * exp(lambda * phi_i)``. This estimator is
    consistent but generally biased at finite particle count.
    """

    corrected_log_weights = corrected_base_log_weights(
        phi,
        smc_weights,
        lambda_final,
        weights_are_log=weights_are_log,
    )
    values = torch.as_tensor(
        f_values,
        dtype=torch.float64,
        device=corrected_log_weights.device,
    )
    if values.ndim == 0 or values.shape[-1] != corrected_log_weights.numel():
        raise ValueError("the last f_values dimension must match the particle count")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("f_values must contain only finite values")
    return torch.sum(values * torch.exp(corrected_log_weights), dim=-1)


def estimate_base_tail_probabilities(
    phi,
    smc_weights,
    lambda_final: float,
    thresholds: Sequence[float] | torch.Tensor,
    *,
    weights_are_log: bool = False,
) -> BaseTailEstimates:
    """Estimate all requested p_M tail probabilities in one broadcasted pass."""

    phi_tensor = _as_vector(phi, name="phi")
    threshold_tensor = _as_vector(
        thresholds,
        name="thresholds",
        device=phi_tensor.device,
    )
    if not bool(torch.isfinite(threshold_tensor).all().item()):
        raise ValueError("thresholds must contain only finite values")
    corrected_log_weights = corrected_base_log_weights(
        phi_tensor,
        smc_weights,
        lambda_final,
        weights_are_log=weights_are_log,
    )
    events = phi_tensor.unsqueeze(0) >= threshold_tensor.unsqueeze(1)
    event_values = events.to(torch.float64)
    probabilities = torch.sum(
        event_values * torch.exp(corrected_log_weights).unsqueeze(0),
        dim=1,
    )
    raw_counts = events.sum(dim=1)
    raw_fractions = raw_counts.to(torch.float64) / phi_tensor.numel()
    corrected_ess = torch.exp(-torch.logsumexp(2.0 * corrected_log_weights, dim=0))
    corrected_ess = torch.clamp(
        corrected_ess,
        min=1.0,
        max=float(phi_tensor.numel()),
    )
    return BaseTailEstimates(
        thresholds=threshold_tensor,
        raw_counts=raw_counts,
        raw_tilted_fractions=raw_fractions,
        estimated_probabilities=probabilities,
        corrected_log_weights=corrected_log_weights,
        corrected_ess=corrected_ess,
        corrected_ess_over_n=corrected_ess / phi_tensor.numel(),
    )


def estimate_base_tails_from_final_samples(
    samples: Sequence[dict[str, Any]],
    *,
    lambda_final: float,
    thresholds: Sequence[float],
) -> BaseTailEstimates:
    """Estimate tails from serialized final particles, never intermediate levels."""

    if not samples:
        raise ValueError("final samples must be non-empty")
    for sample in samples:
        if "phi" not in sample or "log_weight" not in sample:
            raise ValueError("each final sample requires phi and log_weight")
        sample_lambda = float(sample.get("lambda_final", math.nan))
        if not math.isclose(
            sample_lambda,
            float(lambda_final),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("all samples must belong to the requested final lambda")
    return estimate_base_tail_probabilities(
        [float(sample["phi"]) for sample in samples],
        [float(sample["log_weight"]) for sample in samples],
        lambda_final,
        thresholds,
        weights_are_log=True,
    )


def overlap_is_usable(
    corrected_ess_over_n: float,
    raw_event_fraction: float,
    *,
    min_corrected_ess_over_n: float,
    min_raw_event_fraction: float,
    max_raw_event_fraction: float,
) -> bool:
    """Classify overlap diagnostically without changing an SNIS estimate."""

    values = (
        corrected_ess_over_n,
        raw_event_fraction,
        min_corrected_ess_over_n,
        min_raw_event_fraction,
        max_raw_event_fraction,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("overlap diagnostic values must be finite")
    if not 0.0 <= min_corrected_ess_over_n <= 1.0:
        raise ValueError("min_corrected_ess_over_n must be in [0, 1]")
    if not 0.0 <= min_raw_event_fraction <= max_raw_event_fraction <= 1.0:
        raise ValueError("raw event fraction bounds must lie in [0, 1]")
    if not 0.0 <= corrected_ess_over_n <= 1.0 + 1e-12:
        raise ValueError("corrected_ess_over_n must be in [0, 1]")
    if not 0.0 <= raw_event_fraction <= 1.0:
        raise ValueError("raw_event_fraction must be in [0, 1]")
    return bool(
        corrected_ess_over_n >= min_corrected_ess_over_n
        and min_raw_event_fraction
        <= raw_event_fraction
        <= max_raw_event_fraction
    )


def _seed_independent_configuration(payload: dict[str, Any]) -> str:
    config = dict(payload.get("config", {}))
    config.pop("seed", None)
    return json.dumps(
        {
            "config": config,
            "repository_commit": payload.get("repository_commit"),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def aggregate_terminal_lambda_runs(
    payloads: Sequence[dict[str, Any]],
    *,
    min_corrected_ess_over_n: float,
    min_raw_event_fraction: float,
    max_raw_event_fraction: float,
) -> dict[str, Any]:
    """Aggregate independent seeds within, never across, terminal configurations."""

    if not payloads:
        raise ValueError("at least one completed run is required")
    configuration_state: dict[str, dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("algorithm") != "suffix_direct_mh_smc":
            raise ValueError("all runs must use suffix_direct_mh_smc")
        label = str(payload.get("configuration_label", "")).strip()
        if not label:
            raise ValueError("every sweep run requires a configuration_label")
        seed = int(payload["seed"])
        summary = payload["summary"]
        lambda_final = float(summary["lambda_final"])
        corrected_ess_over_n = float(summary["corrected_importance_ess_over_n"])
        thresholds = tuple(float(row["threshold"]) for row in summary["tails"])
        signature = _seed_independent_configuration(payload)
        state = configuration_state.setdefault(
            label,
            {
                "signature": signature,
                "lambda_final": lambda_final,
                "thresholds": thresholds,
                "seeds": set(),
                "ess_by_seed": {},
            },
        )
        if state["signature"] != signature:
            raise ValueError(f"configuration {label!r} mixes different run parameters")
        if not math.isclose(
            float(state["lambda_final"]),
            lambda_final,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"configuration {label!r} mixes terminal lambdas")
        if state["thresholds"] != thresholds:
            raise ValueError(f"configuration {label!r} mixes tail thresholds")
        if seed in state["seeds"]:
            raise ValueError(f"configuration {label!r} repeats seed {seed}")
        state["seeds"].add(seed)
        state["ess_by_seed"][seed] = corrected_ess_over_n
        for tail in summary["tails"]:
            raw_fraction = float(tail["raw_tilted_fraction"])
            estimate = float(tail["estimated_probability_under_p_m"])
            run_rows.append(
                {
                    "configuration": label,
                    "lambda_final": lambda_final,
                    "threshold": float(tail["threshold"]),
                    "seed": seed,
                    "raw_event_count": int(tail["raw_count"]),
                    "raw_event_fraction": raw_fraction,
                    "estimated_probability_under_p_m": estimate,
                    "corrected_ess_over_n": corrected_ess_over_n,
                    "overlap_usable": overlap_is_usable(
                        corrected_ess_over_n,
                        raw_fraction,
                        min_corrected_ess_over_n=min_corrected_ess_over_n,
                        min_raw_event_fraction=min_raw_event_fraction,
                        max_raw_event_fraction=max_raw_event_fraction,
                    ),
                }
            )

    grouped_rows: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in run_rows:
        grouped_rows.setdefault(
            (str(row["configuration"]), float(row["threshold"])),
            [],
        ).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (label, threshold), rows in grouped_rows.items():
        estimates = [float(row["estimated_probability_under_p_m"]) for row in rows]
        corrected_ess = [float(row["corrected_ess_over_n"]) for row in rows]
        raw_fractions = [float(row["raw_event_fraction"]) for row in rows]
        raw_counts = [int(row["raw_event_count"]) for row in rows]
        aggregate_rows.append(
            {
                "configuration": label,
                "lambda_final": float(rows[0]["lambda_final"]),
                "threshold": threshold,
                "completed_seeds": len(rows),
                "mean_raw_event_count": float(statistics.fmean(raw_counts)),
                "mean_raw_event_fraction": float(statistics.fmean(raw_fractions)),
                "mean_estimated_probability_under_p_m": float(
                    statistics.fmean(estimates)
                ),
                "std_estimated_probability_under_p_m": (
                    float(statistics.stdev(estimates)) if len(estimates) > 1 else None
                ),
                "minimum_corrected_ess_over_n": min(corrected_ess),
                "mean_corrected_ess_over_n": float(statistics.fmean(corrected_ess)),
                "usable_seed_count": sum(bool(row["overlap_usable"]) for row in rows),
                "overlap_usable": all(bool(row["overlap_usable"]) for row in rows),
            }
        )
    aggregate_rows.sort(
        key=lambda row: (
            float(row["lambda_final"]),
            float(row["threshold"]),
            str(row["configuration"]),
        )
    )
    run_rows.sort(
        key=lambda row: (
            float(row["lambda_final"]),
            float(row["threshold"]),
            int(row["seed"]),
        )
    )
    configuration_rows = [
        {
            "configuration": label,
            "lambda_final": float(state["lambda_final"]),
            "completed_seeds": len(state["seeds"]),
            "minimum_corrected_ess_over_n": min(state["ess_by_seed"].values()),
            "mean_corrected_ess_over_n": float(
                statistics.fmean(state["ess_by_seed"].values())
            ),
        }
        for label, state in configuration_state.items()
    ]
    configuration_rows.sort(key=lambda row: float(row["lambda_final"]))
    return {
        "diagnostic_thresholds": {
            "min_corrected_ess_over_n": float(min_corrected_ess_over_n),
            "min_raw_event_fraction": float(min_raw_event_fraction),
            "max_raw_event_fraction": float(max_raw_event_fraction),
        },
        "run_estimates": run_rows,
        "configuration_rows": configuration_rows,
        "aggregate_rows": aggregate_rows,
    }
