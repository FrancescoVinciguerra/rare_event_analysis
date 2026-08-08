from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pymbar

from official_annealed_tps_experiment.annealed_tps import (
    OfficialAnnealedTPSResult,
)


@dataclass(frozen=True)
class MBARTailResult:
    """MBAR estimates at the unbiased state from several TPS windows."""

    tails: pd.DataFrame
    lambdas: np.ndarray
    sample_counts: np.ndarray
    log_normalizing_constants: np.ndarray
    overlap_matrix: np.ndarray
    effective_samples: np.ndarray


def collect_tps_window_scores(
    runs: Sequence[OfficialAnnealedTPSResult],
    *,
    burn_in_fraction: float = 0.5,
    score_field: str = "raw_ari",
) -> dict[float, np.ndarray]:
    """Collect post-burn-in scores from every bias window and independent run."""

    if not runs:
        raise ValueError("at least one TPS run is required")
    if not 0.0 <= burn_in_fraction < 1.0:
        raise ValueError("burn_in_fraction must lie in [0, 1)")

    reference_lambdas = tuple(float(value) for value in runs[0].config["lambdas"])
    grouped: dict[float, list[np.ndarray]] = {value: [] for value in reference_lambdas}
    for run in runs:
        lambdas = tuple(float(value) for value in run.config["lambdas"])
        if lambdas != reference_lambdas:
            raise ValueError("all TPS runs must use the same lambda schedule")
        if score_field not in run.trajectory:
            raise ValueError(f"trajectory does not contain {score_field!r}")
        lambda_indices = np.asarray(run.trajectory["lambda_index"], dtype=np.int64)
        scores = np.asarray(run.trajectory[score_field], dtype=np.float64)
        for index, lambda_value in enumerate(reference_lambdas):
            window = scores[lambda_indices == index]
            discard = int(np.floor(burn_in_fraction * window.size))
            retained = window[discard:]
            if retained.size == 0:
                raise ValueError("burn-in removed an entire TPS window")
            grouped[lambda_value].append(retained)

    return {
        lambda_value: np.concatenate(windows)
        for lambda_value, windows in grouped.items()
    }


def estimate_mbar_tails(
    scores_by_lambda: Mapping[float, Sequence[float] | np.ndarray],
    *,
    thresholds: Sequence[float],
) -> MBARTailResult:
    """Estimate unbiased tail probabilities with the paper's MBAR construction."""

    if not scores_by_lambda:
        raise ValueError("scores_by_lambda must not be empty")
    lambdas = np.asarray(list(scores_by_lambda), dtype=np.float64)
    zero_matches = np.flatnonzero(np.isclose(lambdas, 0.0, atol=1e-12, rtol=0.0))
    if zero_matches.size != 1:
        raise ValueError("exactly one lambda=0 state is required")
    score_arrays = [
        np.asarray(scores_by_lambda[float(value)], dtype=np.float64)
        for value in lambdas
    ]
    if any(values.ndim != 1 or values.size == 0 for values in score_arrays):
        raise ValueError("every MBAR state requires a non-empty score vector")
    if any(not np.isfinite(values).all() for values in score_arrays):
        raise ValueError("MBAR scores must be finite")

    sample_counts = np.asarray([values.size for values in score_arrays], dtype=np.int64)
    all_scores = np.concatenate(score_arrays)
    reduced_potentials = np.outer(lambdas, all_scores)
    estimator = pymbar.MBAR(reduced_potentials, sample_counts, verbose=False)

    free_energies = estimator.compute_free_energy_differences()["Delta_f"]
    zero_index = int(zero_matches[0])
    log_normalizing_constants = -np.asarray(
        free_energies[zero_index], dtype=np.float64
    )

    rows = []
    for threshold in map(float, thresholds):
        indicator = (all_scores >= threshold).astype(np.float64)
        # pymbar takes log(0) for samples outside an indicator event. Those
        # entries correctly carry zero observable mass.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in log",
                category=RuntimeWarning,
            )
            expectation = estimator.compute_expectations(indicator)
        rows.append(
            {
                "threshold": threshold,
                "probability": float(expectation["mu"][zero_index]),
                "standard_error": float(expectation["sigma"][zero_index]),
                "pooled_event_count": int(indicator.sum()),
            }
        )

    overlap = estimator.compute_overlap()["matrix"]
    effective_samples = estimator.compute_effective_sample_number()
    return MBARTailResult(
        tails=pd.DataFrame(rows),
        lambdas=lambdas,
        sample_counts=sample_counts,
        log_normalizing_constants=log_normalizing_constants,
        overlap_matrix=np.asarray(overlap, dtype=np.float64),
        effective_samples=np.asarray(effective_samples, dtype=np.float64),
    )
