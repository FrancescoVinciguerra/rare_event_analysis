from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    """Hash a complete JSON-compatible run configuration deterministically."""
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def completed_run_action(
    output_dir: Path,
    fingerprint: str,
    *,
    skip_completed: bool,
    overwrite: bool,
) -> str:
    """Return 'run' or 'skip', rejecting incompatible completed output."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return "run"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    existing = payload.get("configuration_fingerprint")
    if existing == fingerprint and skip_completed:
        return "skip"
    if existing != fingerprint and not overwrite:
        raise FileExistsError(
            f"Incompatible completed run at {output_dir}: fingerprint {existing!r} "
            f"does not match {fingerprint!r}. Use a new experiment label or --overwrite."
        )
    if existing == fingerprint and not overwrite:
        raise FileExistsError(
            f"Completed run already exists at {output_dir}; use --skip-completed or --overwrite."
        )
    return "run"


def temporal_ess_initial_positive(values: Sequence[float]) -> float:
    """Geyer initial-positive-sequence ESS for one scalar time series."""
    samples = np.asarray(values, dtype=np.float64)
    n_samples = samples.size
    if n_samples == 0:
        raise ValueError("temporal ESS requires at least one sample")
    if n_samples < 3 or np.allclose(samples, samples[0]):
        return float(n_samples)
    centered = samples - samples.mean()
    variance = float(np.dot(centered, centered) / n_samples)
    if variance <= 0.0 or not math.isfinite(variance):
        return float(n_samples)
    autocorrelation = np.empty(n_samples, dtype=np.float64)
    autocorrelation[0] = 1.0
    for lag in range(1, n_samples):
        autocovariance = float(
            np.dot(centered[:-lag], centered[lag:]) / (n_samples - lag)
        )
        autocorrelation[lag] = autocovariance / variance

    retained_sum = 0.0
    lag = 1
    while lag < n_samples:
        pair_sum = autocorrelation[lag]
        if lag + 1 < n_samples:
            pair_sum += autocorrelation[lag + 1]
        if not math.isfinite(float(pair_sum)) or pair_sum <= 0.0:
            break
        retained_sum += float(pair_sum)
        lag += 2
    integrated_time = max(1.0, 1.0 + 2.0 * retained_sum)
    return float(np.clip(n_samples / integrated_time, 1.0, n_samples))


def split_rhat(chains: Sequence[Sequence[float]]) -> float | None:
    """Split-R-hat across independent TPS seed chains, when estimable."""
    if len(chains) < 2:
        return None
    arrays = [np.asarray(chain, dtype=np.float64) for chain in chains]
    half = min(array.size for array in arrays) // 2
    if half < 2:
        return None
    split = np.stack([part for array in arrays for part in (array[:half], array[-half:])])
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(half * np.var(np.mean(split, axis=1), ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else math.inf
    variance_hat = ((half - 1.0) / half) * within + between / half
    return float(math.sqrt(max(0.0, variance_hat / within)))


def summarize_independent_estimates(
    values: Sequence[float],
    runtimes: Sequence[float],
) -> dict[str, float | None]:
    estimates = np.asarray(values, dtype=np.float64)
    times = np.asarray(runtimes, dtype=np.float64)
    if estimates.size == 0 or estimates.size != times.size:
        raise ValueError("values and runtimes must be non-empty and have equal length")
    mean = float(estimates.mean())
    mean_runtime = float(times.mean())
    if estimates.size < 2:
        return {
            "mean": mean,
            "standard_deviation": None,
            "ci95_low": None,
            "ci95_high": None,
            "mean_runtime_seconds": mean_runtime,
            "empirical_estimator_efficiency": None,
        }
    standard_deviation = float(estimates.std(ddof=1))
    half_width = 1.96 * standard_deviation / math.sqrt(estimates.size)
    variance = standard_deviation**2
    efficiency = None if variance == 0.0 else float(1.0 / (mean_runtime * variance))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "mean_runtime_seconds": mean_runtime,
        "empirical_estimator_efficiency": efficiency,
    }
