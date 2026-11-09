"""Small statistical summaries shared by direct and SMC validation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


WILSON_Z_95 = 1.959963984540054


def wilson_interval(
    count: int,
    total: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total <= 0 or count < 0 or count > total:
        raise ValueError("Wilson interval requires 0 <= count <= total and total > 0")
    proportion = count / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def rule_of_three_upper_bound(count: int, total: int) -> float | None:
    """Return ``3/N`` only for a zero-count event."""

    if total <= 0 or count < 0:
        raise ValueError("count must be non-negative and total must be positive")
    return min(1.0, 3.0 / total) if count == 0 else None


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute weighted quantiles without treating weights as multiplicities."""

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)
    if value_array.ndim != 1 or weight_array.shape != value_array.shape:
        raise ValueError("values and weights must be one-dimensional and equally sized")
    if value_array.size == 0 or not np.isfinite(value_array).all():
        raise ValueError("values must be non-empty and finite")
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(weight_array > 0.0):
        raise ValueError("at least one weight must be positive")
    if np.any((quantile_array < 0.0) | (quantile_array > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")

    positive = weight_array > 0.0
    order = np.argsort(value_array[positive], kind="stable")
    sorted_values = value_array[positive][order]
    sorted_weights = weight_array[positive][order]
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    return np.interp(
        quantile_array,
        positions,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def boxplot_statistics(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return quartiles and Tukey whiskers for direct or weighted SMC data."""

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 1 or value_array.size == 0 or not np.isfinite(value_array).all():
        raise ValueError("values must be a non-empty finite vector")
    if weights is None:
        q1, median, q3 = np.quantile(value_array, [0.25, 0.5, 0.75], method="linear")
    else:
        q1, median, q3 = weighted_quantile(
            value_array,
            weights,
            [0.25, 0.5, 0.75],
        )
    iqr = float(q3 - q1)
    low_fence = float(q1 - 1.5 * iqr)
    high_fence = float(q3 + 1.5 * iqr)
    inlier_mask = (value_array >= low_fence) & (value_array <= high_fence)
    inliers = value_array[inlier_mask]
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": iqr,
        "whisker_low": float(np.min(inliers)),
        "whisker_high": float(np.max(inliers)),
        "outlier_count": int(value_array.size - inliers.size),
    }
