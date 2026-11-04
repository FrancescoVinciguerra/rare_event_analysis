"""Offline publication figures from frozen MDLM rare-event artifacts.

This program never loads a model and never generates or scores text.  It reads
the completed direct and SMC artifacts, verifies their identities, and writes
CSV-backed figures plus a short report into a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

mpl: Any = None
plt: Any = None


THRESHOLDS = (0.5, 0.9, 0.99, 0.999)
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
METHOD_LABEL = {
    "fk_bootstrap": "FK Steering",
    "trajectory_resample_move": "Trajectory SMC",
}
COLORS = {"fk_bootstrap": "#D55E00", "trajectory_resample_move": "#0072B2"}
MARKERS = {192: "o", 512: "s"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return -math.inf
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def logmeanexp(values: np.ndarray) -> float:
    return logsumexp(values) - math.log(values.size)


def quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    return tuple(float(x) for x in np.quantile(np.asarray(list(values)), [0.25, 0.5, 0.75]))


def bootstrap_rmse_ci(
    estimates: np.ndarray, reference: float, *, seed: int, replicates: int = 5000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    result = np.empty(replicates)
    for index in range(replicates):
        draw = rng.choice(estimates, size=estimates.size, replace=True)
        result[index] = math.sqrt(float(np.mean((draw - reference) ** 2)))
    low, high = np.quantile(result, [0.025, 0.975])
    return float(low), float(high)


def configure_style() -> None:
    load_plotting()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, figures: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(figures / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_plotting() -> None:
    """Import plotting only when figures are rendered, not during CPU extraction."""
    global mpl, plt
    if mpl is None or plt is None:
        import matplotlib as matplotlib_module
        import matplotlib.pyplot as pyplot_module

        mpl = matplotlib_module
        plt = pyplot_module


def read_render_backing(output: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Read the complete CSV backing so rendering can happen off-cluster."""
    csv_dir = output / "csv"

    def typed(name: str, integer: tuple[str, ...], floating: tuple[str, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = read_csv(csv_dir / name)
        for row in rows:
            for key in integer:
                if row.get(key, "") != "":
                    row[key] = int(float(row[key]))
            for key in floating:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
        return rows

    released = typed(
        "run_estimates_released.csv",
        ("particle_budget", "source_run_index"),
        ("threshold", "estimate", "direct_reference", "runtime_seconds", "decoder_particle_steps"),
    )
    rmse_rows = typed(
        "relative_rmse.csv",
        ("particle_budget",),
        (
            "threshold", "reference", "rmse", "relative_rmse", "relative_rmse_ci_low",
            "relative_rmse_ci_high", "median_runtime_seconds", "median_decoder_particle_steps",
        ),
    )
    component_rows = typed(
        "component_estimators.csv",
        ("particle_budget",),
        ("lambda", "threshold", "reverse_snis_estimate", "reverse_ess_over_n"),
    )
    ideal_rows = typed(
        "tilt_ideal_overlap.csv",
        (),
        ("lambda", "log_M_plus", "log_M_minus", "M_plus", "M_minus", "ideal_reverse_ess_over_n"),
    )
    tail_rows = typed("native_toxicity_tail.csv", ("tail_count",), ("toxicity_threshold", "tail_probability"))
    estimator_rows = typed(
        "estimator_comparison.csv",
        ("particle_budget",),
        ("threshold", "relative_rmse"),
    )
    return released, rmse_rows, component_rows, ideal_rows, tail_rows, estimator_rows


def direct_artifacts(
    artifact_root: Path, manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[Path]]:
    decoder_id = manifest["decoder_id"]
    evaluator_id = manifest["evaluator_id"]
    direct_root = artifact_root / "direct" / decoder_id
    evaluation_root = direct_root / "evaluations" / evaluator_id
    score_files = sorted(evaluation_root.glob("scores_*.jsonl"))
    if not score_files:
        raise FileNotFoundError(f"no score shards under {evaluation_root}")
    scores: list[float] = []
    rewards: list[float] = []
    sample_ids: set[str] = set()
    for path in score_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["decoder_id"] != decoder_id or row["evaluator_id"] != evaluator_id:
                    raise ValueError(f"identity mismatch in {path}")
                sample_id = str(row["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate direct sample {sample_id}")
                sample_ids.add(sample_id)
                scores.append(float(row["event_toxic_probability"]))
                rewards.append(float(row["reward"]))
    if len(scores) != 200_000:
        raise ValueError(f"expected 200000 direct scores, found {len(scores)}")
    score_array = np.asarray(scores, dtype=np.float64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    if not np.isfinite(score_array).all() or not np.isfinite(reward_array).all():
        raise FloatingPointError("direct scores/rewards contain NaN or Inf")
    if np.min(score_array) < 0.0 or np.max(score_array) > 1.0:
        raise ValueError("toxicity probability lies outside [0,1]")

    summary = read_json(evaluation_root / "summaries" / "summary_00200000.json")
    if int(summary["samples"]) != 200_000:
        raise ValueError("direct summary is not the frozen 200k summary")
    summary["combined_sampling_and_scoring_seconds"] = float(
        summary["generation_runtime_seconds"]
    ) + float(summary["scoring_runtime_seconds"])
    summary["seconds_per_evaluated_completion"] = (
        summary["combined_sampling_and_scoring_seconds"] / 200_000
    )
    return score_array, reward_array, summary, score_files


def comparison_entries(
    artifact_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    specifications = [
        (manifest["base_suite_id"], "submission"),
        (manifest["matched_runtime_suite_id"], "runtime_matched"),
        (manifest["confirmatory_suite_id"], "confirmatory"),
    ]
    entries: list[dict[str, Any]] = []
    for suite_id, cohort in specifications:
        completion_path = artifact_root / "comparisons" / suite_id / "completion.json"
        payload = read_json(completion_path)
        if payload["suite_id"] != suite_id:
            raise ValueError(f"suite mismatch in {completion_path}")
        for row in payload["grid"]:
            method = str(row["method_key"])
            if method not in METHOD_LABEL:
                continue
            entries.append(
                {
                    "suite_id": suite_id,
                    "cohort": cohort,
                    "method_key": method,
                    "particle_budget": int(row["particle_budget"]),
                    "smc_id": str(row["smc_id"]),
                    "run_directory": Path(row["run_directory"]),
                    "completion_path": completion_path,
                }
            )
    keys = [(x["method_key"], x["particle_budget"], x["cohort"]) for x in entries]
    if len(keys) != len(set(keys)) or len(entries) != 7:
        raise ValueError(f"expected seven unique method/budget/cohort entries, got {keys}")
    return entries


def component_analysis(
    entries: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    component_rows: list[dict[str, Any]] = []
    normalizer_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str, int, float], list[dict[str, float]]] = defaultdict(list)
    expected_decoder = manifest["decoder_id"]
    expected_evaluator = manifest["evaluator_id"]

    for entry in entries:
        paths = sorted((entry["run_directory"] / "components").glob("run_*_tilted_*.json"))
        if len(paths) != 128:
            raise ValueError(f"expected 128 tilted component files in {entry['run_directory']}")
        for path in paths:
            payload = read_json(path)
            if payload["decoder_id"] != expected_decoder or payload["evaluator_id"] != expected_evaluator:
                raise ValueError(f"decoder/evaluator mismatch in {path}")
            summary = payload["summary"]
            lam = float(summary["lambda"])
            if lam not in LAMBDAS:
                raise ValueError(f"unexpected lambda {lam} in {path}")
            particles = payload.get("particles")
            if not particles or len(particles) != int(summary["num_particles"]):
                raise ValueError(f"missing terminal particles in {path}")
            rewards = np.asarray([float(row["reward"]) for row in particles], dtype=np.float64)
            scores = np.asarray(
                [float(row["event_toxic_probability"]) for row in particles], dtype=np.float64
            )
            log_weights = np.asarray(
                [float(row["final_log_weight"]) for row in particles], dtype=np.float64
            )
            if not np.isfinite(rewards).all() or not np.isfinite(scores).all() or not np.isfinite(log_weights).all():
                raise FloatingPointError(f"non-finite particle data in {path}")
            if abs(logsumexp(log_weights)) > 1e-8:
                raise ValueError(f"terminal log weights are not normalized in {path}")
            log_z = float(summary["log_normalizer_estimate"])
            if not math.isfinite(log_z):
                raise FloatingPointError(f"non-finite log normalizer in {path}")
            base = {
                "method_key": entry["method_key"],
                "particle_budget": entry["particle_budget"],
                "cohort": entry["cohort"],
                "source_run_index": int(summary["run_index"]),
                "lambda": lam,
                "seed": int(summary["seed"]),
                "log_normalizer_estimate": log_z,
                "reverse_ess_over_n": float(summary["reverse_weight_ess_over_n"]),
                "runtime_seconds": float(summary["runtime_seconds"]),
                "decoder_particle_steps": int(summary["total_decoder_particle_steps"]),
                "distinct_root_fraction": float(summary["distinct_root_ancestor_fraction"]),
                "distinct_terminal_sequence_fraction": float(summary["distinct_terminal_sequence_fraction"]),
                "component_path": str(path),
            }
            normalizer_rows.append(base.copy())
            for level in payload.get("level_diagnostics", []):
                if level.get("log_normalizer_increment") is not None:
                    increment_rows.append(
                        {
                            **{key: base[key] for key in ("method_key", "particle_budget", "cohort", "source_run_index", "lambda", "seed")},
                            "level": int(level["level"]),
                            "log_normalizer_increment": float(level["log_normalizer_increment"]),
                        }
                    )
            for threshold in THRESHOLDS:
                event = scores >= threshold
                corrected = log_weights - lam * rewards
                log_denominator = logsumexp(corrected)
                log_numerator = logsumexp(corrected[event])
                snis = 0.0 if log_numerator == -math.inf else math.exp(log_numerator - log_denominator)
                unnormalized = 0.0 if log_numerator == -math.inf else math.exp(log_z + log_numerator)
                if not math.isfinite(snis) or not math.isfinite(unnormalized):
                    raise FloatingPointError(f"non-finite estimator reconstructed from {path}")
                row = {
                    **base,
                    "threshold": threshold,
                    "event_discoveries": int(event.sum()),
                    "reverse_snis_estimate": snis,
                    "unnormalized_estimate": unnormalized,
                    "log_reverse_denominator": log_denominator,
                }
                component_rows.append(row)
                grouped[(entry["method_key"], entry["particle_budget"], entry["cohort"], int(summary["run_index"]), threshold)].append(row)

    run_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        method, budget, cohort, source_run, threshold = key
        if sorted(float(row["lambda"]) for row in rows) != list(LAMBDAS):
            raise ValueError(f"incomplete lambda set for {key}")
        run_rows.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "cohort": cohort,
                "source_run_index": source_run,
                "threshold": threshold,
                "reverse_snis_estimate": statistics.mean(float(row["reverse_snis_estimate"]) for row in rows),
                "unnormalized_estimate": statistics.mean(float(row["unnormalized_estimate"]) for row in rows),
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
                "decoder_particle_steps": sum(int(row["decoder_particle_steps"]) for row in rows),
            }
        )
    return component_rows, run_rows, normalizer_rows, increment_rows


def validate_against_released(
    reconstructed: list[dict[str, Any]], released: list[dict[str, str]]
) -> None:
    expected = {
        (
            row["method_key"], int(row["particle_budget"]), row["cohort"],
            int(row["source_run_index"]), float(row["threshold"]),
        ): float(row["estimate"])
        for row in released
    }
    if len(expected) != len(reconstructed):
        raise ValueError("released and reconstructed run counts differ")
    for row in reconstructed:
        key = (
            row["method_key"], int(row["particle_budget"]), row["cohort"],
            int(row["source_run_index"]), float(row["threshold"]),
        )
        if not math.isclose(float(row["reverse_snis_estimate"]), expected[key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"reverse-SNIS reconstruction mismatch for {key}")


def estimator_summaries(
    run_rows: list[dict[str, Any]], direct_reference: dict[float, float]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["method_key"], int(row["particle_budget"]), float(row["threshold"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, budget, threshold), rows in sorted(grouped.items()):
        reference = direct_reference[threshold]
        for estimator in ("reverse_snis_estimate", "unnormalized_estimate"):
            values = np.asarray([float(row[estimator]) for row in rows])
            errors = values - reference
            rmse = math.sqrt(float(np.mean(errors * errors)))
            output.append(
                {
                    "method_key": method,
                    "particle_budget": budget,
                    "threshold": threshold,
                    "estimator": "reverse_snis" if estimator.startswith("reverse") else "unnormalized",
                    "independent_runs": len(values),
                    "direct_reference": reference,
                    "mean_estimate": float(values.mean()),
                    "bias": float(errors.mean()),
                    "sd": float(values.std(ddof=1)),
                    "rmse": rmse,
                    "relative_rmse": rmse / reference,
                    "zero_estimate_fraction": float(np.mean(values == 0.0)),
                    "median_runtime_seconds": statistics.median(float(row["runtime_seconds"]) for row in rows),
                    "median_decoder_particle_steps": statistics.median(int(row["decoder_particle_steps"]) for row in rows),
                }
            )
    return output


def rmse_backing(
    released: list[dict[str, str]], direct_reference: dict[float, float], seconds_per_direct: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in released:
        grouped[(row["method_key"], int(row["particle_budget"]), float(row["threshold"]))].append(row)
    output: list[dict[str, Any]] = []
    for group_index, ((method, budget, threshold), rows) in enumerate(sorted(grouped.items())):
        reference = direct_reference[threshold]
        values = np.asarray([float(row["estimate"]) for row in rows])
        rmse = math.sqrt(float(np.mean((values - reference) ** 2)))
        ci_low, ci_high = bootstrap_rmse_ci(values, reference, seed=20260823 + group_index)
        runtime = statistics.median(float(row["runtime_seconds"]) for row in rows)
        steps = statistics.median(float(row["decoder_particle_steps"]) for row in rows)
        output.append(
            {
                "series": "smc",
                "method_key": method,
                "particle_budget": budget,
                "threshold": threshold,
                "reference": reference,
                "independent_runs": len(values),
                "rmse": rmse,
                "relative_rmse": rmse / reference,
                "relative_rmse_ci_low": ci_low / reference,
                "relative_rmse_ci_high": ci_high / reference,
                "median_runtime_seconds": runtime,
                "median_decoder_particle_steps": steps,
                "matched_direct_samples_by_steps": max(1, int(steps // 32)),
                "matched_direct_samples_by_time": max(1, int(runtime // seconds_per_direct)),
            }
        )
        for criterion, sample_count in (
            ("decoder_steps", max(1, int(steps // 32))),
            ("wall_time", max(1, int(runtime // seconds_per_direct))),
        ):
            direct_rmse = math.sqrt(reference * (1.0 - reference) / sample_count)
            output.append(
                {
                    "series": f"direct_matched_{criterion}",
                    "method_key": method,
                    "particle_budget": budget,
                    "threshold": threshold,
                    "reference": reference,
                    "independent_runs": "analytic_binomial",
                    "rmse": direct_rmse,
                    "relative_rmse": direct_rmse / reference,
                    "relative_rmse_ci_low": "",
                    "relative_rmse_ci_high": "",
                    "median_runtime_seconds": sample_count * seconds_per_direct,
                    "median_decoder_particle_steps": sample_count * 32,
                    "matched_direct_samples_by_steps": sample_count if criterion == "decoder_steps" else "",
                    "matched_direct_samples_by_time": sample_count if criterion == "wall_time" else "",
                }
            )
    return output


def tilt_moments(rewards: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for lam in LAMBDAS:
        log_m_plus = logmeanexp(lam * rewards)
        log_m_minus = logmeanexp(-lam * rewards)
        rows.append(
            {
                "lambda": lam,
                "log_M_plus": log_m_plus,
                "log_M_minus": log_m_minus,
                "M_plus": math.exp(log_m_plus),
                "M_minus": math.exp(log_m_minus),
                "ideal_reverse_ess_over_n": math.exp(-log_m_plus - log_m_minus),
            }
        )
    return rows


def tail_curve(scores: np.ndarray) -> list[dict[str, float | int]]:
    ordered = np.sort(scores)
    unique, first = np.unique(ordered, return_index=True)
    n = scores.size
    return [
        {"toxicity_threshold": float(score), "tail_count": int(n - index), "tail_probability": float((n - index) / n)}
        for score, index in zip(unique, first)
    ]


def make_figures(
    output: Path,
    released: list[dict[str, str]],
    rmse_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    ideal_rows: list[dict[str, float]],
    tail_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir()
    configure_style()

    # Figure 1: relative RMSE versus rarity, with decoder-step-matched direct MC.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for ax, budget in zip(axes, (192, 512)):
        for method in METHOD_LABEL:
            rows = sorted(
                [r for r in rmse_rows if r["series"] == "smc" and r["method_key"] == method and r["particle_budget"] == budget],
                key=lambda r: r["threshold"],
            )
            if not rows:
                continue
            x = np.arange(len(rows))
            y = np.asarray([r["relative_rmse"] for r in rows])
            low = np.asarray([r["relative_rmse_ci_low"] for r in rows])
            high = np.asarray([r["relative_rmse_ci_high"] for r in rows])
            ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker=MARKERS[budget], capsize=3, color=COLORS[method], label=METHOD_LABEL[method])
            direct = sorted(
                [r for r in rmse_rows if r["series"] == "direct_matched_decoder_steps" and r["method_key"] == method and r["particle_budget"] == budget],
                key=lambda r: r["threshold"],
            )
            ax.plot(x, [r["relative_rmse"] for r in direct], linestyle="--", marker=MARKERS[budget], alpha=0.65, color=COLORS[method], label=f"Direct matched to {METHOD_LABEL[method]}")
        ax.set_yscale("log")
        ax.set_xticks(range(4), ["0.50", "0.90", "0.99", "0.999*"])
        ax.set_xlabel(r"Toxicity threshold $a$")
        ax.set_title(f"N={budget}")
    axes[0].set_ylabel(r"Relative RMSE: RMSE / $p_{\rm ref}$")
    axes[1].legend(fontsize=7.3, loc="upper left")
    fig.suptitle("Relative estimation error versus event rarity", y=1.01)
    save_figure(fig, figures, "fig1_relative_rmse_vs_rarity")

    # Figure 2: accuracy versus wall time and decoder work at primary/exploratory tails.
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.0), gridspec_kw={"hspace": 0.35, "wspace": 0.28})
    for row_index, threshold in enumerate((0.99, 0.999)):
        reference = next(float(r["reference"]) for r in rmse_rows if r["threshold"] == threshold)
        max_steps = max(float(r["median_decoder_particle_steps"]) for r in rmse_rows if r["series"] == "smc")
        direct_n = np.geomspace(64, max_steps / 32 * 1.2, 160)
        direct_rel = np.sqrt(reference * (1 - reference) / direct_n) / reference
        seconds_per_direct = next(
            float(r["median_runtime_seconds"]) / float(r["median_decoder_particle_steps"]) * 32
            for r in rmse_rows if r["series"] == "direct_matched_wall_time"
        )
        for col, (metric, xlabel) in enumerate(
            (("median_runtime_seconds", "Measured wall time (s)"), ("median_decoder_particle_steps", "Decoder particle-steps"))
        ):
            ax = axes[row_index, col]
            direct_x = direct_n * seconds_per_direct if col == 0 else direct_n * 32
            ax.plot(direct_x, direct_rel, color="#333333", linestyle="--", linewidth=1.4, label="Direct MC")
            for method in METHOD_LABEL:
                rows = [r for r in rmse_rows if r["series"] == "smc" and r["method_key"] == method and r["threshold"] == threshold]
                for row in rows:
                    ax.scatter(float(row[metric]), float(row["relative_rmse"]), s=48, marker=MARKERS[int(row["particle_budget"])], color=COLORS[method], label=f"{METHOD_LABEL[method]}, N={row['particle_budget']}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Relative RMSE")
            ax.set_title(rf"$a={threshold}$" + (" (exploratory)" if threshold == 0.999 else ""))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0, 0].legend(unique.values(), unique.keys(), fontsize=7.2, loc="best")
    fig.suptitle("Accuracy versus measured and decoder compute", y=0.995)
    save_figure(fig, figures, "fig2_accuracy_vs_compute")

    # Figure 3: run-level distributions.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.27})
    order = [("fk_bootstrap", 192), ("trajectory_resample_move", 192), ("fk_bootstrap", 512), ("trajectory_resample_move", 512)]
    for ax, threshold in zip(axes, (0.99, 0.999)):
        groups = [
            np.asarray([float(r["estimate"]) for r in released if r["method_key"] == method and int(r["particle_budget"]) == budget and float(r["threshold"]) == threshold])
            for method, budget in order
        ]
        positions = np.arange(1, 5)
        parts = ax.violinplot(groups, positions=positions, widths=0.72, showextrema=False)
        for body, (method, _) in zip(parts["bodies"], order):
            body.set_facecolor(COLORS[method]); body.set_alpha(0.20); body.set_edgecolor("none")
        boxes = ax.boxplot(groups, positions=positions, widths=0.32, patch_artist=True, showfliers=False)
        for box, (method, _) in zip(boxes["boxes"], order):
            box.set(facecolor="white", edgecolor=COLORS[method], linewidth=1.3)
        reference = float(next(r["direct_reference"] for r in released if float(r["threshold"]) == threshold))
        ax.axhline(reference, color="#222222", linestyle="--", linewidth=1.2, label="Direct 200k")
        ax.set_yscale("log")
        ax.set_xticks(positions, [f"{'FK' if m == 'fk_bootstrap' else 'Trajectory'}\nN={n}" for m, n in order])
        ax.set_ylabel(r"Estimate $\widehat p$")
        ax.set_title(rf"$a={threshold}$" + (" (exploratory)" if threshold == 0.999 else ""))
        ax.legend(loc="upper left")
    fig.suptitle("Run-level rare-event estimates", y=1.01)
    save_figure(fig, figures, "fig3_run_estimate_distributions")

    # Figure 4: component estimates and reverse-correction ESS, N=512.
    ideal = {float(r["lambda"]): float(r["ideal_reverse_ess_over_n"]) for r in ideal_rows}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), gridspec_kw={"wspace": 0.28})
    for method in METHOD_LABEL:
        means, q1, medians, q3, ess_q1, ess_med, ess_q3 = [], [], [], [], [], [], []
        for lam in LAMBDAS:
            selected = [r for r in component_rows if r["method_key"] == method and r["particle_budget"] == 512 and r["lambda"] == lam and r["threshold"] == 0.99]
            estimates = [float(r["reverse_snis_estimate"]) for r in selected]
            ess = [float(r["reverse_ess_over_n"]) for r in selected]
            a, b, c = quantiles(estimates); q1.append(a); medians.append(b); q3.append(c); means.append(statistics.mean(estimates))
            a, b, c = quantiles(ess); ess_q1.append(a); ess_med.append(b); ess_q3.append(c)
        axes[0].fill_between(LAMBDAS, q1, q3, color=COLORS[method], alpha=0.14)
        axes[0].plot(LAMBDAS, means, marker="o", color=COLORS[method], label=METHOD_LABEL[method])
        axes[1].fill_between(LAMBDAS, ess_q1, ess_q3, color=COLORS[method], alpha=0.14)
        axes[1].plot(LAMBDAS, ess_med, marker="o", color=COLORS[method], label=METHOD_LABEL[method])
    reference = float(next(r["direct_reference"] for r in released if float(r["threshold"]) == 0.99))
    axes[0].axhline(reference, color="#222222", linestyle="--", label="Direct 200k")
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"Component estimate $\widehat p_\lambda$")
    axes[0].set_title(r"A. Component estimates at $a=0.99$")
    axes[1].plot(LAMBDAS, [ideal[x] for x in LAMBDAS], color="#222222", linestyle="--", marker="D", markersize=4, label=r"Ideal $1/[M(\lambda)M(-\lambda)]$")
    axes[1].set_ylim(-0.02, 1.04); axes[1].set_ylabel(r"Reverse-correction ESS / $N$")
    axes[1].set_title("B. Empirical and ideal overlap")
    for ax in axes:
        ax.set_xticks(LAMBDAS); ax.set_xlabel(r"Tilting parameter $\lambda$"); ax.legend(fontsize=7.7)
    fig.suptitle("Tilt strength, component stability, and overlap", y=1.01)
    save_figure(fig, figures, "fig4_tilt_overlap_diagnostic")

    # Figure 5: native toxicity survival curve.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    plot_rows = tail_rows[:: max(1, len(tail_rows) // 6000)]
    ax.step([r["toxicity_threshold"] for r in plot_rows], [r["tail_probability"] for r in plot_rows], where="post", color="#0072B2", linewidth=1.5)
    threshold_styles = {
        0.5: ("#999999", ":"),
        0.9: ("#777777", "--"),
        0.99: ("#555555", "-."),
        0.999: ("#333333", (0, (4, 1, 1, 1))),
    }
    for threshold in THRESHOLDS:
        probability = float(next(r["tail_probability"] for r in tail_rows if float(r["toxicity_threshold"]) >= threshold))
        color, linestyle = threshold_styles[threshold]
        ax.axvline(
            threshold,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            label=rf"$a={threshold}$: $p_{{\rm ref}}={probability:.2g}$",
        )
        ax.scatter([threshold], [probability], color=color, s=20, zorder=3)
    ax.set_yscale("log"); ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Toxicity score threshold $a$"); ax.set_ylabel(r"$P_{P_0}(S_{\rm tox}\geq a)$")
    ax.set_title("Native MDLM toxicity tail from 200,000 IID samples")
    ax.legend(loc="lower left", title="Paper thresholds", fontsize=7.5, title_fontsize=8)
    save_figure(fig, figures, "fig5_native_toxicity_tail")

    # Figure 6: optional exact reconstruction of the unnormalized estimator.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, budget in zip(axes, (192, 512)):
        for method in METHOD_LABEL:
            snis = sorted([r for r in estimator_rows if r["method_key"] == method and r["particle_budget"] == budget and r["estimator"] == "reverse_snis"], key=lambda r: r["threshold"])
            un = sorted([r for r in estimator_rows if r["method_key"] == method and r["particle_budget"] == budget and r["estimator"] == "unnormalized"], key=lambda r: r["threshold"])
            if not snis:
                continue
            x = np.arange(4)
            ax.plot(x, [r["relative_rmse"] for r in snis], marker="o", color=COLORS[method], label=f"{METHOD_LABEL[method]} SNIS")
            ax.plot(x, [r["relative_rmse"] for r in un], marker="x", linestyle="--", color=COLORS[method], label=f"{METHOD_LABEL[method]} unnormalized")
        ax.set_yscale("log"); ax.set_xticks(range(4), ["0.50", "0.90", "0.99", "0.999*"])
        ax.set_xlabel(r"Toxicity threshold $a$"); ax.set_title(f"N={budget}")
    axes[0].set_ylabel("Relative RMSE")
    axes[1].legend(fontsize=7.1, loc="upper left")
    fig.suptitle("Reverse-SNIS versus stored-normalizer estimator", y=1.01)
    save_figure(fig, figures, "fig6_unnormalized_vs_reverse_snis")


def build_report(
    output: Path,
    report_dir: Path,
    direct_summary: dict[str, Any],
    entries: list[dict[str, Any]],
    rmse_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    increments_available: bool,
) -> None:
    lines = [
        "# Offline publication-figure analysis",
        "",
        "## Artifact paths",
        "",
        f"- Frozen final report: `{report_dir}`",
        f"- Direct 200k evaluation: decoder `{direct_summary['decoder_id']}`, evaluator `{direct_summary['evaluator_id']}`",
    ]
    for entry in entries:
        lines.append(f"- {METHOD_LABEL[entry['method_key']]} N={entry['particle_budget']} ({entry['cohort']}): `{entry['run_directory']}`")
    lines.extend(
        [
            "",
            "## Main numerical findings",
            "",
            "Relative RMSE values use the 200k empirical probability as the reference. Direct matched-compute values use the binomial RMSE at the number of IID samples affordable under each measured budget.",
            "",
            "| threshold | method | N | SMC relative RMSE | direct matched decoder work | direct matched wall time |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    smc = [r for r in rmse_rows if r["series"] == "smc"]
    for row in smc:
        step = next(r for r in rmse_rows if r["series"] == "direct_matched_decoder_steps" and r["method_key"] == row["method_key"] and r["particle_budget"] == row["particle_budget"] and r["threshold"] == row["threshold"])
        time = next(r for r in rmse_rows if r["series"] == "direct_matched_wall_time" and r["method_key"] == row["method_key"] and r["particle_budget"] == row["particle_budget"] and r["threshold"] == row["threshold"])
        lines.append(f"| {row['threshold']} | {METHOD_LABEL[row['method_key']]} | {row['particle_budget']} | {row['relative_rmse']:.4g} | {step['relative_rmse']:.4g} | {time['relative_rmse']:.4g} |")

    direct_step_wins = 0
    direct_time_wins = 0
    for row in smc:
        step = next(r for r in rmse_rows if r["series"] == "direct_matched_decoder_steps" and r["method_key"] == row["method_key"] and r["particle_budget"] == row["particle_budget"] and r["threshold"] == row["threshold"])
        time = next(r for r in rmse_rows if r["series"] == "direct_matched_wall_time" and r["method_key"] == row["method_key"] and r["particle_budget"] == row["particle_budget"] and r["threshold"] == row["threshold"])
        direct_step_wins += int(step["relative_rmse"] < row["relative_rmse"])
        direct_time_wins += int(time["relative_rmse"] < row["relative_rmse"])
    lines.extend(
        [
            "",
            "## Matched-budget direct MC",
            "",
            f"Direct MC has lower predicted RMSE in {direct_step_wins}/{len(smc)} method/budget/threshold comparisons when matching decoder particle-steps, and in {direct_time_wins}/{len(smc)} comparisons when matching measured wall time. The complete, unfavorable as well as favorable, comparisons are retained in `csv/relative_rmse.csv`.",
            "",
            "## Unnormalized estimator",
            "",
            "The estimator was reconstructed exactly from stored terminal normalized SMC weights, particle rewards/events, and each component's stored `log_normalizer_estimate`; no normalizer was approximated offline.",
            "",
            f"Per-level normalizer increments available for FK/Trajectory components: **{increments_available}**. Only stored total normalizers are used.",
            "",
            "| threshold | method | N | reverse-SNIS RMSE | unnormalized RMSE | ratio U/SNIS |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for snis in [r for r in estimator_rows if r["estimator"] == "reverse_snis"]:
        un = next(r for r in estimator_rows if r["estimator"] == "unnormalized" and r["method_key"] == snis["method_key"] and r["particle_budget"] == snis["particle_budget"] and r["threshold"] == snis["threshold"])
        lines.append(f"| {snis['threshold']} | {METHOD_LABEL[snis['method_key']]} | {snis['particle_budget']} | {snis['rmse']:.5g} | {un['rmse']:.5g} | {un['rmse']/snis['rmse']:.4g} |")
    lines.extend(
        [
            "",
            "## Recommended figures for the main paper",
            "",
            "1. `fig1_relative_rmse_vs_rarity`: the central probability-estimation result, including the direct matched-work comparator.",
            "2. `fig2_accuracy_vs_compute`: makes the statistical/computational trade-off explicit and prevents a method-only accuracy claim.",
            "3. `fig4_tilt_overlap_diagnostic`: connects lambda choice, component stability, and the direct-score overlap benchmark.",
            "",
            "The run-distribution, native-tail, and unnormalized-estimator figures are better suited to the appendix unless space permits.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    if args.render_only:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"analysis output does not exist: {args.output_dir}")
        if (args.output_dir / "figures").exists():
            raise FileExistsError(f"refusing to overwrite {args.output_dir / 'figures'}")
        make_figures(args.output_dir, *read_render_backing(args.output_dir))
        manifest_path = args.output_dir / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["figures"] = 6
        manifest["figure_render_location"] = "local_from_csv_backing"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, sort_keys=True))
        return

    if args.artifact_root is None or args.report_dir is None:
        parser.error("--artifact-root and --report-dir are required unless --render-only is used")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()

    manifest = read_json(args.report_dir / "manifest.json")
    released = read_csv(args.report_dir / "run_estimates.csv")
    scores, rewards, direct_summary, score_files = direct_artifacts(args.artifact_root, manifest)
    references = {threshold: float(np.mean(scores >= threshold)) for threshold in THRESHOLDS}
    for row in read_csv(args.report_dir / "direct_reference.csv"):
        if not math.isclose(references[float(row["threshold"])], float(row["empirical_probability"]), abs_tol=1e-15):
            raise ValueError("direct reference does not match continuous score shards")

    entries = comparison_entries(args.artifact_root, manifest)
    component_rows, reconstructed, normalizer_rows, increment_rows = component_analysis(entries, manifest)
    increments_available = bool(increment_rows)
    validate_against_released(reconstructed, released)
    estimator_rows = estimator_summaries(reconstructed, references)
    rmse_rows = rmse_backing(released, references, float(direct_summary["seconds_per_evaluated_completion"]))
    ideal_rows = tilt_moments(rewards)
    tail_rows = tail_curve(scores)

    write_csv(csv_dir / "relative_rmse.csv", rmse_rows, list(rmse_rows[0]))
    write_csv(csv_dir / "run_estimates_released.csv", released, list(released[0]))
    write_csv(csv_dir / "component_estimators.csv", component_rows, list(component_rows[0]))
    write_csv(csv_dir / "run_estimators.csv", reconstructed, list(reconstructed[0]))
    write_csv(csv_dir / "estimator_comparison.csv", estimator_rows, list(estimator_rows[0]))
    write_csv(csv_dir / "normalizers.csv", normalizer_rows, list(normalizer_rows[0]))
    write_csv(
        csv_dir / "normalizer_increments.csv",
        increment_rows,
        ["method_key", "particle_budget", "cohort", "source_run_index", "lambda", "seed", "level", "log_normalizer_increment"],
    )
    write_csv(csv_dir / "tilt_ideal_overlap.csv", ideal_rows, list(ideal_rows[0]))
    write_csv(csv_dir / "native_toxicity_tail.csv", tail_rows, list(tail_rows[0]))
    write_csv(
        csv_dir / "direct_cost.csv",
        [{**direct_summary, "score_shards": len(score_files)}],
        list(direct_summary) + ["score_shards"],
    )
    write_csv(
        csv_dir / "artifact_paths.csv",
        [
            {"role": "final_report", "path": str(args.report_dir)},
            {"role": "direct_evaluation", "path": str(score_files[0].parent)},
            *[{"role": f"{row['method_key']}_n{row['particle_budget']}_{row['cohort']}", "path": str(row["run_directory"])} for row in entries],
        ],
        ["role", "path"],
    )

    if not args.skip_figures:
        make_figures(args.output_dir, released, rmse_rows, component_rows, ideal_rows, tail_rows, estimator_rows)
    build_report(args.output_dir, args.report_dir, direct_summary, entries, rmse_rows, estimator_rows, increments_available)
    summary = {
        "kind": "mdlm_offline_publication_figures_v1",
        "decoder_id": manifest["decoder_id"],
        "evaluator_id": manifest["evaluator_id"],
        "direct_samples": int(scores.size),
        "direct_score_shards": len(score_files),
        "component_files": len(normalizer_rows),
        "reconstructed_run_estimates": len(reconstructed),
        "normalizer_totals_available": True,
        "normalizer_increments_available": increments_available,
        "unnormalized_estimator_reconstructed": True,
        "figures": 0 if args.skip_figures else 6,
        "figure_render_location": "pending_local_render" if args.skip_figures else "cluster",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
