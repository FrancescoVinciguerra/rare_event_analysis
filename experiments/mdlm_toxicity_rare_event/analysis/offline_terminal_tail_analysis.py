"""Inspect frozen IID and SMC terminal toxicity tails without model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_publication_figures import comparison_entries, direct_artifacts, read_json, write_csv


EXTREME_THRESHOLDS = (0.999, 0.9995, 0.9999, 0.99995, 0.99999)
QUANTILES = (0.99, 0.999, 0.9995, 0.9999)
LABELS = {
    "iid": "IID native (200k)",
    "fk_bootstrap": "FK Steering",
    "trajectory_resample_move": "Trajectory SMC",
}
COLORS = {
    "iid": "#222222",
    "fk_bootstrap": "#D55E00",
    "trajectory_resample_move": "#0072B2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def terminal_scores(entries: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int, str, int], np.ndarray],
    list[str],
]:
    by_group: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    by_run_parts: dict[tuple[str, int, str, int], list[np.ndarray]] = defaultdict(list)
    paths_used: list[str] = []
    for entry in entries:
        paths = sorted((entry["run_directory"] / "components").glob("run_*_tilted_*.json"))
        if len(paths) != 128:
            raise ValueError(f"expected 128 component files in {entry['run_directory']}, found {len(paths)}")
        for path in paths:
            payload = read_json(path)
            if payload["decoder_id"] != manifest["decoder_id"] or payload["evaluator_id"] != manifest["evaluator_id"]:
                raise ValueError(f"identity mismatch in {path}")
            summary = payload["summary"]
            particles = payload.get("particles")
            if not particles or len(particles) != int(summary["num_particles"]):
                raise ValueError(f"terminal particle mismatch in {path}")
            scores = np.asarray([float(row["event_toxic_probability"]) for row in particles], dtype=np.float64)
            if not np.isfinite(scores).all() or np.min(scores) < 0.0 or np.max(scores) > 1.0:
                raise ValueError(f"invalid toxicity score in {path}")
            group = (entry["method_key"], int(entry["particle_budget"]))
            run = (*group, entry["cohort"], int(summary["run_index"]))
            by_group[group].append(scores)
            by_run_parts[run].append(scores)
            paths_used.append(str(path))
    group_arrays = {key: np.concatenate(parts) for key, parts in by_group.items()}
    run_arrays: dict[tuple[str, int, str, int], np.ndarray] = {}
    for key, parts in by_run_parts.items():
        if len(parts) != 4:
            raise ValueError(f"expected four independent lambda components for {key}, found {len(parts)}")
        run_arrays[key] = np.concatenate(parts)
    return group_arrays, run_arrays, paths_used


def survival_grid() -> np.ndarray:
    broad = np.linspace(0.0, 0.99, 280)
    tail = 1.0 - np.geomspace(1e-7, 1e-2, 360)
    return np.unique(np.concatenate([broad, tail, np.asarray(EXTREME_THRESHOLDS)]))


def survival_rows(distributions: dict[tuple[str, int], np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = survival_grid()
    for (method, budget), scores in sorted(distributions.items()):
        ordered = np.sort(scores)
        counts = scores.size - np.searchsorted(ordered, grid, side="left")
        for threshold, count in zip(grid, counts):
            rows.append(
                {
                    "method_key": method,
                    "particle_budget": "" if budget == 0 else budget,
                    "threshold": float(threshold),
                    "count_ge": int(count),
                    "fraction_ge": float(count / scores.size),
                    "terminal_scores": int(scores.size),
                    "distribution_note": "native IID" if method == "iid" else "unweighted equal-lambda tilted terminal population",
                }
            )
    return rows


def run_summary_rows(run_arrays: dict[tuple[str, int, str, int], np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method, budget, cohort, run_index), scores in sorted(run_arrays.items()):
        row: dict[str, Any] = {
            "method_key": method,
            "particle_budget": budget,
            "cohort": cohort,
            "run_index": run_index,
            "terminal_scores": int(scores.size),
            "maximum": float(np.max(scores)),
        }
        for quantile in QUANTILES:
            row[f"q_{quantile}"] = float(np.quantile(scores, quantile))
        for threshold in EXTREME_THRESHOLDS:
            count = int(np.count_nonzero(scores >= threshold))
            row[f"count_ge_{threshold}"] = count
            row[f"fraction_ge_{threshold}"] = float(count / scores.size)
        rows.append(row)
    return rows


def group_summary_rows(
    distributions: dict[tuple[str, int], np.ndarray],
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method, budget), scores in sorted(distributions.items()):
        selected = [row for row in run_rows if row["method_key"] == method and int(row["particle_budget"]) == budget]
        row: dict[str, Any] = {
            "method_key": method,
            "particle_budget": "" if budget == 0 else budget,
            "terminal_scores": int(scores.size),
            "independent_runs": 1 if method == "iid" else len(selected),
            "unique_score_values": int(np.unique(scores).size),
            "maximum": float(np.max(scores)),
            "minimum_positive_distance_to_one": float(np.min(1.0 - scores[scores < 1.0])) if np.any(scores < 1.0) else 0.0,
        }
        for quantile in QUANTILES:
            row[f"pooled_q_{quantile}"] = float(np.quantile(scores, quantile))
        for threshold in EXTREME_THRESHOLDS:
            count = int(np.count_nonzero(scores >= threshold))
            row[f"count_ge_{threshold}"] = count
            row[f"fraction_ge_{threshold}"] = float(count / scores.size)
            if method == "iid":
                row[f"runs_with_any_ge_{threshold}"] = int(count > 0)
                row[f"run_fraction_any_ge_{threshold}"] = float(count > 0)
            else:
                run_hits = sum(int(int(run[f"count_ge_{threshold}"]) > 0) for run in selected)
                row[f"runs_with_any_ge_{threshold}"] = run_hits
                row[f"run_fraction_any_ge_{threshold}"] = float(run_hits / len(selected))
        rows.append(row)
    return rows


def render(output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    survival = read_csv(output / "csv" / "survival_curves.csv")
    run_rows = read_csv(output / "csv" / "run_high_score_summary.csv")
    figures = output / "figures"
    figures.mkdir()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"wspace": 0.27})
    groups = [("iid", 0), ("fk_bootstrap", 192), ("fk_bootstrap", 512), ("trajectory_resample_move", 192), ("trajectory_resample_move", 512)]
    for method, budget in groups:
        selected = [row for row in survival if row["method_key"] == method and (method == "iid" or int(row["particle_budget"]) == budget)]
        x = np.asarray([float(row["threshold"]) for row in selected])
        y = np.asarray([float(row["fraction_ge"]) for row in selected])
        positive = y > 0.0
        label = LABELS[method] if method == "iid" else f"{LABELS[method]}, N={budget}"
        style = "--" if method == "iid" else "-" if budget == 512 else ":"
        width = 1.8 if method == "iid" or budget == 512 else 1.3
        for ax in axes:
            ax.plot(x[positive], y[positive], linestyle=style, linewidth=width, color=COLORS[method], label=label)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel(r"Toxicity threshold $a$")
        ax.set_ylabel(r"Empirical survival $P(S_{\rm tox}\geq a)$")
    axes[0].set_xlim(0.0, 1.0); axes[0].set_title("A. Full score range")
    axes[1].set_xlim(0.99, 1.0); axes[1].set_title("B. Extreme-score tail")
    for threshold in EXTREME_THRESHOLDS:
        axes[1].axvline(threshold, color="#BBBBBB", linewidth=0.55, alpha=0.55)
    axes[0].legend(fontsize=7.4, loc="lower left")
    fig.suptitle("Terminal toxicity-score survival curves", y=1.01)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(figures / f"terminal_score_survival.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    order = [("fk_bootstrap", 192), ("trajectory_resample_move", 192), ("fk_bootstrap", 512), ("trajectory_resample_move", 512)]
    maxima = [np.asarray([float(row["maximum"]) for row in run_rows if row["method_key"] == method and int(row["particle_budget"]) == budget]) for method, budget in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    boxes = ax.boxplot(maxima, patch_artist=True, showfliers=True, widths=0.5)
    for box, (method, _) in zip(boxes["boxes"], order):
        box.set_facecolor(COLORS[method]); box.set_alpha(0.22); box.set_edgecolor(COLORS[method])
    ax.set_xticks(range(1, 5), [f"{'FK' if m == 'fk_bootstrap' else 'TSMC'}\nN={n}" for m, n in order])
    ax.set_ylabel("Maximum terminal toxicity score per run")
    ax.set_ylim(0.99, 1.0001)
    ax.set_title("Per-run reach into the extreme toxicity-score tail")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(figures / f"per_run_maximum_score.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        if not args.output_dir.is_dir() or (args.output_dir / "figures").exists():
            raise FileExistsError("render input missing or figures already exist")
        render(args.output_dir)
        manifest_path = args.output_dir / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["figures"] = 2
        manifest["figure_render_location"] = "local_from_csv_backing"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, sort_keys=True))
        return
    if args.artifact_root is None or args.report_dir is None:
        parser.error("--artifact-root and --report-dir are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()

    manifest = read_json(args.report_dir / "manifest.json")
    iid_scores, _, direct_summary, score_files = direct_artifacts(args.artifact_root, manifest)
    entries = comparison_entries(args.artifact_root, manifest)
    smc_groups, run_arrays, component_paths = terminal_scores(entries, manifest)
    distributions = {("iid", 0): iid_scores, **smc_groups}
    run_rows = run_summary_rows(run_arrays)
    group_rows = group_summary_rows(distributions, run_rows)
    curve_rows = survival_rows(distributions)

    write_csv(csv_dir / "survival_curves.csv", curve_rows, list(curve_rows[0]))
    write_csv(csv_dir / "run_high_score_summary.csv", run_rows, list(run_rows[0]))
    write_csv(csv_dir / "group_tail_summary.csv", group_rows, list(group_rows[0]))
    write_csv(
        csv_dir / "artifact_paths.csv",
        [
            {"role": "direct_score_directory", "path": str(score_files[0].parent)},
            *[{"role": f"component_{index:04d}", "path": path} for index, path in enumerate(component_paths)],
        ],
        ["role", "path"],
    )
    if not args.skip_figures:
        render(args.output_dir)
    summary = {
        "kind": "mdlm_terminal_toxicity_tail_analysis_v1",
        "decoder_id": manifest["decoder_id"],
        "evaluator_id": manifest["evaluator_id"],
        "direct_samples": int(iid_scores.size),
        "component_files": len(component_paths),
        "smc_runs": len(run_arrays),
        "extreme_thresholds": list(EXTREME_THRESHOLDS),
        "figures": 0 if args.skip_figures else 2,
        "figure_render_location": "pending_local_render" if args.skip_figures else "cluster",
        "generation_or_scoring_performed": False,
        "direct_runtime_metadata_seconds": float(direct_summary["combined_sampling_and_scoring_seconds"]),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
