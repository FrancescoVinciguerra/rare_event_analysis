from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / "data" / "components"
FIGURES = REPOSITORY_ROOT / "reproduced_figures"
PAPER_FIGURES = FIGURES
with (REPOSITORY_ROOT / "data" / "final_analysis" / "direct_reference.csv").open(
    newline="", encoding="utf-8"
) as _handle:
    DIRECT_REFERENCE = next(
        float(row["empirical_probability"])
        for row in csv.DictReader(_handle)
        if float(row["threshold"]) == 0.99
    )
LAMBDAS = np.asarray([0.25, 0.50, 0.75, 1.00])
COLORS = {"FK": "#D55E00", "Trajectory SMC": "#0072B2"}

SOURCES = {
    "fk512_submission.json": ("FK", 512, "submission"),
    "fk512_confirmatory.json": ("FK", 512, "confirmatory"),
    "trajectory512_submission.json": ("Trajectory SMC", 512, "submission"),
    "trajectory512_confirmatory.json": ("Trajectory SMC", 512, "confirmatory"),
    "fk192_confirmatory.json": ("FK", 192, "confirmatory"),
    "trajectory192_submission.json": ("Trajectory SMC", 192, "submission"),
    "trajectory192_confirmatory.json": ("Trajectory SMC", 192, "confirmatory"),
}

FIELDS = [
    "method", "particle_budget", "cohort", "run_index", "seed", "lambda",
    "reverse_snis_estimate", "reverse_weight_ess_over_n",
    "tilted_event_discoveries", "tilted_event_fraction_not_p0_estimate",
    "distinct_root_ancestor_fraction", "normalized_root_ancestor_entropy",
    "largest_root_family_fraction", "distinct_terminal_sequence_fraction",
    "largest_terminal_duplicate_fraction", "resampling_count",
    "mcmc_acceptance_rate", "reward_mean", "reward_min", "reward_max",
]


def configure_style():
    mpl.rcParams.update({
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
    })


def load_rows():
    with (ROOT / "component_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_threshold_rows():
    rows = []
    with (ROOT / "component_estimates_all_thresholds.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            row["method"] = (
                "FK" if row["method_key"] == "fk_bootstrap" else "Trajectory SMC"
            )
            rows.append(row)
    return rows


def save_csv(rows):
    with (ROOT / "component_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def values(rows, method, budget, lam, field):
    return np.asarray([
        float(row[field]) for row in rows
        if row["method"] == method
        and int(row["particle_budget"]) == budget
        and float(row["lambda"]) == lam
        and row[field] is not None
    ])


def quartiles(array):
    return np.quantile(array, [0.25, 0.50, 0.75])


def add_median_iqr(ax, rows, field, method, ylabel, *, budget=512):
    stats = np.asarray([quartiles(values(rows, method, budget, lam, field)) for lam in LAMBDAS])
    ax.fill_between(LAMBDAS, stats[:, 0], stats[:, 2], color=COLORS[method], alpha=0.14, linewidth=0)
    ax.plot(LAMBDAS, stats[:, 1], marker="o", markersize=5, linewidth=1.6, color=COLORS[method], label=method)
    ax.set_ylabel(ylabel)


def save_figure(fig, stem):
    for output_dir in {FIGURES, PAPER_FIGURES}:
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def component_estimate_and_ess(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.65), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    offsets = {"FK": -0.015, "Trajectory SMC": 0.015}
    for method in ("FK", "Trajectory SMC"):
        means, q25, q75 = [], [], []
        for lam in LAMBDAS:
            array = np.asarray([
                float(row["estimate"])
                for row in rows
                if row["method"] == method
                and int(row["particle_budget"]) == 512
                and float(row["lambda"]) == lam
                and float(row["threshold"]) == 0.99
            ])
            lo, _, hi = quartiles(array)
            means.append(array.mean()); q25.append(lo); q75.append(hi)
        means = np.asarray(means); q25 = np.asarray(q25); q75 = np.asarray(q75)
        shifted = LAMBDAS + offsets[method]
        ax.vlines(shifted, q25, q75, color=COLORS[method], linewidth=3.0, alpha=0.30)
        ax.plot(shifted, means, marker="o", markersize=5, linewidth=1.5, color=COLORS[method], label=method)
    ax.axhline(DIRECT_REFERENCE, color="#222222", linestyle="--", linewidth=1.2, label="Direct reference (200k)")
    ax.set_yscale("log")
    ax.set_xticks(LAMBDAS)
    ax.set_xlabel(r"Tilting parameter $\lambda$")
    ax.set_ylabel(r"Component estimate $\widehat p_\lambda$")
    ax.set_title(r"A. SNIS component estimates at $a=0.99$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for method in ("FK", "Trajectory SMC"):
        selected = [row for row in rows if float(row["threshold"]) == 0.99]
        add_median_iqr(ax, selected, "reverse_ess_over_n", method, r"Correction ESS / $N$")
    ax.set_xticks(LAMBDAS)
    ax.set_xlabel(r"Tilting parameter $\lambda$")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("B. Stability of SNIS correction weights")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Component estimates and correction-weight ESS (N=512)", y=1.02, fontsize=11)
    save_figure(fig, "fig7_component_estimate_reverse_ess_n512")


def genealogy_and_terminal_diversity(rows):
    panels = [
        ("distinct_root_ancestor_fraction", "Distinct root fraction", "A. Surviving initial lineages", (0, 1.04)),
        ("distinct_terminal_sequence_fraction", "Distinct terminal sequence fraction", "B. Unique terminal token sequences", (0, 1.04)),
        ("largest_root_family_fraction", "Largest root-family fraction", "C. Dominant lineage", (0, 0.25)),
        ("largest_terminal_duplicate_fraction", "Largest terminal duplicate fraction", "D. Largest duplicate output group", (0, 0.25)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.5), gridspec_kw={"hspace": 0.40, "wspace": 0.30})
    for ax, (field, ylabel, title, ylim) in zip(axes.flat, panels):
        for method in ("FK", "Trajectory SMC"):
            add_median_iqr(ax, rows, field, method, ylabel)
        ax.set_xticks(LAMBDAS)
        ax.set_xlabel(r"Tilting parameter $\lambda$")
        ax.set_ylim(*ylim)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Genealogical diversity versus terminal diversity (N=512)", y=1.01, fontsize=11)
    save_figure(fig, "fig8_genealogy_terminal_diversity_n512")


def write_aggregate_summary(rows):
    metrics = [
        "reverse_snis_estimate", "reverse_weight_ess_over_n",
        "distinct_root_ancestor_fraction", "distinct_terminal_sequence_fraction",
        "largest_root_family_fraction", "largest_terminal_duplicate_fraction",
    ]
    fieldnames = ["method", "particle_budget", "lambda", "runs"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_q25", f"{metric}_q75"])
    output = []
    for method in ("FK", "Trajectory SMC"):
        for budget in (192, 512):
            for lam in LAMBDAS:
                record = {"method": method, "particle_budget": budget, "lambda": lam}
                count = 0
                for metric in metrics:
                    array = values(rows, method, budget, lam, metric)
                    count = len(array)
                    if count:
                        q25, median, q75 = quartiles(array)
                        record.update({
                            f"{metric}_mean": float(array.mean()),
                            f"{metric}_median": float(median),
                            f"{metric}_q25": float(q25),
                            f"{metric}_q75": float(q75),
                        })
                record["runs"] = count
                if count:
                    output.append(record)
    with (ROOT / "component_metrics_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def main():
    configure_style()
    rows = load_rows()
    threshold_rows = load_threshold_rows()
    component_estimate_and_ess(threshold_rows)
    genealogy_and_terminal_diversity(rows)
    print(f"Validated {len(rows)} released component rows")


if __name__ == "__main__":
    main()
