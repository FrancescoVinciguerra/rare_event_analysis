from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / "data" / "final_analysis"
OUT = REPOSITORY_ROOT / "reproduced_figures"

METHOD_LABEL = {
    ("fk_bootstrap", 192): "FK, N=192",
    ("trajectory_resample_move", 192): "Trajectory SMC, N=192",
    ("fk_bootstrap", 512): "FK, N=512",
    ("trajectory_resample_move", 512): "Trajectory SMC, N=512",
}
ORDER = [
    ("fk_bootstrap", 192),
    ("trajectory_resample_move", 192),
    ("fk_bootstrap", 512),
    ("trajectory_resample_move", 512),
]
COLORS = {"fk_bootstrap": "#D55E00", "trajectory_resample_move": "#0072B2"}
THRESHOLDS = [0.5, 0.9, 0.99, 0.999]


def load_csv(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem: str):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style():
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


def qband(values):
    arr = np.asarray(values, dtype=float)
    return np.quantile(arr, [0.25, 0.5, 0.75])


def figure_estimator_distribution(run_rows, direct_rows):
    threshold = 0.99
    reference = next(
        float(row["empirical_probability"])
        for row in direct_rows
        if float(row["threshold"]) == threshold
    )
    groups = []
    for method, budget in ORDER:
        groups.append(
            np.array(
                [
                    float(r["estimate"])
                    for r in run_rows
                    if r["method_key"] == method
                    and int(r["particle_budget"]) == budget
                    and float(r["threshold"]) == threshold
                ]
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"wspace": 0.28})
    rng = np.random.default_rng(20260822)
    positions = np.arange(1, len(groups) + 1)
    labels = [METHOD_LABEL[key].replace(", ", "\n") for key in ORDER]

    ax = axes[0]
    parts = ax.violinplot(groups, positions=positions, widths=0.72, showextrema=False)
    for body, (method, _) in zip(parts["bodies"], ORDER):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("none")
        body.set_alpha(0.22)
    bp = ax.boxplot(groups, positions=positions, widths=0.30, patch_artist=True, showfliers=False)
    for box, (method, _) in zip(bp["boxes"], ORDER):
        box.set(facecolor="white", edgecolor=COLORS[method], linewidth=1.4)
    for med in bp["medians"]:
        med.set(color="#202020", linewidth=1.4)
    for whisker in bp["whiskers"]:
        whisker.set(color="#777777", linewidth=0.9)
    for cap in bp["caps"]:
        cap.set(color="#777777", linewidth=0.9)
    for pos, vals, (method, _) in zip(positions, groups, ORDER):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, s=10, alpha=0.42, color=COLORS[method], linewidth=0)
    ax.axhline(reference, color="#111111", linestyle="--", linewidth=1.2, label="Direct reference (200k)")
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(r"Rare-event estimate $\widehat p$")
    ax.set_title(r"A. Run-level estimates at $\tau=0.99$")
    ax.legend(loc="upper left")

    ax = axes[1]
    errors = [np.abs(vals - reference) for vals in groups]
    bp = ax.boxplot(errors, positions=positions, widths=0.56, patch_artist=True, showfliers=False)
    for box, (method, _) in zip(bp["boxes"], ORDER):
        box.set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.24, linewidth=1.3)
    for med in bp["medians"]:
        med.set(color="#111111", linewidth=1.5)
    for pos, vals, (method, _) in zip(positions, errors, ORDER):
        jitter = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(pos + jitter, vals, s=10, alpha=0.42, color=COLORS[method], linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(r"Absolute error $|\widehat p-p_{\rm ref}|$")
    ax.set_title("B. Error relative to the direct reference")
    save(fig, "fig1_estimator_distribution_tau099")


def figure_forest(bootstrap_rows):
    comparisons = [
        ("trajectory_n512_vs_fk_n512", "Same particle count\nTrajectory 512 / FK 512", -0.13),
        ("trajectory_n192_vs_fk_n512_runtime_matched", "Approximately cost matched\nTrajectory 192 / FK 512", 0.13),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(THRESHOLDS))
    markers = ["o", "s"]
    colors = ["#0072B2", "#009E73"]
    for idx, (key, label, offset) in enumerate(comparisons):
        rows = sorted(
            [r for r in bootstrap_rows if r["comparison"] == key],
            key=lambda r: float(r["threshold"]),
        )
        center = np.array([float(r["rmse_ratio"]) for r in rows])
        low = np.array([float(r["rmse_ratio_ci_low"]) for r in rows])
        high = np.array([float(r["rmse_ratio_ci_high"]) for r in rows])
        ax.errorbar(
            x + offset,
            center,
            yerr=np.vstack([center - low, high - center]),
            fmt=markers[idx],
            markersize=6,
            capsize=3,
            color=colors[idx],
            linewidth=1.4,
            label=label,
        )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    ax.fill_between([-0.5, 3.5], 0.08, 1.0, color="#009E73", alpha=0.055, zorder=-2)
    ax.set_yscale("log")
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.13, 2.35)
    ax.set_xticks(x, ["0.50", "0.90", "0.99", "0.999*"])
    ax.set_xlabel(r"Toxicity threshold $\tau$")
    ax.set_ylabel("RMSE ratio (Trajectory SMC / FK)")
    ax.set_title("Relative estimation error across rare-event thresholds")
    ax.legend(loc="upper left", ncol=2, fontsize=8.3)
    ax.text(3.43, 1.04, "equal error", ha="right", va="bottom", fontsize=8, color="#555555")
    ax.text(0.01, 0.02, "Below 1 favors Trajectory SMC. Bars: run-level bootstrap 95% CI.  *Exploratory tail.", transform=ax.transAxes, fontsize=8, color="#555555")
    save(fig, "fig2_rmse_ratio_forest")


def figure_cost_frontier(summary_rows):
    data = {
        (r["method_key"], int(r["particle_budget"]), float(r["threshold"])): r
        for r in summary_rows
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.7), sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.27})
    for ax, threshold in zip(axes.flat, THRESHOLDS):
        for method in ("fk_bootstrap", "trajectory_resample_move"):
            xs, ys = [], []
            for budget in (192, 512):
                row = data[(method, budget, threshold)]
                xs.append(float(row["median_runtime_seconds"]))
                ys.append(float(row["rmse"]))
            ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.5, color=COLORS[method], label="FK" if method == "fk_bootstrap" else "Trajectory SMC")
            for xx, yy, budget in zip(xs, ys, (192, 512)):
                ax.annotate(f"N={budget}", (xx, yy), xytext=(5, 4), textcoords="offset points", fontsize=7.7, color=COLORS[method])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(220, 2200)
        ax.set_xticks([250, 700, 1900], ["250", "700", "1900"])
        ax.set_title(rf"$\tau={threshold}$" + (" (exploratory)" if threshold == 0.999 else ""))
        ax.set_ylabel("RMSE")
    for ax in axes[-1, :]:
        ax.set_xlabel("Median wall time per independent run (s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Accuracy–cost frontier (lower left is better)", y=0.995, fontsize=11)
    save(fig, "fig3_accuracy_cost_frontier")


def grouped_metric(rows, metric):
    out = defaultdict(list)
    for r in rows:
        if int(r["particle_budget"]) != 512:
            continue
        value = r[metric]
        if value == "":
            continue
        out[(r["method_key"], float(r["lambda"]))].append(float(value))
    return out


def figure_genealogy(rows):
    metrics = [
        ("reverse_ess_over_n", r"Reverse ESS / $N$", "A. Weight diversity"),
        ("distinct_root_fraction", "Distinct root fraction", "B. Genealogical diversity"),
        ("largest_root_family_fraction", "Largest root-family fraction", "C. Genealogical concentration"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45), gridspec_kw={"wspace": 0.31})
    lambdas = np.array([0.25, 0.5, 0.75, 1.0])
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        grouped = grouped_metric(rows, metric)
        for method in ("fk_bootstrap", "trajectory_resample_move"):
            quant = np.array([qband(grouped[(method, lam)]) for lam in lambdas])
            ax.fill_between(lambdas, quant[:, 0], quant[:, 2], color=COLORS[method], alpha=0.14, linewidth=0)
            ax.plot(lambdas, quant[:, 1], marker="o", color=COLORS[method], linewidth=1.6, markersize=5, label="FK" if method == "fk_bootstrap" else "Trajectory SMC")
        ax.set_xticks(lambdas)
        ax.set_xlabel(r"Tilting parameter $\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.04 if metric != "largest_root_family_fraction" else 0.25)
    axes[0].legend(loc="lower left")
    fig.suptitle("Weight and genealogical degeneracy at N=512", y=1.02, fontsize=11)
    save(fig, "fig4_genealogy_vs_lambda_n512")


def figure_algorithm_diagnostics(rows):
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.45), gridspec_kw={"wspace": 0.32})
    lambdas = np.array([0.25, 0.5, 0.75, 1.0])
    diagnostics = [
        ("log_identity_residual", r"$|\log \widehat Z_\lambda + \log \widehat m_{-\lambda}|$", "A. Normalizer identity residual"),
        ("resampling_count", "Number of resampling steps", "B. Resampling intensity"),
    ]
    for ax, (metric, ylabel, title) in zip(axes[:2], diagnostics):
        grouped = grouped_metric(rows, metric)
        for method in ("fk_bootstrap", "trajectory_resample_move"):
            values = []
            lows = []
            highs = []
            for lam in lambdas:
                vals = np.abs(np.array(grouped[(method, lam)], dtype=float)) if metric == "log_identity_residual" else np.array(grouped[(method, lam)], dtype=float)
                q = np.quantile(vals, [0.25, 0.5, 0.75])
                lows.append(q[0]); values.append(q[1]); highs.append(q[2])
            values = np.maximum(values, 1e-16) if metric == "log_identity_residual" else np.asarray(values)
            lows = np.maximum(lows, 1e-16) if metric == "log_identity_residual" else np.asarray(lows)
            highs = np.maximum(highs, 1e-16) if metric == "log_identity_residual" else np.asarray(highs)
            ax.fill_between(lambdas, lows, highs, color=COLORS[method], alpha=0.14, linewidth=0)
            ax.plot(lambdas, values, marker="o", color=COLORS[method], linewidth=1.6, markersize=5, label="FK" if method == "fk_bootstrap" else "Trajectory SMC")
        if metric == "log_identity_residual":
            ax.set_yscale("log")
        ax.set_xticks(lambdas)
        ax.set_xlabel(r"Tilting parameter $\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    grouped = grouped_metric(rows, "mcmc_acceptance_rate")
    med, lo, hi = [], [], []
    for lam in lambdas:
        q = qband(grouped[("trajectory_resample_move", lam)])
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    axes[2].fill_between(lambdas, lo, hi, color=COLORS["trajectory_resample_move"], alpha=0.16, linewidth=0)
    axes[2].plot(lambdas, med, marker="o", color=COLORS["trajectory_resample_move"], linewidth=1.6, markersize=5)
    axes[2].set_xticks(lambdas)
    axes[2].set_ylim(0.58, 1.0)
    axes[2].set_xlabel(r"Tilting parameter $\lambda$")
    axes[2].set_ylabel("MH acceptance rate")
    axes[2].set_title("C. Trajectory mutation acceptance")
    axes[0].legend(loc="upper left")
    fig.suptitle("Algorithm diagnostics at N=512", y=1.02, fontsize=11)
    save(fig, "fig5_algorithm_diagnostics_n512")


def wilson_interval(successes: int, total: int, z: float = 1.96):
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def figure_validation_panels(run_rows, summary_rows, direct_rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), gridspec_kw={"hspace": 0.42, "wspace": 0.29})
    x = np.arange(len(THRESHOLDS))
    direct = {float(r["threshold"]): r for r in direct_rows}

    # Probability curve: means with run-level IQR, plus direct Wilson intervals.
    ax = axes[0, 0]
    offsets = [-0.15, -0.05, 0.05, 0.15]
    markers = ["o", "s", "o", "s"]
    for offset, marker, (method, budget) in zip(offsets, markers, ORDER):
        centers, lows, highs = [], [], []
        for threshold in THRESHOLDS:
            vals = np.asarray([
                float(r["estimate"]) for r in run_rows
                if r["method_key"] == method
                and int(r["particle_budget"]) == budget
                and float(r["threshold"]) == threshold
            ])
            q25, q75 = np.quantile(vals, [0.25, 0.75])
            centers.append(vals.mean()); lows.append(q25); highs.append(q75)
        centers = np.asarray(centers); lows = np.asarray(lows); highs = np.asarray(highs)
        alpha = 1.0 if budget == 512 else 0.52
        ax.vlines(x + offset, lows, highs, color=COLORS[method], alpha=alpha, linewidth=1.1)
        ax.scatter(
            x + offset, centers, marker=marker, s=24,
            color=COLORS[method], alpha=alpha,
            label=METHOD_LABEL[(method, budget)], zorder=3,
        )
    direct_mean = np.asarray([float(direct[t]["empirical_probability"]) for t in THRESHOLDS])
    direct_low = np.asarray([float(direct[t]["wilson_95_low"]) for t in THRESHOLDS])
    direct_high = np.asarray([float(direct[t]["wilson_95_high"]) for t in THRESHOLDS])
    ax.fill_between(x, direct_low, direct_high, color="#333333", alpha=0.12, linewidth=0)
    ax.plot(x, direct_mean, color="#222222", marker="D", markersize=4, linewidth=1.2, label="Direct 200k (Wilson 95%)")
    ax.set_yscale("log")
    ax.set_xticks(x, ["0.50", "0.90", "0.99", "0.999*"])
    ax.set_xlabel(r"Toxicity threshold $\tau$")
    ax.set_ylabel(r"Estimated probability $\widehat p$")
    ax.set_title("A. Probability curve across rarity levels")
    ax.legend(fontsize=7.1, ncol=2, loc="lower left")

    # Bias-variance decomposition at the primary threshold and N=512.
    ax = axes[0, 1]
    methods = ["fk_bootstrap", "trajectory_resample_move"]
    labels = ["FK", "Trajectory SMC"]
    bias_sq, variance, mse = [], [], []
    for method in methods:
        vals = np.asarray([
            float(r["estimate"]) for r in run_rows
            if r["method_key"] == method
            and int(r["particle_budget"]) == 512
            and float(r["threshold"]) == 0.99
        ])
        ref = float(direct[0.99]["empirical_probability"])
        b2 = float((vals.mean() - ref) ** 2)
        v = float(np.mean((vals - vals.mean()) ** 2))
        bias_sq.append(b2); variance.append(v); mse.append(b2 + v)
    positions = np.arange(2)
    ax.bar(positions, variance, color=[COLORS[m] for m in methods], alpha=0.72, label="Run-to-run variance")
    ax.bar(positions, bias_sq, bottom=variance, color=[COLORS[m] for m in methods], alpha=0.25, hatch="///", label="Squared bias")
    for pos, total, var in zip(positions, mse, variance):
        ax.text(pos, total * 0.82, f"{100 * var / total:.1f}%\nvariance", ha="center", va="top", fontsize=8, color="white")
    ax.set_ylim(0, max(mse) * 1.08)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xticks(positions, labels)
    ax.set_ylabel(r"MSE contribution at $\tau=0.99$")
    ax.set_title("B. Bias–variance decomposition (N=512)")
    ax.legend(fontsize=8, loc="lower left")

    # Independent cohort replication.
    ax = axes[1, 0]
    cohorts = ["submission", "confirmatory"]
    for method, marker in zip(methods, ["o", "s"]):
        values = []
        for cohort in cohorts:
            errors = np.asarray([
                float(r["error"]) for r in run_rows
                if r["method_key"] == method
                and int(r["particle_budget"]) == 512
                and float(r["threshold"]) == 0.99
                and r["cohort"] == cohort
            ])
            values.append(math.sqrt(float(np.mean(errors * errors))))
        ax.plot([0, 1], values, marker=marker, markersize=6, linewidth=1.5, color=COLORS[method], label="FK" if method == "fk_bootstrap" else "Trajectory SMC")
        for pos, value in enumerate(values):
            ax.annotate(f"{value:.2e}", (pos, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.6, color=COLORS[method])
    ax.set_xticks([0, 1], ["Submission\n32 runs", "Confirmatory\n32 runs"])
    ax.set_ylabel(r"RMSE at $\tau=0.99$")
    ax.set_title("C. Replication across independent cohorts")
    ax.legend(fontsize=8, loc="upper left")

    # Zero-estimate rate in the exploratory tail.
    ax = axes[1, 1]
    rates, lower, upper, counts = [], [], [], []
    for method in methods:
        rows = [
            r for r in run_rows
            if r["method_key"] == method
            and int(r["particle_budget"]) == 512
            and float(r["threshold"]) == 0.999
        ]
        successes = sum(r["zero_estimate"] == "True" for r in rows)
        lo, hi = wilson_interval(successes, len(rows))
        rates.append(successes / len(rows)); lower.append(lo); upper.append(hi); counts.append((successes, len(rows)))
    ax.bar(positions, rates, width=0.58, color=[COLORS[m] for m in methods], alpha=0.72)
    ax.errorbar(positions, rates, yerr=np.vstack([np.asarray(rates) - lower, np.asarray(upper) - rates]), fmt="none", ecolor="#222222", capsize=4, linewidth=1.1)
    for pos, value, (successes, total) in zip(positions, rates, counts):
        ax.text(pos, value + 0.055, f"{successes}/{total}\n({100*value:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Fraction of independent runs")
    ax.set_title(r"D. Zero estimates at $\tau=0.999$ (exploratory)")
    ax.text(0.02, 0.03, "Whiskers: Wilson 95% intervals", transform=ax.transAxes, fontsize=7.8, color="#555555")
    save(fig, "fig6_probability_bias_cohort_zero_rate")


def write_cost_breakdown(summary_rows):
    rows_by_key = {
        (r["method_key"], int(r["particle_budget"]), float(r["threshold"])): r
        for r in summary_rows
    }
    fields = [
        "method", "particle_budget", "independent_runs", "median_wall_time_seconds",
        "median_decoder_particle_steps", "median_classifier_evaluations",
        "peak_cuda_memory_gib", "rmse_tau_0p99",
    ]
    with (OUT / "table1_accuracy_cost.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, budget in ORDER:
            row = rows_by_key[(method, budget, 0.99)]
            writer.writerow({
                "method": METHOD_LABEL[(method, budget)].split(",")[0],
                "particle_budget": budget,
                "independent_runs": row["independent_runs"],
                "median_wall_time_seconds": f"{float(row['median_runtime_seconds']):.3f}",
                "median_decoder_particle_steps": f"{float(row['median_decoder_particle_steps']):.0f}",
                "median_classifier_evaluations": f"{float(row['median_classifier_texts_scored']):.0f}",
                "peak_cuda_memory_gib": f"{float(row['peak_cuda_memory_allocated_bytes']) / (1024**3):.3f}",
                "rmse_tau_0p99": f"{float(row['rmse']):.9g}",
            })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    run_rows = load_csv("run_estimates.csv")
    summary_rows = load_csv("summary.csv")
    bootstrap_rows = load_csv("bootstrap_comparisons.csv")
    genealogy_rows = load_csv("normalizer_and_genealogy.csv")
    direct_rows = load_csv("direct_reference.csv")
    figure_estimator_distribution(run_rows, direct_rows)
    figure_forest(bootstrap_rows)
    figure_cost_frontier(summary_rows)
    figure_genealogy(genealogy_rows)
    figure_algorithm_diagnostics(genealogy_rows)
    figure_validation_panels(run_rows, summary_rows, direct_rows)
    write_cost_breakdown(summary_rows)
    print(f"Wrote paper figures and cost table to {OUT}")


if __name__ == "__main__":
    main()
