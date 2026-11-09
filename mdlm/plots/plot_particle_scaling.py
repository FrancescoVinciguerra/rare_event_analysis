"""Render the preselected fixed-tilt comparisons from the completed Avon CSVs."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


SIZES = (128, 160, 192, 256)
EVENTS = {0.99: 0.5, 0.995: 0.5, 0.999: 2.5}
METHODS = (
    ("smc", "SMC non normalizzato", "#276a9d", -0.17),
    ("crude", "Crude MC a pari tempo", "#626972", 0.17),
)
ZERO_ERROR = math.log10(2)
REFERENCE_COLOR = "#ac392c"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_results(root):
    runs = read_csv(root / "runs.csv")
    summary = read_csv(root / "summary.csv")
    for row in runs:
        event, tilt = float(row["threshold"]), float(row["lambda"])
        if (row["primary_tilt"] == "True") != (tilt == EVENTS[event]):
            raise ValueError("A row is assigned to the wrong primary tilt")
    return group_results(runs, summary, tuple(EVENTS.items()))


def group_results(runs, summary, comparisons, *, sizes=SIZES):
    groups = {(event, tilt, n): [] for event, tilt in comparisons for n in sizes}
    lookup = {}
    for row in runs:
        key = (float(row["threshold"]), float(row["lambda"]), int(row["N"]))
        if key in groups:
            groups[key].append(row)
    for row in summary:
        group_key = (float(row["threshold"]), float(row["lambda"]), int(row["N"]))
        if group_key in groups:
            key = (*group_key, row["method"])
            if key in lookup:
                raise ValueError("Duplicate summary")
            lookup[key] = row

    for key, group in groups.items():
        if len(group) != 24 or {int(r["replicate"]) for r in group} != set(range(24)):
            raise ValueError("Expected 24 distinct replicates in every comparison")
        if len({r["seed"] for r in group}) != 24 or len({r["crude_pool_start"] for r in group}) != 24:
            raise ValueError("Repeated SMC seed or MC block within a comparison")
        for row in group:
            seconds, cost = float(row["smc_seconds"]), float(row["crude_calibration_seconds_per_path"])
            count = int(row["crude_n"])
            if count != math.floor(seconds / cost):
                raise ValueError("MC budget differs from the calibrated SMC time")
            if not math.isclose(float(row["crude"]), int(row["crude_hits"]) / count, rel_tol=1e-12):
                raise ValueError("Crude estimate differs from its saved hit count")
        for method, _, _, _ in METHODS:
            ratios = np.array([float(r[f"{method}_ratio"]) for r in group])
            logs = np.array([float(r[f"{method}_log_relative_error"]) for r in group])
            entry = lookup[(*key, method)]
            if not np.isfinite(ratios).all() or (ratios < 0).any():
                raise ValueError("Invalid estimate ratio")
            if not np.allclose(logs, np.log1p(np.abs(ratios - 1)) / math.log(10), rtol=1e-12, atol=1e-15):
                raise ValueError("Stored log-relative errors differ from the saved ratios")
            checks = {
                "relative_standard_deviation": ratios.std(ddof=1),
                "mean_ratio": ratios.mean(),
                "median_ratio": np.median(ratios),
                "zero_count": np.count_nonzero(ratios == 0),
                "within_factor_ten": np.count_nonzero((ratios >= 0.1) & (ratios <= 10)),
            }
            if any(not math.isclose(value, float(entry[key]), rel_tol=1e-12) for key, value in checks.items()):
                raise ValueError("Summary differs from the individual estimates")
    return groups, lookup


def box(ax, values, position, color, *, width=0.25):
    ax.boxplot(
        values, positions=[position], widths=width, whis=1.5, patch_artist=True,
        showmeans=True, showfliers=True, manage_ticks=False,
        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.42},
        whiskerprops={"color": color}, capprops={"color": color},
        medianprops={"color": "#15232e", "linewidth": 1.5},
        meanprops={"marker": "D", "markerfacecolor": "#d99118", "markeredgecolor": "#473307", "markersize": 4.2},
        flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": color, "markersize": 3.5},
    )


def boxplots(root, groups, summaries, metric, *, comparisons=tuple(EVENTS.items()),
             prefix="", estimated_tilts=()):
    ratio = metric == "ratio"
    zoom = 2.0 if ratio else 0.4
    reference = 1.0 if ratio else ZERO_ERROR
    title = "Rapporto stima / riferimento" if ratio else "Errore relativo logaritmico"
    ylabel = (r"$\widehat p_A/\widehat p_{\mathrm{ref}}$" if ratio else
              r"$\log_{10}(1+|\widehat p_A/\widehat p_{\mathrm{ref}}-1|)$")
    headings = ("Tutte le run · zoom", "Tutte le run · distribuzione completa", "Solo stime positive · distribuzione completa")
    height = 1.4 + 3 * len(comparisons)
    fig, axes = plt.subplots(len(comparisons), len(headings), figsize=(15.4, height), squeeze=False)
    values = {
        (event, tilt, n, method): np.array([(float(r[f"{method}_ratio"]), float(r[f"{method}_{metric}"]))
                                           for r in groups[event, tilt, n]])
        for event, tilt in comparisons for n in SIZES for method, _, _, _ in METHODS
    }
    for row_index, (event, tilt) in enumerate(comparisons):
        same_event = [pair for (a, _, _, _), pair in values.items() if a == event]
        maximum = max(reference, *(pair[:, 1].max() for pair in same_event))
        positives = np.concatenate([pair[pair[:, 0] > 0, 0] for pair in same_event])
        for col, ax in enumerate(axes[row_index]):
            xlabels = []
            for i, n in enumerate(SIZES):
                zeros = []
                for method, _, color, offset in METHODS:
                    pair = values[event, tilt, n, method]
                    zero_count = int((pair[:, 0] == 0).sum())
                    zeros.append(zero_count)
                    data = pair[:, 1] if col < 2 else pair[pair[:, 0] > 0, 1]
                    if data.size:
                        box(ax, data, i + offset, color)
                    if col == 0:
                        outside = int((data > zoom).sum())
                        if outside:
                            ax.text(i + offset, 1.018, f"↑{outside}", ha="center", va="bottom",
                                    transform=ax.get_xaxis_transform(), color=color, fontsize=8)
                xlabels.append(f"{n}\n{zeros[0]} / {zeros[1]}" if col == 2 else str(n))

            if col == 0:
                ax.set_ylim(-zoom * 0.035, zoom)
            elif ratio and col == 1:
                ax.set_yscale("symlog", linthresh=0.01)
                ax.set_ylim(-0.002, maximum * 1.38)
            elif ratio:
                ax.set_yscale("log")
                lower = min(0.1, float(positives.min()) * 0.65) if positives.size else 0.1
                ax.set_ylim(lower, maximum * 1.38)
            else:
                ax.set_ylim(-maximum * 0.035, maximum * 1.10)
            ax.axhline(reference, color=REFERENCE_COLOR, ls="--", lw=1, alpha=0.8)
            ax.set_xticks(range(len(SIZES)), xlabels, fontsize=9)
            ax.set_xlim(-0.57, len(SIZES) - 0.43)
            ax.grid(axis="y", color="#dce2e7", lw=0.65, alpha=0.8)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(headings[col], fontsize=11, weight="bold", pad=17)
            if col == 0:
                ref = float(summaries[event, tilt, SIZES[0], "smc"]["p_reference"])
                mark = "*" if tilt in estimated_tilts else ""
                ax.set_ylabel(f"q ≥ {event:g}   λ = {tilt:g}{mark}\n"
                              f"$\\widehat p_{{\\mathrm{{ref}}}}$ = {ref:.4g}",
                              fontsize=10, labelpad=10)

    subtitle = ("Stesse 24 run per taglia a tutti i tilt · crude MC abbinato al tempo SMC · * tempo stimato"
                if estimated_tilts else "24 repliche per taglia · crude MC abbinato al tempo SMC")
    fig.suptitle(title, x=0.085, y=1 - 0.22 / height, ha="left", fontsize=19, weight="bold")
    fig.text(0.085, 1 - 0.60 / height, subtitle, fontsize=11)
    fig.text(0.98, 1 - 0.44 / height, ylabel, ha="right", fontsize=15)
    handles = [Patch(facecolor=color, alpha=0.55, label=label) for _, label, color, _ in METHODS]
    handles.extend((
        Line2D([], [], color="#15232e", lw=1.5, label="Mediana"),
        Line2D([], [], color="#d99118", marker="D", ls="none", label="Media"),
        Line2D([], [], color=REFERENCE_COLOR, ls="--",
               label="Rapporto = 1" if ratio else "Errore relativo 100% · anche quando la stima è zero"),
    ))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.60 / height), ncol=5,
               frameon=False, fontsize=9.5)
    scale_note = " Rapporto completo in scala symlog con gli zeri, logaritmica per le sole stime positive." if ratio else ""
    fig.text(0.5, 0.375 / height, "Box 25°–75° percentile · baffi a 1,5 IQR · cerchi fuori dai baffi · ↑ numero oltre lo zoom."
             + scale_note, ha="center", fontsize=8.5)
    fig.text(0.5, 0.166 / height, "Asse orizzontale = particelle N. Sotto N nel pannello a destra: zeri su 24 per SMC / crude MC.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=1.56 / height, top=1 - 1.25 / height,
                        hspace=0.43, wspace=0.28)
    filename = "boxplots_ratio.png" if ratio else "boxplots_log_relative_error.png"
    figure_dir = root / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / (prefix + filename), dpi=210, facecolor="white")
    plt.close(fig)


def relative_sd(root, summaries, *, comparisons=tuple(EVENTS.items()),
                prefix="", estimated_tilts=()):
    columns = 3
    rows = math.ceil(len(comparisons) / columns)
    height = 4.2 + 3.05 * (rows - 1)
    fig, grid = plt.subplots(rows, columns, figsize=(13.5, height), sharey=True, squeeze=False)
    axes = grid.ravel()
    for ax, (event, tilt) in zip(axes, comparisons):
        for method, label, color, _ in METHODS:
            values = [float(summaries[event, tilt, n, method]["relative_standard_deviation"]) for n in SIZES]
            ax.plot(SIZES, values, color=color, label=label, marker="o", lw=1.8)
        ax.plot(SIZES, [float(summaries[event, tilt, n, "crude"]["crude_theory_rsd"]) for n in SIZES],
                color="#454c55", linestyle=":", lw=1.5, label="Crude MC teorica al riferimento")
        ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", lw=0.9, alpha=0.7)
        mark = "*" if tilt in estimated_tilts else ""
        ax.set_title(f"q ≥ {event:g}    λ = {tilt:g}{mark}", fontsize=12, weight="bold")
        ticks = [f"{n}\n{float(summaries[event, tilt, n, 'smc']['mean_smc_seconds']):.0f} s" for n in SIZES]
        ax.set_xticks(SIZES, ticks)
        ax.set_xlabel("Particelle N e tempo SMC medio")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", color="#dce2e7", lw=0.65)
        ax.set_axisbelow(True)
    for ax in axes[len(comparisons):]:
        ax.set_visible(False)
    axes[0].set_ylim(bottom=0)
    for ax in grid[:, 0]:
        ax.set_ylabel(r"$s(\widehat p_A)/\widehat p_{\mathrm{ref}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.23 / height), ncol=3, frameon=False)
    fig.suptitle("Deviazione standard relativa a pari tempo", x=0.07, y=1 - 0.125 / height, ha="left", fontsize=18, weight="bold")
    foot = ("Stesse 24 run per taglia a tutti i tilt, zeri inclusi. * Tempo SMC stimato dalle transizioni della run."
            if estimated_tilts else
            "Dispersione tra 24 stime per taglia, zeri inclusi. La curva teorica crude usa la probabilità di riferimento.")
    fig.text(0.5, 0.113 / height, foot,
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=1.26 / height, top=1 - 0.84 / height,
                        wspace=0.18, hspace=0.65)
    figure_dir = root / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / (prefix + "relative_std_by_population.png"), dpi=210, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    groups, summaries = load_results(args.results)
    boxplots(args.results, groups, summaries, "ratio")
    boxplots(args.results, groups, summaries, "log_relative_error")
    relative_sd(args.results, summaries)


if __name__ == "__main__":
    main()
