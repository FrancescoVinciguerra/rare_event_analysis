"""Compare the selected same-ladder estimators over their per-run compute budgets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_particle_scaling import (
    METHODS, REFERENCE_COLOR, SIZES, ZERO_ERROR, Line2D, Patch, PercentFormatter,
    box, group_results, plt, read_csv,
)


SELECTED = ((0.99, 0.5, "lambda0p5_"), (0.995, 1.0, "lambda1_"), (0.999, 2.5, ""))
COMPARISONS = tuple((event, tilt) for event, tilt, _ in SELECTED)
LOW_BUDGET_COMPARISONS = ((.99, .5), (.995, 1.), (.999, .5), (.999, 3.))


def load_selected(root, *, low_budget=False):
    runs, summaries = [], []
    comparisons, sizes = (LOW_BUDGET_COMPARISONS, (64, 96)) if low_budget else (COMPARISONS, SIZES)
    if low_budget:
        runs, summaries = read_csv(root / "runs.csv"), read_csv(root / "summary.csv")
    else:
        for event, tilt, stem in SELECTED:
            for filename, destination in (("runs.csv", runs), ("summary.csv", summaries)):
                destination.extend(row for row in read_csv(root / (stem + filename))
                                   if (float(row["threshold"]), float(row["lambda"])) == (event, tilt))
    groups, validated = group_results(runs, summaries, comparisons, sizes=sizes)
    for n in sizes:
        ordered = sorted(comparisons, key=lambda pair: pair[1])
        paired = [sorted(groups[event, tilt, n], key=lambda row: int(row["seed"])) for event, tilt in ordered]
        for checkpoints in zip(*paired, strict=True):
            if (len({(r["seed"], r["replicate"], r["crude_pool_start"]) for r in checkpoints}) != 1 or
                    any(int(a["crude_n"]) > int(b["crude_n"]) for a, b in zip(checkpoints, checkpoints[1:]))):
                raise ValueError("Checkpoints must share a seed and MC block with increasing budgets")
            for a, b in zip(checkpoints, checkpoints[1:]):
                if float(a["lambda"]) == float(b["lambda"]) and any(a[k] != b[k] for k in
                        ("crude_n", "smc_seconds", "smc_native_transitions")):
                    raise ValueError("Events on the same checkpoint must use the same budget")
    return groups, validated


def budget_axis(summaries, event, tilt, sizes, low_budget):
    entries = [summaries[event, tilt, n, "smc"] for n in sizes]
    seconds = np.array([float(r["mean_smc_seconds"]) for r in entries])
    positions = (np.array([float(r["mean_crude_n"]) * float(r["p_reference"]) for r in entries])
                 if low_budget else seconds)
    labels = [
        (f"{x:.1%} · {t:.0f} s" if low_budget else f"{t:.0f} s") + f"\n{n} / {float(r['mean_crude_n']):.0f}"
        for x, t, n, r in zip(positions, seconds, sizes, entries)
    ]
    spacing = float(np.diff(positions).min())
    return positions, labels, spacing


def event_label(event, tilt, low_budget):
    return f"q ≥ {event:g}   λ = {tilt:g}" + ("*" if not low_budget and tilt < 2.5 else "")


def format_axis(ax, positions, labels, spacing):
    ax.set_xticks(positions, labels, fontsize=8.5)
    ax.set_xlim(positions[0] - 0.56 * spacing, positions[-1] + 0.56 * spacing)
    ax.grid(axis="y", color="#dce2e7", lw=0.65)
    ax.set_axisbelow(True)


def error_boxes(root, groups, summaries, *, logarithmic, low_budget=False):
    reference = ZERO_ERROR if logarithmic else 1.0
    zoom = 0.4 if logarithmic else 1.5
    metric = "log_relative_error" if logarithmic else "relative_error"
    title = "Errore relativo logaritmico" if logarithmic else "Errore relativo assoluto"
    formula = (r"$\log_{10}(1+|\widehat p_A/\widehat p_{\mathrm{ref}}-1|)$" if logarithmic else
               r"$|\widehat p_A/\widehat p_{\mathrm{ref}}-1|$")
    headings = ("Tutte le run · zoom", "Tutte le run · distribuzione completa",
                "Solo stime positive · distribuzione completa")
    comparisons, sizes = (LOW_BUDGET_COMPARISONS, (64, 96)) if low_budget else (COMPARISONS, SIZES)
    height = 10.7 + 3.1 * (len(comparisons) - 3)
    fig, axes = plt.subplots(len(comparisons), 3, figsize=(15.4, height))
    for row_index, (event, tilt) in enumerate(comparisons):
        positions, labels, spacing = budget_axis(summaries, event, tilt, sizes, low_budget)
        values = {}
        for n in sizes:
            for method, _, _, _ in METHODS:
                rows = groups[event, tilt, n]
                ratios = np.array([float(r[f"{method}_ratio"]) for r in rows])
                errors = (np.array([float(r[f"{method}_log_relative_error"]) for r in rows])
                          if logarithmic else abs(ratios - 1))
                values[n, method] = (ratios, errors)
        maximum = max(zoom, *(errors.max() for _, errors in values.values()))
        for col, ax in enumerate(axes[row_index]):
            for i, n in enumerate(sizes):
                zeros = []
                for method, _, color, offset in METHODS:
                    ratios, errors = values[n, method]
                    zeros.append(int((ratios == 0).sum()))
                    data = errors[ratios > 0] if col == 2 else errors
                    position = positions[i] + offset * spacing
                    if data.size:
                        box(ax, data, position, color, width=spacing * 0.25)
                    if col == 0 and (outside := int((data > zoom).sum())):
                        ax.text(position, 1.01, f"↑{outside}", transform=ax.get_xaxis_transform(),
                                ha="center", va="bottom", color=color, fontsize=8)
                if col == 2:
                    ax.text(positions[i], 1.01, f"{zeros[0]} / {zeros[1]}",
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
            ax.axhline(reference, color=REFERENCE_COLOR, ls="--", lw=1)
            upper = zoom if col == 0 else maximum * 1.09
            ax.set_ylim(-upper * 0.025, upper)
            if not logarithmic:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            format_axis(ax, positions, labels, spacing)
            if row_index == 0:
                ax.set_title(headings[col], fontsize=11, weight="bold", pad=20)
            if col == 0:
                ref = float(summaries[event, tilt, sizes[0], "smc"]["p_reference"])
                ax.set_ylabel(event_label(event, tilt, low_budget) + "\n" + rf"$\widehat p_{{\rm ref}}={ref:.4g}$",
                              fontsize=10, labelpad=9)
    fig.suptitle(title, x=0.082, y=0.979, ha="left", fontsize=19, weight="bold")
    fig.text(0.082, 0.945, "Stesse 24 ladder per taglia · crude MC abbinato al budget di ogni stima", fontsize=11)
    fig.text(0.975, 0.958, formula, ha="right", fontsize=15)
    handles = [Patch(facecolor=color, alpha=0.55, label=label) for _, label, color, _ in METHODS]
    handles.extend((
        Line2D([], [], color="#15232e", lw=1.5, label="Mediana"),
        Line2D([], [], color="#d99118", marker="D", ls="none", label="Media"),
        Line2D([], [], color=REFERENCE_COLOR, ls="--", label="Errore relativo 100% · anche per stima zero"),
    ))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.059), ncol=5,
               frameon=False, fontsize=9.5)
    fig.text(0.5, 0.038, "Box 25°–75° percentile · baffi a 1,5 IQR · cerchi oltre i baffi. ↑ oltre lo zoom. A destra, sopra i box: zeri SMC / crude.",
             ha="center", fontsize=8.5)
    axis_note = ("Ascissa: % del budget di un evento atteso · tempo medio misurato, N / traiettorie crude medie."
                 if low_budget else
                 "Ascissa: tempo medio in secondi, N / traiettorie crude medie. * Tempo intermedio stimato dalle transizioni.")
    fig.text(0.5, 0.017, axis_note,
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.084, right=0.98, bottom=0.155, top=0.875, hspace=0.48, wspace=0.27)
    figure_dir = root / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"selected_tilts_{metric}_by_budget.png", dpi=210, facecolor="white")
    plt.close(fig)


def dispersion(root, summaries, *, low_budget=False):
    comparisons, sizes = (LOW_BUDGET_COMPARISONS, (64, 96)) if low_budget else (COMPARISONS, SIZES)
    fig, axes = plt.subplots(1, len(comparisons), figsize=(4.5 * len(comparisons), 4.3), sharey=True)
    for ax, (event, tilt) in zip(axes, comparisons):
        positions, labels, spacing = budget_axis(summaries, event, tilt, sizes, low_budget)
        for method, label, color, _ in METHODS:
            rsd = [float(summaries[event, tilt, n, method]["relative_standard_deviation"]) for n in sizes]
            ax.plot(positions, rsd, color=color, label=label, marker="o", lw=1.8)
            if low_budget:
                for position, n in zip(positions, sizes):
                    s = summaries[event, tilt, n, method]
                    if int(s["zero_count"]) == int(s["runs"]):
                        ax.annotate("24 zeri", (position, 0), xytext=(0, 8 if method == "smc" else 18),
                                    textcoords="offset points", color=color, ha="center", fontsize=8)
        theory = [float(summaries[event, tilt, n, "crude"]["crude_theory_rsd"]) for n in sizes]
        ax.plot(positions, theory, color="#454c55", linestyle=":", lw=1.5, label="Crude MC teorica al riferimento")
        ax.axhline(1, color=REFERENCE_COLOR, ls="--", lw=0.9, alpha=0.7)
        ax.set_title(event_label(event, tilt, low_budget), fontsize=12, weight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        format_axis(ax, positions, labels, spacing)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(r"$s(\widehat p_A)/\widehat p_{\mathrm{ref}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False)
    fig.suptitle("Deviazione standard relativa al crescere del budget", x=0.075, y=0.97,
                 ha="left", fontsize=18, weight="bold")
    fig.text(0.5, 0.063, "24 stime per punto, zeri inclusi. La curva teorica crude usa la probabilità di riferimento.",
             ha="center", fontsize=8.5)
    axis_note = ("Ascissa: % del budget di un evento atteso · tempo medio misurato, N / traiettorie crude medie."
                 if low_budget else
                 "Ascissa: tempo medio, N / traiettorie crude medie. * Tempi intermedi stimati dalle transizioni.")
    fig.text(0.5, 0.030, axis_note,
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.31, top=0.80, wspace=0.19)
    figure_dir = root / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "selected_tilts_relative_std_by_budget.png", dpi=210, facecolor="white")
    plt.close(fig)


def write_summary(root, groups, summaries, *, low_budget=False):
    output = []
    comparisons, sizes = (LOW_BUDGET_COMPARISONS, (64, 96)) if low_budget else (COMPARISONS, SIZES)
    for event, tilt in comparisons:
        for n in sizes:
            rows = groups[event, tilt, n]
            for method, _, _, _ in METHODS:
                s = summaries[event, tilt, n, method]
                ratios = np.array([float(r[f"{method}_ratio"]) for r in rows])
                errors = abs(ratios - 1)
                output.append(dict(
                    threshold=event, **{"lambda": tilt}, N=n, method=method,
                    p_reference=s["p_reference"], runs=len(rows), zero_count=s["zero_count"],
                    mean_smc_seconds=s["mean_smc_seconds"], smc_seconds_measured=low_budget or tilt == 2.5,
                    mean_crude_n=s["mean_crude_n"],
                    fraction_one_expected_crude_hit=float(s["mean_crude_n"]) * float(s["p_reference"]),
                    mean_smc_native_transitions=float(np.mean([int(r["smc_native_transitions"]) for r in rows])),
                    mean_relative_error=float(errors.mean()), median_relative_error=float(np.median(errors)),
                    relative_rmse=float(np.sqrt(np.mean(errors**2))),
                    mean_log_relative_error=s["mean_log_relative_error"],
                    median_log_relative_error=s["median_log_relative_error"],
                    relative_standard_deviation=s["relative_standard_deviation"],
                    within_factor_ten=s["within_factor_ten"],
                ))
    with (root / "selected_tilts_by_budget.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--low-budget", action="store_true")
    args = parser.parse_args()
    root = args.results
    options = dict(low_budget=args.low_budget)
    groups, summaries = load_selected(root, **options)
    error_boxes(root, groups, summaries, logarithmic=False, **options)
    error_boxes(root, groups, summaries, logarithmic=True, **options)
    dispersion(root, summaries, **options)
    write_summary(root, groups, summaries, **options)


if __name__ == "__main__":
    main()
