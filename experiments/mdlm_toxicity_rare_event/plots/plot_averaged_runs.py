"""Plot matched-budget comparisons from the saved analysis rows."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#4878b7", "#174275", "#e99540", "#b95d17")
METHODS = ("smc_pair", "smc_large", "crude_pair", "crude_large")


def plot(output, rows):
    cases = sorted({(r["threshold"], r["tilt"], r["small_n"]) for r in rows})
    fig, axes = plt.subplots(2, len(cases), figsize=(15, 7.4), sharey="col", squeeze=False)
    for col, (q, tilt, n) in enumerate(cases):
        group = [r for r in rows if (r["threshold"], r["tilt"], r["small_n"]) == (q, tilt, n)]
        series = [np.array([r[f"{m}_ratio"] for r in group]) for m in METHODS]
        labels = (f"SMC\n2×{n}", f"SMC\n{2*n}", "MC\nbudget coppia", "MC\nbudget singola")
        axes[0, col].set_title(f"q ≥ {str(q).replace('.', ',')}  ·  λ = {str(tilt).replace('.', ',')}")
        for row, ax in enumerate(axes[:, col]):
            ax.axhspan(.1, 10, color="#eaf2ec", zorder=0)
            ax.axhline(1, color="#465146", linestyle="--", lw=1, zorder=1)
            for pos, (values, color) in enumerate(zip(series, COLORS, strict=True), 1):
                shown = values if row == 0 else values[values > 0]
                if not len(shown):
                    continue
                ax.boxplot(shown, positions=[pos], widths=.55, patch_artist=True, manage_ticks=False,
                           showmeans=True, showfliers=True, boxprops={"facecolor": color, "alpha": .58},
                           medianprops={"color": "black", "linewidth": 1.5},
                           meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#222222",
                                      "markersize": 4},
                           flierprops={"marker": "o", "markersize": 3, "markerfacecolor": color})
            ax.set_yscale("symlog", linthresh=.01)
            ax.set_ylim(bottom=0)
            ax.set_xlim(.5, 4.5)
            ax.set_xticks(range(1, 5), [f"{label}\n{np.count_nonzero(v == 0)}/{len(v)} zeri"
                                       for label, v in zip(labels, series, strict=True)], fontsize=8)
            ax.grid(axis="y", alpha=.22)
    axes[0, 0].set_ylabel("Stima / riferimento\nTutte le stime")
    axes[1, 0].set_ylabel("Stima / riferimento\nSolo stime positive")
    fig.suptitle("Una run grande o due piccole a pari tempo GPU", fontsize=17)
    fig.text(.5, .02, "Diamante bianco = media. Linea nera = mediana. Box = 50% centrale. "
             "Fascia verde = entro un fattore 10 dal riferimento. I conteggi degli zeri usano tutte le stime.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=230, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.runs.open(newline="", encoding="utf-8") as handle:
        rows = [
            {
                "threshold": float(row["threshold"]),
                "tilt": float(row["tilt"]),
                "small_n": int(row["small_n"]),
                **{f"{method}_ratio": float(row[f"{method}_ratio"]) for method in METHODS},
            }
            for row in csv.DictReader(handle)
        ]
    if not rows:
        parser.error("--runs contains no matched comparisons")
    plot(args.output or args.runs.parent / "plots/paired_runs_matched_gpu.png", rows)


if __name__ == "__main__":
    main()
