"""Combine the saved SMC campaigns without pooling seeds or changing the estimator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from plot_particle_scaling import (
    METHODS, REFERENCE_COLOR, ZERO_ERROR, Line2D, Patch, PercentFormatter, box, group_results, plt, read_csv,
)
from plot_selected_tilts_by_budget import load_selected

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs
from diffusion_toxicity_smc_experiment.particle_scaling import comparison_rows, summarize, validated_pool
from diffusion_toxicity_smc_experiment.same_run_reuse import write_csv


SIZES = (64, 96, 128, 160, 192, 256)
ULTRA_LOW_SIZES = (32, 48)


def panels(ultra_low):
    sizes = ULTRA_LOW_SIZES + SIZES if ultra_low else SIZES
    return (
        ("q ≥ 0,99 · λ = 0,5", ((.99, .5, sizes),)),
        ("q ≥ 0,995 · λ = 1", ((.995, 1., sizes),)),
        ("q ≥ 0,999 · λ = 2,5", ((.999, 2.5, SIZES),)),
    )


def combine(previous, low, ultra_low=None):
    previous_groups, _ = load_selected(previous)
    low_groups, _ = load_selected(low, low_budget=True)
    config_path = Path(__file__).resolve().parents[1] / "configs/mdlm_owt_low_particle_budget.yaml"
    study, configs = study_configs(config_path)
    pool, _ = validated_pool(low.parent, study)
    rows, status = comparison_rows({**study, "checkpoint_tilts": {.99: [], .995: [], .999: [2.5]}},
                                   configs, low, pool)
    if not status["complete"]:
        raise ValueError("The low-budget checkpoint campaign is incomplete")
    for n in SIZES[:2]:
        low_groups[.999, 2.5, n] = [r for r in rows if r["N"] == n]
    panel_specs = panels(ultra_low)
    expected = {(event, tilt, n) for _, segments in panel_specs for event, tilt, sizes in segments for n in sizes}
    campaigns = [(low_groups, low, 3.), (previous_groups, previous, 2.5)]
    if ultra_low:
        ultra_groups, _ = group_results(read_csv(ultra_low / "runs.csv"), read_csv(ultra_low / "summary.csv"),
                                        ((.99, .5), (.995, 1.)), sizes=ULTRA_LOW_SIZES)
        campaigns.insert(0, (ultra_groups, ultra_low, 1.5))
    combined = {}
    for groups, source, final_tilt in campaigns:
        for (event, tilt, n), rows in groups.items():
            key = (event, tilt, n)
            if key not in expected:
                continue
            if key in combined or len({r["seed"] for r in rows}) != 24:
                raise ValueError("Expected one cohort of 24 distinct seeds per comparison")
            combined[key] = [
                dict(
                    campaign=source.name, N=n, seed=int(r["seed"]), replicate=int(r["replicate"]),
                    **{"lambda": tilt}, threshold=event, p_reference=float(r["p_reference"]),
                    primary_tilt=True, ladder_final_tilt=final_tilt,
                    checkpoint_is_terminal=tilt == final_tilt,
                    smc_seconds_measured=source != previous or tilt == final_tilt,
                    smc_native_transitions=int(r["smc_native_transitions"]),
                    crude_pool_start=int(r["crude_pool_start"]), crude_n=int(r["crude_n"]),
                    smc_seconds=float(r["smc_seconds"]), event_particles=int(r["event_particles"]),
                    crude_hits=int(r["crude_hits"]),
                    **{field: float(r[field]) for method, _, _, _ in METHODS for field in (
                        method, f"log_{method}", f"{method}_ratio", f"{method}_log_relative_error",
                    )},
                ) for r in rows
            ]
    if combined.keys() != expected or any(len(rows) != 24 for rows in combined.values()):
        raise ValueError("Missing comparison group")
    for n in {key[2] for key in expected}:
        cohorts = [cohort for key, cohort in combined.items() if key[2] == n]
        identities = [{(r["seed"], r["replicate"], r["crude_pool_start"]) for r in cohort} for cohort in cohorts]
        if any(ids != identities[0] for ids in identities[1:]):
            raise ValueError("Checkpoints must come from the same ladder for each N")
    rows = [r for cohort in combined.values() for r in cohort]
    table = summarize(rows)
    for s in table:
        group = combined[s["threshold"], s["lambda"], s["N"]]
        ratio = np.array([r[f"{s['method']}_ratio"] for r in group])
        s.update(
            campaign=group[0]["campaign"], checkpoint_is_terminal=group[0]["checkpoint_is_terminal"],
            smc_seconds_measured=group[0]["smc_seconds_measured"],
            fraction_one_expected_crude_hit=s["mean_crude_n"] * s["p_reference"],
            positive_runs=s["runs"] - s["zero_count"],
            median_relative_error=float(np.median(abs(ratio - 1))),
            relative_rmse=float(np.sqrt(np.mean((ratio - 1)**2))),
        )
    lookup = {(s["threshold"], s["lambda"], s["N"], s["method"]): s for s in table}
    return rows, table, combined, lookup


def metrics_figure(output, lookup, panel_specs):
    metrics = (("positive_runs", "Run positive su 24"),
               ("within_factor_ten", "Entro un fattore 10 su 24"),
               ("relative_standard_deviation", "Dev. standard relativa"))
    fig, axes = plt.subplots(3, len(panel_specs), figsize=(14.6, 9.3), sharex="col")
    for col, (title, segments) in enumerate(panel_specs):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for event, tilt, sizes in segments:
                x = [lookup[event, tilt, n, "smc"]["fraction_one_expected_crude_hit"] for n in sizes]
                for method, _, color, _ in METHODS:
                    y = [lookup[event, tilt, n, method][metric] for n in sizes]
                    ax.plot(x, y, color=color, marker="o", linestyle="-", ms=5, lw=1.7)
            ax.grid(axis="y", color="#dce2e7")
            ax.set_axisbelow(True)
            if any(lookup[event, tilt, n, "smc"]["fraction_one_expected_crude_hit"] >= 1
                   for event, tilt, sizes in segments for n in sizes):
                ax.axvline(1, color=REFERENCE_COLOR, linestyle=":", lw=1)
            if row < 2:
                ax.set_ylim(-.7, 24.9)
                ax.set_yticks((0, 6, 12, 18, 24))
                if row == 1:
                    ax.axhline(12, color="#9ca5ad", linestyle=":", lw=1)
            else:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
                ax.set_xlabel("Budget crude equivalente")
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title, fontsize=11.5, weight="bold")
    handles = [Line2D([], [], color=c, marker="o", label=label) for _, label, c, _ in METHODS]
    fig.suptitle("Come cambia la stima aumentando il budget", x=.072, y=.975, ha="left", fontsize=18, weight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .062), frameon=False, ncol=4, fontsize=9)
    small_sizes = panel_specs[0][1][0][2]
    sizes_note = ("Nei primi due pannelli si aggiungono N = 32 e 48. " if small_sizes != SIZES else "")
    fig.text(.5, .032, sizes_note + "Il 100% del budget è il costo crude di un evento atteso.",
             ha="center", fontsize=9)
    fig.text(.5, .013, "Ogni punto include 24 run, anche quelle a zero. A λ=2,5, N=64 e 96 sono checkpoint intermedi.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.071, right=.98, top=.90, bottom=.16, hspace=.24, wspace=.29)
    figure_dir = output / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "combined_compute_scaling_metrics.png", dpi=210, facecolor="white")
    plt.close(fig)


def error_figure(output, groups, lookup, panel_specs, *, logarithmic):
    ref_error = ZERO_ERROR if logarithmic else 1.
    zoom = .4 if logarithmic else 1.5
    fig, axes = plt.subplots(len(panel_specs), 2, figsize=(14, 10.2))
    for row, (title, segments) in enumerate(panel_specs):
        keys = [(event, tilt, n) for event, tilt, sizes in segments for n in sizes]
        plotted = {}
        for i, key in enumerate(keys):
            for method, _, _, _ in METHODS:
                ratio = np.array([r[f"{method}_ratio"] for r in groups[key]])
                errors = (np.array([r[f"{method}_log_relative_error"] for r in groups[key]])
                          if logarithmic else abs(ratio - 1))
                plotted[i, method] = errors
        max_error = max(zoom, *(data.max() for data in plotted.values()))
        labels = []
        for key in keys:
            s = lookup[(*key, "smc")]
            mark = "" if s["smc_seconds_measured"] else "*"
            labels.append(f"{s['fraction_one_expected_crude_hit']:.1%}\nN={key[2]}{mark}")
        for col, ax in enumerate(axes[row]):
            for i, key in enumerate(keys):
                for method, _, color, offset in METHODS:
                    data = plotted[i, method]
                    box(ax, data, i + offset, color)
                    if col == 0 and (clipped := int((data > zoom).sum())):
                        ax.text(i + offset, 1.012, f"↑{clipped}", transform=ax.get_xaxis_transform(),
                                color=color, ha="center", va="bottom", fontsize=8)
                if col == 1:
                    zeros = [lookup[(*key, m)]["zero_count"] for m in ("smc", "crude")]
                    ax.text(i, 1.012, f"{zeros[0]} / {zeros[1]}", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8)
            ax.axhline(ref_error, color=REFERENCE_COLOR, linestyle="--", lw=1)
            upper = zoom if col == 0 else max_error * 1.10
            ax.set_ylim(-upper * .024, upper)
            ax.set_xticks(range(len(keys)), labels, fontsize=8.5)
            ax.set_xlim(-.5, len(keys) - .5)
            ax.grid(axis="y", color="#dce2e7")
            ax.set_axisbelow(True)
            if not logarithmic:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            if row == 0:
                ax.set_title(("Tutte le run · zoom", "Tutte le run · distribuzione completa")[col],
                             fontsize=11, weight="bold", pad=17)
            if col == 0:
                ax.set_ylabel(title)
    handles = [Patch(facecolor=color, alpha=.55, label=label) for _, label, color, _ in METHODS]
    handles += [Line2D([], [], color="#15232e", label="Mediana"),
                Line2D([], [], color="#d99118", marker="D", ls="none", label="Media"),
                Line2D([], [], color=REFERENCE_COLOR, ls="--", label="Errore relativo 100%")]
    title = "Errore relativo logaritmico" if logarithmic else "Errore relativo assoluto"
    fig.suptitle(title, x=.08, y=.98, ha="left", fontsize=18, weight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .064), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .039, "Box 25°–75°, baffi a 1,5 IQR. ↑ oltre lo zoom. Sopra i box completi: zeri SMC / crude su 24.",
             ha="center", fontsize=8.5)
    fig.text(.5, .018, "Ascissa: budget crude equivalente e N, in ordine di budget. * Tempo intermedio stimato dalle transizioni.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.087, right=.98, bottom=.14, top=.915, hspace=.53, wspace=.24)
    metric = "log_relative_error" if logarithmic else "relative_error"
    figure_dir = output / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"combined_compute_scaling_{metric}.png", dpi=210, facecolor="white")
    plt.close(fig)


def uncertainty_figure(output, groups, lookup, panel_specs, *, positive_only, label_tilt=False):
    statistics = (
        (r"Media di $\widehat p_A/p_{\mathrm{rif}}$", "mean_ratio", "D", np.mean, {}),
        (r"Mediana di $\widehat p_A/p_{\mathrm{rif}}$", "median_ratio", "s", np.median, {}),
        (r"Deviazione standard relativa $s(\widehat p_A)/p_{\mathrm{rif}}$",
         "relative_standard_deviation", "o", np.std, {"ddof": 1}),
    )
    fig, axes = plt.subplots(len(panel_specs), len(statistics),
                             figsize=(17.7, 4 * len(panel_specs)), sharex="row")
    rng = np.random.default_rng(99925)
    interval_rows = []
    for row, (title, segments) in enumerate(panel_specs):
        keys = [(event, tilt, n) for event, tilt, sizes in segments for n in sizes]
        keys.sort(key=lambda key: lookup[(*key, "smc")]["fraction_one_expected_crude_hit"])
        budgets = [lookup[(*key, "smc")]["fraction_one_expected_crude_hit"] for key in keys]
        zero_counts = [tuple(lookup[(*key, method)]["zero_count"] for method, _, _, _ in METHODS)
                       for key in keys]
        labels = []
        for key, budget, (smc_zeros, crude_zeros) in zip(keys, budgets, zero_counts, strict=True):
            point = f"{key[1]:g}".replace(".", ",") if label_tilt else str(key[2])
            labels.append(f"{budget:.0%}\n{point}\n{smc_zeros}/{crude_zeros}/{len(groups[key])}")
        reference = f"{lookup[(*keys[0], 'smc')]['p_reference']:.4g}".replace(".", ",")
        series = {(method, field): [] for method, _, _, _ in METHODS
                  for _, field, _, _, _ in statistics}
        for key, budget in zip(keys, budgets, strict=True):
            runs = groups[key]
            # Resample whole runs. Share draws between methods with the same eligible run set.
            resamples = {}
            for method, _, _, _ in METHODS:
                ratios = np.array([r[f"{method}_ratio"] for r in runs])
                hit_field = "crude_hits" if method == "crude" else "event_particles"
                found = np.array([r[hit_field] > 0 for r in runs])
                if not np.isfinite(ratios).all() or np.any(ratios < 0) or not np.array_equal(found, ratios > 0):
                    raise ValueError(f"Invalid estimate or event count for {key}, {method}")
                if positive_only:
                    ratios = ratios[found]
                count = len(ratios)
                pool = hit_field if positive_only else "all"
                if count > 1 and pool not in resamples:
                    resamples[pool] = rng.integers(count, size=(20000, count))
                bootstrap = ratios[resamples[pool]] if count > 1 else None
                for _, field, _, statistic, kwargs in statistics:
                    minimum = 2 if field == "relative_standard_deviation" else 1
                    value = float(statistic(ratios, **kwargs)) if count >= minimum else None
                    if not positive_only and not np.isclose(
                        value, lookup[(*key, method)][field], rtol=1e-12, atol=1e-15,
                    ):
                        raise ValueError(f"Point estimate differs from the summary for {key}, {method}")
                    lo, hi = None, None
                    if bootstrap is not None:
                        lo, hi = np.quantile(statistic(bootstrap, axis=1, **kwargs), [.025, .975])
                    series[method, field].append((value, lo, hi))
                    interval_rows.append(dict(
                        threshold=key[0], **{"lambda": key[1]}, N=key[2], method=method, statistic=field,
                        subset="positive_only" if positive_only else "all", runs=count, total_runs=len(runs),
                        zero_runs=int((~found).sum()), budget_fraction=budget, estimate=value, lower_95=lo, upper_95=hi,
                    ))
        for col, (heading, field, marker, _, _) in enumerate(statistics):
            ax = axes[row, col]
            for method, _, color, _ in METHODS:
                values, lo, hi = np.asarray(series[method, field], dtype=float).T
                ax.fill_between(budgets, lo, hi, color=color, alpha=.16, lw=0)
                ax.plot(budgets, values, color=color, marker=marker, ms=4, lw=1.65, zorder=4)
                # Retain interval bounds when only one budget has a defined interval.
                for i, (lower, upper) in enumerate(zip(lo, hi, strict=True)):
                    if np.isfinite(lower) and not np.isfinite(lo[max(0, i - 1):i]).any() \
                            and not np.isfinite(lo[i + 1:i + 2]).any():
                        ax.vlines(budgets[i], lower, upper, color=color, alpha=.6)
            if row == 0:
                ax.set_title(heading, fontsize=11.5, weight="bold", pad=9)
            if col == 0:
                ax.set_ylabel(title + f"\nRiferimento {reference}\nRapporto (%)", fontsize=9.5)
            elif col == 1:
                ax.set_ylabel("Rapporto (%)", fontsize=9.5)
            else:
                ax.set_ylabel("Dev. standard relativa (%)", fontsize=9.5)
            if col < 2:
                ax.axhline(1, color=REFERENCE_COLOR, lw=.85, alpha=.8)
            linthresh = .001 if col < 2 and keys[0][0] == .999 else .5
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.margins(y=.07)
            ax.set_ylim(bottom=-.08 * linthresh)
            tick_candidates = ([0, .001, .01, .1, 1, 2, 5, 10, 100] if linthresh == .001
                               else [0, .25, .5, 1, 2, 5, 10, 100])
            ticks = [tick for tick in tick_candidates if tick <= ax.get_ylim()[1]]
            ax.set_yticks(ticks, [f"{tick * 100:g}%".replace(".", ",") for tick in ticks])
            ax.grid(axis="y", color="#dce2e7")
            ax.set_axisbelow(True)
            ax.set_xscale("log")
            ax.set_xticks(budgets, labels)
            ax.minorticks_off()
            ax.tick_params(axis="x", labelbottom=True, labelsize=7.2)
            point_label = "tilt λ" if label_tilt else "N"
            ax.set_xlabel(f"Budget crude equivalente · {point_label} · zeri SMC / zeri MC / totale per metodo",
                          fontsize=8.5, labelpad=6)
    sample_label = "Solo run con evento" if positive_only else "Tutte le run"
    fig.suptitle(f"Stime a pari tempo GPU · {sample_label}",
                 x=.086, y=.98, ha="left", fontsize=18, weight="bold")
    fig.text(.086, .937, "Statistiche tra i seed, espresse in percentuale del riferimento. La linea rossa indica un rapporto pari a 1.",
             ha="left", fontsize=10)
    handles = [Line2D([], [], color=color, marker="o", label=label)
               for _, label, color, _ in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .066), frameon=False, ncol=2)
    note = ("Gli zeri sono contati sulle 24 run originali. Le curve usano solo le stime positive. Con una sola run sono visibili solo media e mediana."
            if positive_only else
            "Ogni punto riassume 24 run per metodo, zeri inclusi. Il budget è espresso rispetto al costo crude di un evento atteso.")
    fig.text(.5, .046, note, ha="center", fontsize=9)
    fig.text(.5, .028, "Fasce bootstrap puntuali al 95%, 20.000 ricampionamenti per taglia. Ascisse logaritmiche e ordinate lineari vicino a zero, logaritmiche più in alto.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.086, right=.98, bottom=.16, top=.89, hspace=.64, wspace=.30)
    suffix = "_positive" if positive_only else ""
    figure_dir = output / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"rare_event_bootstrap_uncertainty{suffix}.png", dpi=210, facecolor="white")
    plt.close(fig)
    return interval_rows


def budget_statistic_figure(root, *, relative_sd=False, suffix="", label_tilt=False, extra_panels=()):
    if relative_sd:
        styles = (("relative_standard_deviation", "Deviazione standard relativa", "o", 1.85, .15),)
    else:
        styles = (("mean_ratio", "Media", "D", 1.85, .13),
                  ("median_ratio", "Mediana", "s", 1.15, .08))
    first_field = styles[0][0]
    subsets = (("all", "Tutte le run"), ("positive_only", "Solo run con evento"))
    fields = {field for field, *_ in styles}
    records = read_csv(root / f"rare_event_bootstrap_uncertainty{'' if extra_panels else suffix}.csv")
    events = sorted({float(r["threshold"]) for r in records})
    panel_data = [([r for r in records if float(r["threshold"]) == event], label_tilt) for event in events]
    panel_data.extend(extra_panels)
    single_event = len(panel_data) == 1
    wide = any(by_tilt and len({float(r["lambda"]) for r in data}) > 12 for data, by_tilt in panel_data)
    if extra_panels:
        size = (21 if wide else 15, 3.15 * len(panel_data))
    elif single_event:
        size = (21, 6.2) if wide else (13, 5.1)
    else:
        size = (15, 10.8)
    fig, axes = plt.subplots(len(panel_data), 2, figsize=size, sharey="row", squeeze=False)
    for row, (data, by_tilt) in enumerate(panel_data):
        event, = {float(r["threshold"]) for r in data}
        point_field, fixed_field = ("lambda", "N") if by_tilt else ("N", "lambda")
        fixed, = {float(r[fixed_field]) for r in data}
        selected = [r for r in data if r["statistic"] in fields]
        index = {(float(r[point_field]), r["subset"], r["method"], r["statistic"]): r for r in selected}
        if not selected or len(index) != len(selected):
            raise ValueError("Expected one value per statistic, budget, method and subset")
        points = sorted({key[0] for key in index},
                        key=lambda point: float(index[point, "all", "smc", first_field]["budget_fraction"]))
        long_ladder = by_tilt and len(points) > 12
        budgets, labels = [], []
        for point in points:
            smc, crude = (index[point, "all", method, first_field] for method, _, _, _ in METHODS)
            budgets.append(float(smc["budget_fraction"]))
            precision = 2 if single_event or event == .9995 else 0
            budget_label = f"{budgets[-1]:.{precision}%}".replace(".", ",")
            point_label = f"{point:g}".replace(".", ",")
            labels.append(f"{budget_label}\n{point_label}\n{smc['zero_runs']}/{crude['zero_runs']}/{smc['total_runs']}")
        for col, (subset, heading) in enumerate(subsets):
            ax = axes[row, col]
            for method, _, color, _ in METHODS:
                for field, _, marker, width, alpha in styles:
                    series = [index[point, subset, method, field] for point in points]
                    values, lo, hi = np.array([
                        [float(r[key]) if r[key] not in (None, "") else np.nan
                         for key in ("estimate", "lower_95", "upper_95")]
                        for r in series
                    ]).T
                    ax.fill_between(budgets, lo, hi, color=color, alpha=alpha, lw=0)
                    valid = np.isfinite(lo) & np.isfinite(hi)
                    isolated = valid & ~np.r_[False, valid[:-1]] & ~np.r_[valid[1:], False]
                    ax.vlines(np.asarray(budgets)[isolated], lo[isolated], hi[isolated], color=color, alpha=.6)
                    ax.plot(budgets, values, color=color, lw=width, marker=marker, ms=5,
                            markerfacecolor="white" if field == "median_ratio" else color, zorder=3)
            if not relative_sd:
                ax.axhline(1, color=REFERENCE_COLOR, lw=1, alpha=.8)
            if subset == "positive_only" and all(
                int(index[point, subset, "crude", first_field]["runs"]) == 0 for point in points
            ):
                ax.text(.98, .97, "Crude MC: nessuna run positiva", transform=ax.transAxes,
                        ha="right", va="top", color="#515c66", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
            if not by_tilt:
                ax.set_xscale("log")
            ax.set_xticks(budgets, labels, fontsize=7 if long_ladder else 8)
            ax.minorticks_off()
            ax.grid(axis="y", color="#dce2e7")
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(heading, loc="left", weight="bold", fontsize=12)
            if col == 0:
                unit = "Dev. standard relativa (%)" if relative_sd else "Rapporto (%)"
                fixed_label = "N" if by_tilt else "λ"
                ax.set_ylabel(f"q ≥ {event:g} · {fixed_label} = {fixed:g}".replace(".", ",") + f"\n{unit}",
                              fontsize=10)
            if extra_panels or row == len(panel_data) - 1:
                label = "tilt λ" if by_tilt else "N"
                ax.set_xlabel(f"Budget crude equivalente · {label} · zeri SMC / zeri MC / totale per metodo", fontsize=9)
        ax = axes[row, 0]
        linthresh = .5
        candidates = [0, .25, .5, 1, 2, 5, 10]
        if event >= .999:
            linthresh = .025 if relative_sd else .001
            candidates = ([0, .01, .025, .05, .1, .25, .5, 1, 2, 5] if relative_sd
                          else [0, .001, .01, .1, 1, 2, 5, 10])
            if single_event or event == .9995:
                candidates = ([0, .01, .1, 1, 10, 100, 1000] if relative_sd
                              else [0, .001, .01, .1, 1, 10, 100, 1000])
            if long_ladder and not relative_sd:
                linthresh = .000001
                candidates = [0, .000001, .00001, .0001, .001, .01, .1, 1, 10]
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.margins(y=.07)
        ax.set_ylim(bottom=-.08 * linthresh)
        ticks = [tick for tick in candidates if tick <= ax.get_ylim()[1]]
        ax.set_yticks(ticks, [f"{tick * 100:g}%".replace(".", ",") for tick in ticks])
        ax.minorticks_off()
        axes[row, 1].minorticks_off()
    title = ("Deviazione standard relativa al crescere del budget" if relative_sd
             else "Media e mediana delle stime al crescere del budget")
    subtitle = (r"$s(\widehat p_A)/p_{\mathrm{rif}}$ tra i seed. Confronto a pari tempo GPU." if relative_sd
                else r"Rapporto $\widehat p_A/p_{\mathrm{rif}}$ tra i seed. Confronto a pari tempo GPU.")
    pooled = False
    if single_event and records[0].get("reference_hits"):
        reference = records[0]
        pooled = reference.get("reference_basis") == "pooled_crude"
        label = "Pool crude aggregato" if pooled else "Riferimento crude"
        samples = f"{int(reference['reference_samples']):,}".replace(",", ".")
        subtitle += f" {label}: {reference['reference_hits']} eventi su {samples}."
    if extra_panels:
        reference = extra_panels[0][0][0]
        pooled = reference.get("reference_basis") == "pooled_crude"
        samples = f"{int(reference['reference_samples']):,}".replace(",", ".")
        label = "riferimento aggregato" if pooled else "riferimento indipendente"
        subtitle += f" Per q ≥ 0,9995, {label} di {reference['reference_hits']} eventi su {samples}."
    fig.suptitle(title, x=.085, y=.978, ha="left", fontsize=18, weight="bold")
    fig.text(.085, .90 if single_event else .941, subtitle, fontsize=11)
    handles = [Line2D([], [], color=color, lw=2, label=label) for _, label, color, _ in METHODS]
    if not relative_sd:
        handles.extend(Line2D([], [], color="#2e3a43", marker=marker, lw=width, label=label,
                              markerfacecolor="#2e3a43" if field == "mean_ratio" else "white")
                       for field, label, marker, width, _ in styles)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .085 if single_event else .063),
               ncol=len(handles), frameon=False)
    point_label = "N o tilt λ" if extra_panels else ("tilt λ" if label_tilt else "N")
    fig.text(.5, .063 if single_event else .043,
             f"Sotto ogni punto: budget, {point_label} e zeri SMC / zeri MC / totale originale per metodo. Le ordinate coincidono tra le due colonne.",
             ha="center", fontsize=9)
    note = ("Fasce bootstrap puntuali al 95%. Servono almeno due stime per calcolare la deviazione standard."
            if relative_sd else
            "Fasce bootstrap puntuali al 95%. Il 100% in ordinata corrisponde al riferimento. Il 100% del budget equivale a un evento crude atteso.")
    if pooled:
        note = "Fasce bootstrap puntuali al 95% a riferimento fisso. Il pool aggregato include i prefissi usati per crude MC a pari tempo."
    fig.text(.5, .023, note, ha="center", fontsize=9)
    bottom, top = (.135, .915) if extra_panels else ((.30, .81) if single_event else (.16, .895))
    fig.subplots_adjust(left=.085, right=.98, bottom=bottom, top=top, hspace=.53, wspace=.14)
    statistic = "relative_sd" if relative_sd else "mean_median"
    figure_dir = root / 'plots'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"combined_compute_scaling_{statistic}{suffix}.png", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--low", required=True, type=Path)
    parser.add_argument("--ultra-low", type=Path)
    plot = parser.add_mutually_exclusive_group()
    plot.add_argument("--mean-median-only", action="store_true", help="Render the mean and median figure")
    plot.add_argument("--relative-sd-only", action="store_true", help="Render the relative standard deviation figure")
    args = parser.parse_args()
    if args.mean_median_only or args.relative_sd_only:
        budget_statistic_figure(args.low, relative_sd=args.relative_sd_only)
        return
    if args.previous is None:
        parser.error("--previous is required for the full combined analysis")
    rows, table, groups, lookup = combine(args.previous, args.low, args.ultra_low)
    panel_specs = panels(args.ultra_low)
    write_csv(args.low / "combined_compute_scaling_runs.csv", rows)
    write_csv(args.low / "combined_compute_scaling_summary.csv", table)
    metrics_figure(args.low, lookup, panel_specs)
    error_figure(args.low, groups, lookup, panel_specs, logarithmic=False)
    error_figure(args.low, groups, lookup, panel_specs, logarithmic=True)
    uncertainty = uncertainty_figure(args.low, groups, lookup, panel_specs, positive_only=False)
    uncertainty.extend(uncertainty_figure(args.low, groups, lookup, panel_specs, positive_only=True))
    write_csv(args.low / "rare_event_bootstrap_uncertainty.csv", uncertainty)
    print(f"Verified {len(rows)} event-level comparisons and {len(table)} method-by-budget summaries")


if __name__ == "__main__":
    main()
