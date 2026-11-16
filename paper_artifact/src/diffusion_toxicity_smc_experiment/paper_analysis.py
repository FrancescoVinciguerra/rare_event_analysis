"""Analyse saved ladders using the existing log-space estimators."""
from collections import defaultdict
from io import BytesIO
import math

import numpy as np

from .crude import fixed_mc_block, native_clock_count
from .storage import atomic_write_json, write_csv


def checkpoint_targets(settings):
    targets = defaultdict(list)
    for item in settings["ladder"]["observe"]:
        targets[item["threshold"]].append(item["tilt"])
    return targets


def _log_estimates(reverse, event, log_z):
    numerator = float(np.logaddexp.reduce(reverse[event], initial=-np.inf))
    return log_z + numerator, numerator - float(np.logaddexp.reduce(reverse)), int(event.sum())


def terminal_log_estimates(population, threshold):
    particles = population["particles"]
    summary = population["summary"]
    reverse = np.array([p["final_log_weight"] - summary["lambda"] * p["reward"] for p in particles])
    event = np.array([p["event_toxic_probability"] >= threshold for p in particles])
    return _log_estimates(reverse, event, summary["log_normalizer_estimate"])


def checkpoint_log_estimates(checkpoint, threshold):
    reverse = np.asarray(checkpoint["normalized_log_weights"]) - checkpoint["lambda"] * np.asarray(checkpoint["rewards"])
    event = np.asarray(checkpoint["scores"]) >= threshold
    return _log_estimates(reverse, event, checkpoint["log_z_hat"])


def select_populations(population, targets, refs, *, sweeps_per_level, denoising_steps):
    """Use the actual post-MH checkpoints, with the measured cumulative cost."""
    summary = population["summary"]
    final_tilt = summary["lambda"]
    native = summary["total_decoder_particle_steps"] + summary["num_particles"] + summary["mcmc_proposals"]
    terminal = dict(
        tilt=final_tilt, seconds=summary["runtime_seconds"], native=native,
        acceptance=summary["mcmc_acceptance_rate"], ess=summary["final_smc_ess"],
        roots=summary["distinct_root_ancestors"], checkpoint=None,
    )
    if not targets:
        yield terminal, refs
        return

    n = summary["num_particles"]
    moves = population["mutation_diagnostics"]
    counts = [(0, 0, (denoising_steps + 1) * n)]
    for move in moves:
        cumulative = counts[-1][2] + move["rollback_particle_steps"] + move["proposals"]
        counts.append((move["level"], move["sweep"] + 1, cumulative))
    checkpoints = population["population_checkpoints"]
    previous_seconds = -1.0
    for cp, (level, sweep, transitions) in zip(checkpoints, counts, strict=True):
        seconds = cp["runtime_seconds_cumulative"]
        if ((cp["level"], cp["sweep"], cp["native_transitions_cumulative"]) != (level, sweep, transitions)
                or cp["num_particles"] != n or seconds <= previous_seconds):
            raise ValueError("Invalid checkpoint time or transition ledger")
        previous_seconds = seconds
    if previous_seconds != terminal["seconds"] or counts[-1][2] != native:
        raise ValueError("Final checkpoint cost differs from the measured terminal cost")

    selected = {cp["lambda"]: cp for cp in checkpoints if cp["sweep"] == sweeps_per_level}
    levels = {row["eta"]: row for row in population["level_diagnostics"]}
    for tilt in sorted({tilt for tilts in targets.values() for tilt in tilts}):
        events = {threshold: ref for threshold, ref in refs.items() if tilt in targets[threshold]}
        if tilt == final_tilt:
            yield terminal, events
            continue
        cp = selected[tilt]
        level = levels[tilt]
        used_moves = [move for move in moves if move["level"] <= cp["level"]]
        if cp["log_z_hat"] != level["log_normalizer_estimate"]:
            raise ValueError("Checkpoint and level normalizers disagree")
        yield dict(
            tilt=tilt, seconds=cp["runtime_seconds_cumulative"], native=cp["native_transitions_cumulative"],
            acceptance=sum(m["accepted"] for m in used_moves) / sum(m["proposals"] for m in used_moves),
            ess=cp["smc_ess_fraction"] * n, roots=level["distinct_root_ancestors"], checkpoint=cp,
        ), events


def comparison_rows(settings, records, crude):
    targets = checkpoint_targets(settings)
    refs = {q: crude.hits[q] / crude.samples for q in targets}
    for run, record in sorted(records.items(), key=lambda item: (item[0].particles, item[0].replicate)):
        p = record["population"]
        timing = record["timing"]
        if timing["excluded_from_estimates_and_measured_run_times"] is not True:
            raise ValueError("Crude calibration must be independent of estimates and SMC timing")
        calibration = timing["seconds_per_path"]
        states = select_populations(p, targets, refs,
                                    sweeps_per_level=settings["ladder"]["moves_per_level"],
                                    denoising_steps=settings["model"]["decoder"]["denoising_steps"])
        for state, events in states:
            seconds = state["seconds"]
            n = native_clock_count(seconds, calibration)
            prefix, start, _ = fixed_mc_block(crude.matched_scores, seed_rank=run.replicate,
                                              total_seeds=settings["grid"]["replicates"], required=n)
            for threshold, ref in events.items():
                cp = state["checkpoint"]
                if cp is None:
                    log_smc, log_snis, events_found = terminal_log_estimates(p, threshold)
                    saved = p["corrections"]
                else:
                    log_smc, log_snis, events_found = checkpoint_log_estimates(cp, threshold)
                    saved = cp["corrections"]
                smc, snis = math.exp(log_smc), math.exp(log_snis)
                if threshold in settings["ladder"]["recorded_thresholds"]:
                    old = saved[str(threshold)]
                    if (events_found != old["event_particles"] or not math.isclose(old["smc"], smc, rel_tol=1e-12)
                            or not math.isclose(old["snis"], snis, rel_tol=1e-12)):
                        raise ValueError(f"Reconstructed corrections differ for {run.filename}")
                hits = int(np.count_nonzero(prefix >= threshold))
                yield {"N": run.particles, "seed": run.seed, "replicate": run.replicate,
                       "timing_origin": timing["origin"], "timing_group": timing["group"],
                       "threshold": threshold, "lambda": state["tilt"], "p_reference": ref,
                       "smc": smc, "snis": snis, "log_smc": log_smc, "log_snis": log_snis,
                       "crude": hits / n, "crude_hits": hits, "crude_n": n, "crude_pool_start": start,
                       "smc_seconds": seconds, "crude_seconds_per_path": calibration,
                       "native_transitions": state["native"], "event_particles": events_found,
                       "mh_acceptance": state["acceptance"], "smc_ess": state["ess"]}


METRICS = {
    "median_absolute_relative_error": ("Median absolute relative error", lambda x: np.median(abs(x - 1), axis=-1)),
    "mean_absolute_relative_error": ("Mean absolute relative error", lambda x: np.mean(abs(x - 1), axis=-1)),
    "relative_rmse": ("Relative RMSE", lambda x: np.sqrt(np.mean((x - 1)**2, axis=-1))),
    "relative_standard_deviation": ("Relative standard deviation across runs", lambda x: np.std(x, axis=-1, ddof=1)),
    "zero_estimate_rate": ("Zero-estimate rate", lambda x: np.mean(x == 0, axis=-1)),
}


def summarize(rows, figures):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["timing_group"], row["threshold"], row["lambda"], row["N"]].append(row)
    metrics = [*figures["metrics"], *figures["summary_only_metrics"]]
    output = []
    for (timing, threshold, tilt, n), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row["replicate"])
        runs = len(group)
        ref = group[0]["p_reference"]
        if ref <= 0:
            raise ValueError(f"Relative metrics require at least one crude reference event at threshold {threshold}")
        rng = np.random.default_rng(figures["bootstrap_seed"])
        indices = rng.integers(runs, size=(figures["bootstrap_resamples"], runs))
        for method in ("smc", "crude"):
            values = np.asarray([row[method] for row in group]) / ref
            for metric in metrics:
                statistic = METRICS[metric][1]
                estimate, low, high = None, None, None
                if metric != "relative_standard_deviation" or runs > 1:
                    estimate = float(statistic(values))
                    low, high = np.quantile(statistic(values[indices]), figures["bootstrap_quantiles"]).tolist()
                output.append(dict(timing_group=timing, threshold=threshold, **{"lambda": tilt}, N=n,
                                   method=method, metric=metric, runs=runs, zero_count=int((values == 0).sum()),
                                   budget=float(np.mean([row["crude_n"] for row in group])) * ref,
                                   estimate=estimate, lower=low, upper=high))
    return output


def render_figures(summary, figures, crude, output):
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    lo_quantile, hi_quantile = figures["bootstrap_quantiles"]
    confidence = 100 * (hi_quantile - lo_quantile)
    groups = defaultdict(list)
    for row in summary:
        if row["metric"] in figures["metrics"]:
            groups[row["timing_group"], row["threshold"], row["lambda"], row["metric"]].append(row)
    styles = {"smc": ("RollBack-SMC", "#14698f", "o"), "crude": ("Crude MC", "#a44e28", "s")}
    with matplotlib.rc_context({"text.usetex": figures["usetex"], "font.size": 12, "axes.axisbelow": True}):
        for (timing, q, tilt, metric), records in sorted(groups.items()):
            title = METRICS[metric][0]
            fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
            caption_counts = []
            try:
                for method, (label, color, marker) in styles.items():
                    rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["budget"])
                    caption_counts.append(f"{label} zero runs: " + ", ".join(f"{r['zero_count']}/{r['runs']}" for r in rows))
                    rows = [r for r in rows if r["estimate"] is not None]
                    if rows:
                        x, y, lo, hi = np.array([[r[k] for k in ("budget", "estimate", "lower", "upper")] for r in rows]).T
                        ax.plot(x, y, color=color, marker=marker, label=label)
                        ax.vlines(x, lo, hi, color=color, linewidth=1)
                        ax.plot(np.r_[x, x], np.r_[lo, hi], linestyle="none", marker="_", color=color)
                ax.set(title=title, xlabel="Expected rare events under equal-cost MC",
                       ylabel=title.removesuffix(" across runs") + (r" (\%)" if figures["usetex"] else " (%)"))
                ax.grid(axis="y", alpha=.25)
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                if ax.lines:
                    ax.legend(frameon=False)
                key = output / "figures"
                name = f"q{q:g}_lambda{tilt:g}".replace(".", "p")
                relative = f"{timing}/{name}/{metric}.png"
                paper = key / "without_captions" / relative
                captioned = key / "with_captions" / relative
                paper.parent.mkdir(parents=True, exist_ok=True)
                captioned.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(paper, dpi=250)
                lines = [f"Threshold {q:g}, tilt {tilt:g}. Reference {crude.hits[q]}/{crude.samples}.",
                         "One unit on the horizontal axis is one expected crude MC event.",
                         f"Bars show pointwise {confidence:g} percent bootstrap intervals across independent runs."]
                if metric == "zero_estimate_rate":
                    lines.extend(caption_counts)
                append_caption(paper, captioned, lines)
            finally:
                plt.close(fig)


def append_caption(paper, captioned, lines):
    """Keep the plotted pixels identical in both exports."""
    from matplotlib import pyplot as plt
    from PIL import Image

    caption = plt.figure(figsize=(8, .23 * (len(lines) + 1)))
    try:
        caption.text(.08, .9, "\n".join(lines), va="top", fontsize=9)
        with BytesIO() as buf:
            caption.savefig(buf, format="png", dpi=250)
            buf.seek(0)
            with Image.open(paper) as plot, Image.open(buf) as footer:
                result = Image.new("RGB", (plot.width, plot.height + footer.height), "white")
                result.paste(plot, (0, 0))
                result.paste(footer, (0, plot.height))
                result.save(captioned, dpi=(250, 250))
    finally:
        plt.close(caption)


def write_reference(root, crude):
    atomic_write_json(root / "reference.json", {"samples": crude.samples, "hits": crude.hits,
                      "cohorts": crude.cohorts, "scores_manifest_sha256": crude.manifest_sha256,
                      "p_reference": {q: n / crude.samples for q, n in crude.hits.items()}})


def analyse(settings, root, records, crude):
    write_reference(root, crude)
    rows = list(comparison_rows(settings, records, crude))
    write_csv(root / "runs.csv", rows)
    summary = summarize(rows, settings["figures"])
    write_csv(root / "summary.csv", summary)
    render_figures(summary, settings["figures"], crude, root)
