"""Recompute the 0.9995 event on the saved fixed-tilt, wall-clock-matched study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from combine_particle_budgets import budget_statistic_figure, read_csv, write_csv
from plot_particle_scaling import Line2D, METHODS, Patch, plt

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT.parent / "shared"))

from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs
from diffusion_toxicity_smc_experiment.crude_tail_reference import validated_scores
from diffusion_toxicity_smc_experiment.particle_scaling import (
    checkpoint_log_estimates, terminal_log_estimates, validated_pool,
)
from diffusion_toxicity_smc_experiment.same_run_reuse import fixed_mc_block, mc_pool, native_clock_count
from diffusion_toxicity_smc_experiment.storage import atomic_write_json
from rea_llms.statistics import wilson_interval


THRESHOLD = .9995
SUFFIX = "_q09995"
POOLED_REFERENCE = "pooled_reference_q09995.json"


def read_json(path):
    return json.loads(path.read_bytes())


def matched_runs(root, original, pool):
    for row in original:
        n, seed, tilt = int(row["N"]), int(row["seed"]), float(row["lambda"])
        campaign = root / row["campaign"]
        source, = campaign.glob(f"workers/worker_*/populations/*seed{seed}.json")
        population = read_json(source)
        summary = population["summary"]
        if summary["seed"] != seed or summary["num_particles"] != n or population["phi"]["name"] != "logit":
            raise ValueError(f"Unexpected population in {source}")
        if row["checkpoint_is_terminal"] == "True":
            if summary["lambda"] != tilt:
                raise ValueError(f"Wrong terminal tilt in {source}")
            old, _, old_hits = terminal_log_estimates(population, .999)
            log_smc, _, hits = terminal_log_estimates(population, THRESHOLD)
            max_score = max(p["event_toxic_probability"] for p in population["particles"])
        else:
            checkpoint, = (c for c in population["population_checkpoints"] if c["lambda"] == tilt and c["sweep"] == 4)
            if not math.isclose(checkpoint["runtime_seconds_cumulative"], float(row["smc_seconds"]), rel_tol=1e-12):
                raise ValueError(f"Wrong checkpoint cost in {source}")
            old, _, old_hits = checkpoint_log_estimates(checkpoint, .999)
            log_smc, _, hits = checkpoint_log_estimates(checkpoint, THRESHOLD)
            max_score = max(checkpoint["scores"])
        crude_n = int(row["crude_n"])
        prefix, start, _ = fixed_mc_block(pool, seed_rank=int(row["replicate"]), total_seeds=24, required=crude_n)
        crude_hits = int((prefix >= THRESHOLD).sum())
        if (old_hits != int(row["event_particles"]) or not math.isclose(old, float(row["log_smc"]), rel_tol=1e-12)
                or start != int(row["crude_pool_start"]) or int((prefix >= .999).sum()) != int(row["crude_hits"])
                or log_smc > old or hits > old_hits):
            raise ValueError(f"Failed to reproduce the original matched result for seed {seed}")
        yield dict(
            threshold=THRESHOLD, **{"lambda": tilt}, N=n, seed=seed, campaign=row["campaign"],
            checkpoint_is_terminal=row["checkpoint_is_terminal"], smc_seconds=float(row["smc_seconds"]),
            smc_native_transitions=int(row["smc_native_transitions"]), crude_n=crude_n, crude_pool_start=start,
            event_particles=hits, max_score=max_score, crude_hits=crude_hits,
            smc=math.exp(log_smc), log_smc=log_smc, crude=crude_hits / crude_n,
            population_source=str(source.relative_to(root)),
        )


def matched_ladder(root, pool, levels):
    logit = root / "camera_ready_same_run_logit"
    blocks = read_json(logit / "saved_MC_blocks.json")
    original = {(int(r["seed"]), float(r["lambda"])): r for r in read_csv(logit / "checkpoint_runs.csv")
                if float(r["threshold"]) == .999 and int(r["sweep"]) == 4 and float(r["lambda"]) in levels}
    for source in sorted(logit.glob("workers/worker_*/populations/*.json")):
        population = read_json(source)
        summary = population["summary"]
        seed, n = summary["seed"], summary["num_particles"]
        if n != 128 or population["phi"]["name"] != "logit" or summary["lambda"] != 8.:
            raise ValueError(f"Unexpected saved ladder in {source}")
        block = blocks[str(seed)]
        start, scores = block["pool_start"], np.asarray(block["scores"])
        if not np.array_equal(scores, pool[start:start + len(scores)]):
            raise ValueError(f"Saved MC prefix differs from its source for seed {seed}")
        for cp in population["population_checkpoints"]:
            tilt = cp["lambda"]
            if cp["sweep"] != 4 or tilt not in levels:
                continue
            row = original.pop((seed, tilt))
            seconds = cp["runtime_seconds_cumulative"]
            crude_n = int(row["crude_n"])
            old, _, old_hits = checkpoint_log_estimates(cp, .999)
            log_smc, _, hits = checkpoint_log_estimates(cp, THRESHOLD)
            if (not math.isclose(seconds, float(row["smc_runtime_seconds_cumulative"]), rel_tol=1e-12)
                    or crude_n != native_clock_count(seconds, float(row["calibration_seconds_per_path"]))
                    or crude_n > len(scores) or start != int(row["MC_pool_start"])
                    or int((scores[:crude_n] >= .999).sum()) != int(row["crude_event_particles"])
                    or old_hits != int(row["event_particles"]) or log_smc > old
                    or not math.isclose(math.exp(old), float(row["smc"]), rel_tol=1e-12)):
                raise ValueError(f"Failed to reproduce the checkpoint at seed {seed}, tilt {tilt}")
            crude_hits = int((scores[:crude_n] >= THRESHOLD).sum())
            yield dict(
                threshold=THRESHOLD, **{"lambda": tilt}, N=n, seed=seed, campaign=logit.name,
                checkpoint_is_terminal=cp["level_is_final"], smc_seconds=seconds,
                smc_native_transitions=cp["native_transitions_cumulative"], crude_n=crude_n, crude_pool_start=start,
                event_particles=hits, max_score=max(cp["scores"]), crude_hits=crude_hits,
                smc=math.exp(log_smc), log_smc=log_smc, crude=crude_hits / crude_n,
                population_source=str(source.relative_to(root)),
            )
    if original:
        raise ValueError("Some original checkpoint comparisons were missing")


def bootstrap_rows(runs, reference_hits, reference_samples, *, expected_runs=24, pooled=False):
    reference = reference_hits / reference_samples
    rng = np.random.default_rng(99950)
    for n, tilt in sorted({(r["N"], r["lambda"]) for r in runs}):
        group = [r for r in runs if (r["N"], r["lambda"]) == (n, tilt)]
        if len(group) != expected_runs or len({r["seed"] for r in group}) != expected_runs:
            raise ValueError(f"Expected {expected_runs} seeds for N={n}, tilt={tilt}")
        budgets = [r["crude_n"] for r in group]
        paired_draws = rng.integers(expected_runs, size=(20000, expected_runs))
        for method in ("smc", "crude"):
            ratios = np.array([r[method] / reference for r in group])
            positive = ratios[ratios > 0]
            for subset, values in (("all", ratios), ("positive_only", positive)):
                count = len(values)
                draws = None
                if count > 1:
                    indices = paired_draws if subset == "all" else rng.integers(count, size=(20000, count))
                    draws = values[indices]
                for field, statistic, kwargs in (
                    ("mean_ratio", np.mean, {}), ("median_ratio", np.median, {}),
                    ("relative_standard_deviation", np.std, {"ddof": 1}),
                ):
                    minimum = 2 if field == "relative_standard_deviation" else 1
                    value = float(statistic(values, **kwargs)) if count >= minimum else None
                    lo, hi = None, None
                    if draws is not None:
                        lo, hi = np.quantile(statistic(draws, axis=1, **kwargs), [.025, .975])
                    yield dict(
                        threshold=THRESHOLD, **{"lambda": tilt}, N=n, method=method, statistic=field,
                        subset=subset, runs=count, total_runs=len(group), zero_runs=len(group) - len(positive),
                        budget_fraction=float(np.mean(budgets)) * reference, mean_crude_n=float(np.mean(budgets)),
                        p_reference=reference, reference_hits=reference_hits, reference_samples=reference_samples,
                        reference_basis="pooled_crude" if pooled else "separate_crude",
                        estimate=value, lower_95=lo, upper_95=hi,
                    )


def build_pooled_reference(root, original_seed_audit):
    study, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_extreme_tail_scaling.yaml")
    pilot, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_low_particle_budget.yaml")
    direct = study["reference"]
    audit = original_seed_audit.read_text(encoding="utf-8")
    seeds = [int(seed) for seed in re.findall(r"^ORIGINAL_SEED=(\d+)$", audit, re.M)]
    shard_count, = re.findall(r"^SHARDS=(\d+)$", audit, re.M)
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) != int(shard_count):
        raise ValueError("The original crude shard inventory is incomplete or has duplicate seeds")
    streams = set(seeds)
    cohorts = [dict(
        name="original_direct", samples=direct["samples"][0], hits=direct["discoveries"][0],
        seed_streams=len(seeds), source=direct["source"], source_sha256=direct["source_sha256"],
        seed_inventory_sha256=hashlib.sha256("\n".join(map(str, sorted(seeds))).encode()).hexdigest(),
    )]
    for config in (study, pilot):
        pool, sources = validated_pool(root, config)
        batch_seeds = {r["seed"] for r in sources}
        if streams & batch_seeds:
            raise ValueError(f"Overlapping seed streams in {config['mc_pool']}")
        streams.update(batch_seeds)
        cohorts.append(dict(name=config["mc_pool"], samples=len(pool), hits=int((pool >= THRESHOLD).sum()),
                            seed_streams=len(batch_seeds), scores_sha256=config["mc_pool_scores_sha256"]))

    crude_root = root / "camera_ready_crude_tail_reference"
    manifest = read_json(crude_root / "manifest.json")
    reported = read_json(crude_root / "reference.json")
    status = read_json(crude_root / "report_status.json")
    if (not status["complete"] or reported["threshold"] != THRESHOLD
            or any(manifest[k] != direct[k] for k in ("decoder_id", "evaluator_id"))):
        raise ValueError("The latest crude campaign must be complete and use the same decoder and evaluator")
    hits, count, batches = 0, 0, 0
    for worker, planned in enumerate(manifest["planned_batches"]):
        for seed, n in planned:
            if seed in streams:
                raise ValueError(f"Repeated crude stream {seed}")
            streams.add(seed)
            path = crude_root / "workers" / f"worker_{worker}" / "baselines" / f"seed{seed}.json"
            scores = validated_scores(read_json(path), seed, n)
            hits += int((scores >= THRESHOLD).sum())
            count += len(scores)
            batches += 1
    if (hits, count, batches) != (reported["hits"], reported["samples"], status["completed_runs"]):
        raise ValueError("Latest crude scores disagree with the completion report")
    cohorts.append(dict(name=crude_root.name, samples=count, hits=hits, seed_streams=batches))
    count = sum(r["samples"] for r in cohorts)
    hits = sum(r["hits"] for r in cohorts)
    result = dict(threshold=THRESHOLD, hits=hits, samples=count, p_hat=hits / count,
                  wilson_95=wilson_interval(hits, count), reference_basis="pooled_crude",
                  decoder_id=direct["decoder_id"], evaluator_id=direct["evaluator_id"],
                  distinct_seed_streams=len(streams), includes_matched_crude_pool=True, cohorts=cohorts)
    atomic_write_json(root / "camera_ready_low_particle_budget" / POOLED_REFERENCE, result)
    print(f"Pooled reference: {hits}/{count}, distinct seed streams: {len(streams)}")


def use_pooled_reference(root, fallback):
    path = root / "camera_ready_low_particle_budget" / POOLED_REFERENCE
    return read_json(path) if path.exists() else fallback


def extreme_tail_results(root):
    output = root / "camera_ready_extreme_tail_scaling"
    crude_root = root / "camera_ready_crude_tail_reference"
    status = read_json(output / "report_status.json")
    crude_status = read_json(crude_root / "report_status.json")
    reference = read_json(crude_root / "reference.json")
    manifest = read_json(crude_root / "manifest.json")
    study, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_extreme_tail_scaling.yaml")
    if (not status["complete"] or status["completed_runs"] != study["independent_runs"]
            or not crude_status["complete"] or reference["threshold"] != THRESHOLD
            or not 0 < reference["hits"] <= reference["samples"] == reference["target_samples"]
            or any(manifest[key] != study["reference"][key] for key in ("decoder_id", "evaluator_id"))):
        raise ValueError("Expected completed SMC and crude reference with the same decoder and evaluator")

    runs = []
    cohorts = {}
    for row in read_csv(output / "runs.csv"):
        tilt, n, seed = float(row["lambda"]), int(row["N"]), int(row["seed"])
        crude_n, crude_hits = int(row["crude_n"]), int(row["crude_hits"])
        smc, crude = float(row["smc"]), float(row["crude"])
        seconds = float(row["smc_seconds"])
        matched_n = native_clock_count(seconds, float(row["crude_calibration_seconds_per_path"]))
        if (float(row["threshold"]) != THRESHOLD or n != 256 or matched_n != crude_n
                or not math.isclose(crude, crude_hits / crude_n, rel_tol=1e-12)
                or not math.isfinite(smc) or smc < 0 or (smc > 0) != (int(row["event_particles"]) > 0)):
            raise ValueError(f"Invalid saved comparison for seed {seed}, tilt {tilt}")
        cohorts.setdefault(tilt, set()).add((seed, int(row["crude_pool_start"])))
        runs.append(dict(N=n, **{"lambda": tilt}, seed=seed, smc_seconds=seconds, crude_n=crude_n, smc=smc, crude=crude))
    if (set(cohorts) != set(study["checkpoint_tilts"][THRESHOLD])
            or any(cohort != cohorts[.5] for cohort in cohorts.values())):
        raise ValueError("Each tilt must use the same saved set of seeds and crude blocks")
    return runs, reference, study["independent_runs"]


def plot_extreme_tail_scaling(root):
    output = root / "camera_ready_extreme_tail_scaling"
    runs, reference, expected_runs = extreme_tail_results(root)
    reference = use_pooled_reference(root, reference)
    suffix = SUFFIX + "_tilt9"
    intervals = list(bootstrap_rows(runs, reference["hits"], reference["samples"],
                                   expected_runs=expected_runs, pooled=reference.get("includes_matched_crude_pool", False)))
    write_csv(output / f"rare_event_bootstrap_uncertainty{suffix}.csv", intervals)
    budget_statistic_figure(output, suffix=suffix, label_tilt=True)
    budget_statistic_figure(output, relative_sd=True, suffix=suffix, label_tilt=True)
    print(f"Plotted {len(runs)} saved comparisons with reference {reference['hits']}/{reference['samples']}")


def plot_combined_tail_scaling(root):
    output = root / "camera_ready_low_particle_budget"
    latest, reference, expected_runs = extreme_tail_results(root)
    reference = use_pooled_reference(root, reference)
    pooled = reference.get("includes_matched_crude_pool", False)
    cohorts = (
        ("fixed_tilt", output / f"combined_compute_scaling_runs{SUFFIX}.csv", 24),
        ("ladder_n128", root / "camera_ready_same_run_logit" / f"combined_compute_scaling_runs{SUFFIX}_tilt7.csv", 12),
    )
    panels = []
    tail_intervals = []
    for series, path, count in cohorts:
        rows = []
        for r in read_csv(path):
            if float(r["threshold"]) != THRESHOLD:
                raise ValueError(f"Wrong threshold in {path}")
            rows.append(dict(N=int(r["N"]), **{"lambda": float(r["lambda"])}, seed=int(r["seed"]),
                             crude_n=int(r["crude_n"]), smc=float(r["smc"]), crude=float(r["crude"])))
        intervals = list(bootstrap_rows(rows, reference["hits"], reference["samples"], expected_runs=count, pooled=pooled))
        panels.append((intervals, series != "fixed_tilt"))
        tail_intervals.extend(dict(series=series, **r) for r in intervals)
    intervals = list(bootstrap_rows(latest, reference["hits"], reference["samples"],
                                   expected_runs=expected_runs, pooled=pooled))
    panels.append((intervals, True))
    tail_intervals.extend(dict(series="ladder_n256", **r) for r in intervals)

    suffix = "_with_tail"
    budget_statistic_figure(output, suffix=suffix, extra_panels=panels)
    budget_statistic_figure(output, relative_sd=True, suffix=suffix, extra_panels=panels)
    refs = {float(r["threshold"]): float(r["p_reference"])
            for r in read_csv(output / "combined_compute_scaling_summary.csv")}
    base_intervals = [dict(series="fixed_tilt", p_reference=refs[float(r["threshold"])], **r)
                      for r in read_csv(output / "rare_event_bootstrap_uncertainty.csv")]
    all_intervals = tail_intervals + base_intervals
    fields = list(tail_intervals[0])
    write_csv(output / f"rare_event_bootstrap_uncertainty{suffix}.csv",
              [{field: row.get(field) for field in fields} for row in all_intervals])
    print(f"Combined {len(all_intervals)} bootstrap summaries across four thresholds")


def plot_camera_ready_tail(root, output):
    runs, separate_reference, expected_runs = extreme_tail_results(root)
    reference = use_pooled_reference(root, separate_reference)
    if not reference.get("includes_matched_crude_pool"):
        raise ValueError("Build the audited pooled crude reference first")
    ref = reference["p_hat"]
    tilts = sorted({r["lambda"] for r in runs})
    groups = {tilt: sorted((r for r in runs if r["lambda"] == tilt), key=lambda r: r["seed"])
              for tilt in tilts}
    if any(len(g) != expected_runs or len({r["seed"] for r in g}) != expected_runs for g in groups.values()):
        raise ValueError("Expected one observation per independent seed at each tilt")
    minutes = {tilt: np.mean([r["smc_seconds"] for r in group]) / 60 for tilt, group in groups.items()}
    selected = (6.5, 9.)
    colors = {method: color for method, _, color, _ in METHODS}
    ref_color = "#ac392c"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        fig, (discovery, estimates) = plt.subplots(1, 2, figsize=(7.3, 3.35))
        x = [minutes[tilt] for tilt in tilts]
        for method, label in (("smc", "RollBack-SMC"), ("crude", "Crude MC")):
            hits = [sum(r[method] > 0 for r in groups[tilt]) for tilt in tilts]
            color = colors[method]
            lo, hi = np.array([wilson_interval(h, expected_runs) for h in hits]).T
            discovery.fill_between(x, lo, hi, color=color, alpha=.12, lw=0)
            discovery.plot(x, np.asarray(hits) / expected_runs, "-o", color=color, ms=2.9, lw=1.4, label=label)
            for tilt in selected:
                count = hits[tilts.index(tilt)]
                discovery.annotate(f"{count}/{expected_runs}", (minutes[tilt], count / expected_runs),
                                   xytext=(0, 7), textcoords="offset points", ha="center",
                                   color=color, fontsize=7, weight="bold")
        discovery.set_title("(a) Event discovery", loc="left", fontsize=10, weight="bold")
        discovery.set_xlabel("Mean GPU time per run (min)")
        discovery.set_ylabel("Runs with at least one hit")
        discovery.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        discovery.set_ylim(-.055, 1.02)
        discovery.set_xlim(min(x) - .6, max(x) + 1.5)

        lo, hi = reference["wilson_95"]
        estimates.axhspan(lo / ref, hi / ref, color=ref_color, alpha=.13, lw=0)
        estimates.axhline(1, color=ref_color, lw=.85)
        leading = []
        for i, tilt in enumerate(selected):
            rows = groups[tilt]
            for method, _, color, offset in METHODS:
                ratios = np.array([r[method] / ref for r in rows])
                center = i + offset
                positions = center + np.linspace(-.074, .074, len(rows))
                estimates.scatter(positions, ratios, s=9, color=color, alpha=.72, zorder=3)
                estimates.scatter(center + .105, np.mean(ratios), marker="D", s=23,
                                  color=color, edgecolor="white", linewidth=.45, zorder=5)
                estimates.scatter(center + .105, np.median(ratios), marker="s", s=20,
                                  facecolors="none", edgecolors=color, linewidth=.85, zorder=6)
                if method == "smc":
                    largest = int(np.argmax(ratios))
                    leading.append((rows[largest]["seed"], positions[largest], ratios[largest]))
        if leading[0][0] == leading[1][0]:
            _, start_x, start_y = leading[0]
            _, end_x, end_y = leading[1]
            estimates.plot((start_x, end_x), (start_y, end_y), color=colors["smc"], ls=":", lw=.9)
            estimates.annotate("Same leading seed", ((start_x + end_x) / 2, (start_y + end_y) / 2),
                               xytext=(0, 7), textcoords="offset points", ha="center", color=colors["smc"], fontsize=7)
        estimates.set_title("(b) Probability estimates", loc="left", fontsize=10, weight="bold")
        estimates.set_yscale("symlog", linthresh=1e-8)
        estimates.set_ylim(-3e-9, 9)
        estimates.set_xlim(-.41, 1.41)
        estimates.set_yticks([0, 1e-8, 1e-6, 1e-4, 1e-2, 1],
                            ["0", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", "1"])
        estimates.set_xticks(range(len(selected)), [rf"$\lambda={tilt:g}$" for tilt in selected])
        estimates.set_ylabel(r"$\widehat p_A / p_{\mathrm{ref}}$")
        for ax in (discovery, estimates):
            ax.grid(axis="y", color="#dce2e7", lw=.5)
            ax.set_axisbelow(True)
            ax.minorticks_off()
            ax.spines[["top", "right"]].set_visible(False)
        handles, _ = discovery.get_legend_handles_labels()
        handles.extend((
            Line2D([], [], marker="D", color="#26323c", linestyle="none", label="Mean"),
            Line2D([], [], marker="s", color="#26323c", markerfacecolor="none", linestyle="none", label="Median"),
            Patch(facecolor=ref_color, alpha=.13, label="Reference 95% interval"),
        ))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=5, frameon=False, fontsize=7)
        fig.subplots_adjust(left=.11, right=.983, top=.91, bottom=.26, wspace=.37)
        for extension in ("pdf", "png"):
            fig.savefig(output.with_suffix(f".{extension}"), dpi=300, facecolor="white")
        plt.close(fig)
    print(f"Plotted {len(runs)} comparisons for the camera-ready figure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Directory with the saved campaigns")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--saved-ladder", action="store_true", help="Use the 12 saved N128 logit ladders")
    mode.add_argument("--extreme-tail-scaling", action="store_true", help="Plot the completed N256 study through tilt 9")
    mode.add_argument("--combined-tail-scaling", action="store_true",
                      help="Add the saved 0.9995 studies to the combined comparison figures")
    mode.add_argument("--build-pooled-reference", type=Path, metavar="SEED_AUDIT",
                      help="Build the full reference using the original crude seed inventory")
    mode.add_argument("--camera-ready-figure", type=Path, metavar="OUTPUT",
                      help="Render the two-panel publication figure as PDF and PNG")
    parser.add_argument("--max-tilt", type=float, choices=tuple(k / 2 for k in range(5, 17)),
                        help="Last saved ladder checkpoint to include (default: 5)")
    args = parser.parse_args()
    if args.max_tilt is not None and not args.saved_ladder:
        parser.error("--max-tilt requires --saved-ladder")
    root = args.root
    if args.camera_ready_figure is not None:
        plot_camera_ready_tail(root, args.camera_ready_figure)
        return
    if args.build_pooled_reference is not None:
        build_pooled_reference(root, args.build_pooled_reference)
        return
    if args.combined_tail_scaling:
        plot_combined_tail_scaling(root)
        return
    if args.extreme_tail_scaling:
        plot_extreme_tail_scaling(root)
        return
    output = root / ("camera_ready_same_run_logit" if args.saved_ladder else "camera_ready_low_particle_budget")
    study, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_low_particle_budget.yaml")
    pool, pool_sources = validated_pool(root, study)
    other, other_sources = mc_pool(root / "camera_ready_gradual_budget", study)
    logit = root / "camera_ready_same_run_logit"
    policy = read_json(logit / "comparison_policy.json")
    held_out = read_json(logit / "reference_3_thresholds.json")
    actual_sources = [{**s, "file": Path(s["file"]).as_posix()} for s in other_sources]
    overlap = {s["seed"] for s in pool_sources} & {s["seed"] for s in other_sources}
    if actual_sources != policy["MC_sources"] or overlap:
        raise ValueError("Reference pool is not the verified, separate set of trajectories")
    reference = np.concatenate([other[start:end] for start, end in held_out["retained_index_ranges"]])
    if len(reference) != held_out["samples"] or int((reference >= .995).sum()) != held_out["hits"]:
        raise ValueError("Held-out reference fails its original threshold check")
    reference_hits = int((reference >= THRESHOLD).sum())
    if not reference_hits:
        raise ValueError("A positive held-out reference is required for normalized plots")
    chosen_reference = use_pooled_reference(root, dict(hits=reference_hits, samples=len(reference)))
    reference_hits, reference_samples = chosen_reference["hits"], chosen_reference["samples"]
    reference_low, reference_high = wilson_interval(reference_hits, reference_samples)
    if args.saved_ladder:
        max_tilt = args.max_tilt if args.max_tilt is not None else 5.
        levels = {k / 2 for k in range(5, int(2 * max_tilt) + 1)}
        runs = list(matched_ladder(root, other, levels))
        suffix = SUFFIX + f"_tilt{max_tilt:g}".replace(".", "p")
        expected_runs, expected_rows = 12, 12 * len(levels)
    else:
        original = [r for r in read_csv(output / "combined_compute_scaling_runs.csv")
                    if float(r["threshold"]) == .999 and float(r["lambda"]) == 2.5]
        runs = list(matched_runs(root, original, pool))
        suffix, expected_runs, expected_rows = SUFFIX, 24, 144
    if len(runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} saved comparisons")
    intervals = list(bootstrap_rows(runs, reference_hits, reference_samples, expected_runs=expected_runs,
                                   pooled=chosen_reference.get("includes_matched_crude_pool", False)))
    write_csv(output / f"combined_compute_scaling_runs{suffix}.csv", runs)
    write_csv(output / f"rare_event_bootstrap_uncertainty{suffix}.csv", intervals)
    budget_statistic_figure(output, suffix=suffix, label_tilt=args.saved_ladder)
    budget_statistic_figure(output, relative_sd=True, suffix=suffix, label_tilt=args.saved_ladder)
    print(f"Reference: {reference_hits}/{reference_samples}, Wilson 95% [{reference_low:.6g}, {reference_high:.6g}]")
    matched_pool = other if args.saved_ladder else pool
    print(f"Full matched MC pool: {int((matched_pool >= THRESHOLD).sum())}/{len(matched_pool)}")
    for n, tilt in sorted({(r["N"], r["lambda"]) for r in runs}):
        group = [r for r in runs if (r["N"], r["lambda"]) == (n, tilt)]
        for field in ("event_particles", "crude_hits"):
            print(f"N={n} tilt={tilt:g} {field}: {sum(r[field] for r in group)} hits, "
                  f"{sum(r[field] > 0 for r in group)} positive runs")


if __name__ == "__main__":
    main()
