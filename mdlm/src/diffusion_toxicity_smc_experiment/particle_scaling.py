"""Compare terminal RollBack SMC populations with saved, time-matched crude draws."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .accuracy_budget import assigned_runs, study_configs
from .pipeline import audit_population
from .paper_analysis import checkpoint_log_estimates, select_populations, terminal_log_estimates
from .same_run_reuse import calibration_for, fixed_mc_block, mc_pool, native_clock_count, write_csv
from .storage import atomic_write_json


def validated_pool(output_root, study):
    pool, sources = mc_pool(output_root / study["mc_pool"], study)
    additional = study.get("mc_pool_additional")
    if additional:
        extra, extra_sources = mc_pool(output_root / additional, study)
        if {r["seed"] for r in sources} & {r["seed"] for r in extra_sources}:
            raise ValueError("Repeated crude random stream across source pools")
        pool = np.concatenate((pool, extra))
        sources = ([dict(r, pool=study["mc_pool"]) for r in sources]
                   + [dict(r, pool=additional) for r in extra_sources])
    digest = hashlib.sha256(pool.astype("<f8", copy=False).tobytes()).hexdigest()
    if len(pool) != study["mc_pool_samples"] or digest != study["mc_pool_scores_sha256"]:
        raise ValueError("Crude scores differ from the frozen independent pool")
    if not np.isfinite(pool).all() or np.any((pool < 0) | (pool > 1)):
        raise ValueError("Invalid score in the saved crude pool")
    return pool, sources


def log_relative_error(delta):
    """log10(1 + abs(exp(delta) - 1)), including log(0) = -inf."""
    if delta >= 0:
        return delta / math.log(10)
    return math.log1p(-math.expm1(delta)) / math.log(10)


def selected_populations(population, study, refs, *, sweeps_per_level=4, denoising_steps=32):
    return select_populations(population, study.get("checkpoint_tilts"), refs,
                              sweeps_per_level=sweeps_per_level, denoising_steps=denoising_steps)


def summarize(rows):
    results = []
    for n, tilt, threshold in sorted({(r["N"], r["lambda"], r["threshold"]) for r in rows}):
        group = [r for r in rows if (r["N"], r["lambda"], r["threshold"]) == (n, tilt, threshold)]
        ref = group[0]["p_reference"]
        for method in ("smc", "crude"):
            estimates = [r[method] for r in group]
            ratios = [r[f"{method}_ratio"] for r in group]
            sd = statistics.stdev(ratios) if len(group) > 1 else None
            results.append(dict(
                N=n, **{"lambda": tilt}, threshold=threshold, p_reference=ref,
                primary_tilt=group[0]["primary_tilt"], method=method, runs=len(group),
                zero_count=sum(r[f"log_{method}"] == -math.inf for r in group),
                mean=statistics.mean(estimates), median=statistics.median(estimates),
                mean_ratio=statistics.mean(ratios), median_ratio=statistics.median(ratios),
                relative_standard_deviation=sd, variance=None if sd is None else (ref * sd)**2,
                mean_log_relative_error=statistics.mean(r[f"{method}_log_relative_error"] for r in group),
                median_log_relative_error=statistics.median(r[f"{method}_log_relative_error"] for r in group),
                within_factor_ten=sum(.1 <= ratio <= 10 for ratio in ratios),
                mean_smc_seconds=statistics.mean(r["smc_seconds"] for r in group),
                mean_crude_n=statistics.mean(r["crude_n"] for r in group),
                crude_theory_rsd=math.sqrt(statistics.mean((1 - ref) / (ref * r["crude_n"]) for r in group)),
            ))
    return results


def comparison_rows(study, configs, root, pool):
    tasks = assigned_runs(study, configs, 0, 1)
    task_lookup = {seed: (config, i // len(configs)) for i, (config, seed) in enumerate(tasks)}
    refs = dict(zip(study["thresholds"], (hits / n for hits, n in zip(
        study["reference"]["discoveries"], study["reference"]["samples"], strict=True)), strict=True))
    rows, seen = [], set()
    for file in sorted(root.glob("workers/worker_*/populations/*.json")):
        population = json.loads(file.read_bytes())
        summary = population["summary"]
        seed, n, tilt = summary["seed"], summary["num_particles"], summary["lambda"]
        config, replicate = task_lookup[seed]
        case = config["_exploration"]
        if seed in seen or n != case["N"] or tilt != case["schedule"][-1]:
            raise ValueError("Duplicate seed or mismatched saved configuration")
        if population["phi"]["name"] != study["phi"] or not population["level_diagnostics"][-1]["terminal"]:
            raise ValueError("Wrong potential or nonterminal population")
        seen.add(seed)
        audit_population(population, config)
        _, calibration = calibration_for(root, population)
        for state, events in selected_populations(population, study, refs):
            seconds, tilt = state["seconds"], state["tilt"]
            crude_n = native_clock_count(seconds, calibration["seconds_per_path"])
            prefix, start, _ = fixed_mc_block(pool, seed_rank=replicate,
                                            total_seeds=study["independent_runs"], required=crude_n)
            cp = state["checkpoint"]
            for threshold, ref in events.items():
                if cp is None:
                    smc, snis, event_particles = terminal_log_estimates(population, threshold)
                    corrections = population["corrections"]
                else:
                    smc, snis, event_particles = checkpoint_log_estimates(cp, threshold)
                    corrections = cp["corrections"]
                if study.get("paper_final_scaling") and threshold not in study["recorded_thresholds"]:
                    # Reverse ESS depends on weights and rewards, independent of the event threshold.
                    reverse_ess = corrections[str(study["recorded_thresholds"][0])]["reverse_ess_fraction"]
                    saved = None
                else:
                    saved = corrections[str(threshold)]
                    reverse_ess = saved["reverse_ess_fraction"]
                crude_hits = int((prefix >= threshold).sum())
                crude = math.log(crude_hits / crude_n) if crude_hits else -math.inf
                if saved is not None and (saved["event_particles"] != event_particles or any(
                    not math.isclose(math.exp(v), saved[m], rel_tol=1e-12)
                    for m, v in (("smc", smc), ("snis", snis))
                )):
                    raise ValueError("Log-space reconstruction differs from the saved estimators")
                row = dict(configuration=case["id"], seed=seed, replicate=replicate, N=n,
                           **{"lambda": tilt}, threshold=threshold, p_reference=ref,
                           primary_tilt=bool(study.get("checkpoint_tilts")) or tilt == study["target_tilts"][threshold],
                           event_particles=event_particles, crude_hits=crude_hits, crude_n=crude_n,
                           crude_pool_start=start, smc_seconds=seconds,
                           crude_calibration_seconds_per_path=calibration["seconds_per_path"],
                           smc_native_transitions=state["native"], smc_acceptance=state["acceptance"],
                           smc_ess=state["ess"], reverse_ess_fraction=reverse_ess,
                           distinct_root_ancestors=state["roots"])
                if study.get("checkpoint_tilts"):
                    exact_n = state["native"] // 33
                    exact_prefix, _, _ = fixed_mc_block(pool, seed_rank=replicate,
                                                        total_seeds=study["independent_runs"], required=exact_n)
                    row.update(crude_native_n=exact_n, crude_native_hits=int((exact_prefix >= threshold).sum()),
                               fraction_one_expected_crude_hit=crude_n * ref)
                for method, log_estimate in (("smc", smc), ("snis", snis), ("crude", crude)):
                    delta = log_estimate - math.log(ref)
                    row.update({f"log_{method}": log_estimate, method: math.exp(log_estimate),
                                f"{method}_ratio": math.exp(delta),
                                f"{method}_log_relative_error": log_relative_error(delta)})
                rows.append(row)

    return rows, dict(completed_runs=len(seen), expected_runs=len(tasks), complete=len(seen) == len(tasks))


def report(study, configs, output_root, pool, sources):
    root = output_root / study["output_subdirectory"]
    rows, status = comparison_rows(study, configs, root, pool)
    if rows:
        write_csv(root / "runs.csv", rows)
        write_csv(root / "summary.csv", summarize(rows))
        atomic_write_json(root / "comparison_policy.json", dict(
            MC_sources=sources, MC_pool_samples=len(pool), MC_new_draws=0,
            MC_budget="floor(synchronized SMC seconds / calibrated MC seconds per trajectory)",
            MC_blocks="disjoint across replicate ranks, nested prefixes reused across configurations",
            MC_calibration_reused_on_same_physical_GPU=True, reference=study["reference"],
            target_tilts=study.get("checkpoint_tilts", study["target_tilts"]),
            original_job_for_calibration="2079904"))
    atomic_write_json(root / "report_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify-pool", "report"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    study, configs = study_configs(args.study)
    pool, sources = validated_pool(args.output_root, study)
    if args.stage == "verify-pool":
        print(f"Verified {len(pool)} saved trajectories in {len(sources)} files")
    else:
        status = report(study, configs, args.output_root, pool, sources)
        print(json.dumps(status))
        if not status["complete"]:
            raise SystemExit("Incomplete campaign. Completed populations are saved for resume.")


if __name__ == "__main__":
    main()
