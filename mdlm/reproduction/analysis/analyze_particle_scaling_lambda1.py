"""Compare checkpoints at 0.5, 1 and 2.5 from the same completed ladders."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from plot_particle_scaling import boxplots, group_results, read_csv, relative_sd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from diffusion_toxicity_smc_experiment.particle_scaling import log_relative_error, summarize, validated_pool
from diffusion_toxicity_smc_experiment.same_run_reuse import write_csv


EVENTS = (0.99, 0.995)
CHECKPOINT_TILTS = (0.5, 1.0)
THRESHOLD_TOLERANCE = 2 * np.finfo(np.float32).eps


def checkpoint(population, tilt):
    levels = {row["eta"]: row for row in population["level_diagnostics"]}
    current, following = levels[tilt], levels[tilt + 0.5]
    # The incoming state at the next level follows the last MH move at this tilt.
    weights = np.asarray(following["previous_normalized_log_weights"], dtype=np.float64)
    scores = np.asarray(following["rewards_before_mutation"], dtype=np.float64)
    normalizer = following["log_z_before_weight"]
    if (current["log_normalizer_estimate"] != normalizer or
            not math.isclose(float(np.logaddexp.reduce(weights)), 0, abs_tol=1e-12) or
            len(weights) != len(scores) or len(weights) != population["summary"]["num_particles"]):
        raise ValueError(f"Invalid saved population at lambda={tilt:g}")
    if not math.isclose(math.exp(-float(np.logaddexp.reduce(2 * weights))),
                        current["ess_after_resampling"], rel_tol=1e-12):
        raise ValueError("Recovered ESS does not match the level diagnostics")
    return current, weights, scores, normalizer


def verify_historical_timing(root, tilt):
    """Check the cost proxy on earlier logit runs that recorded both clocks."""
    errors = []
    for file in sorted(root.glob("workers/worker_*/populations/*.json")):
        pop = json.loads(file.read_bytes())
        checkpoints = {row["lambda"]: row for row in pop["population_checkpoints"] if row["sweep"] == 4}
        early, late = checkpoints[tilt], checkpoints[2.5]
        _, weights, scores, normalizer = checkpoint(pop, tilt)
        if (not np.array_equal(weights, early["normalized_log_weights"]) or
                not np.array_equal(scores, early["rewards"]) or normalizer != early["log_z_hat"]):
            raise ValueError("Offline reconstruction differs from the recorded checkpoint")
        estimated = (late["runtime_seconds_cumulative"] * early["native_transitions_cumulative"]
                     / late["native_transitions_cumulative"])
        errors.append(estimated / early["runtime_seconds_cumulative"] - 1)
    if len(errors) != 12:
        raise ValueError("Expected 12 historical runs with measured checkpoint times")
    return errors


def estimate_fields(method, log_estimate, reference):
    delta = log_estimate - math.log(reference)
    return {
        f"log_{method}": log_estimate,
        method: math.exp(log_estimate),
        f"{method}_ratio": math.exp(delta),
        f"{method}_log_relative_error": log_relative_error(delta),
    }


def reconstruct(root, original, tilt):
    study = json.loads((root / "workers/worker_0/manifest.json").read_bytes())["study"]
    pool, sources = validated_pool(root.parent, study)
    policy = json.loads((root / "comparison_policy.json").read_bytes())
    canonical = lambda rows: [dict(row, file=row["file"].replace("\\", "/")) for row in rows]
    if canonical(sources) != canonical(policy["MC_sources"]):
        raise ValueError("Saved crude pool differs from the completed campaign")
    originals = {(int(r["seed"]), float(r["threshold"])): r for r in original
                 if float(r["lambda"]) == 2.5 and float(r["threshold"]) in EVENTS}
    minimum_distances = {a: math.inf for a in EVENTS}
    rows = []
    for file in sorted(root.glob("workers/worker_*/populations/N*_L25_seed*.json")):
        raw = file.read_bytes()
        pop = json.loads(raw)
        terminal = pop["summary"]
        if pop["phi"]["name"] != "logit" or terminal["lambda"] != 2.5:
            raise ValueError("Expected a completed logit ladder to lambda=2.5")
        n, seed = terminal["num_particles"], terminal["seed"]
        current, weights, phi, log_z = checkpoint(pop, tilt)
        q = np.exp(-np.logaddexp(0, -phi))
        reverse = weights - tilt * phi
        reverse_mass = float(np.logaddexp.reduce(reverse))
        moves = pop["mutation_diagnostics"]
        first_moves = [row for row in moves if row["level"] <= current["level"]]
        if len(first_moves) != 4 * current["level"] or any(row["proposals"] != n for row in first_moves):
            raise ValueError(f"Incomplete lambda={tilt:g} transition ledger")
        native = 33 * n + sum(row["rollback_particle_steps"] + row["proposals"] for row in first_moves)
        full_native = 33 * n + sum(row["rollback_particle_steps"] + row["proposals"] for row in moves)
        seconds = terminal["runtime_seconds"] * native / full_native
        for threshold in EVENTS:
            r = originals[seed, threshold]
            if (full_native != int(r["smc_native_transitions"]) or
                    terminal["runtime_seconds"] != float(r["smc_seconds"])):
                raise ValueError("Full ladder differs from the original cost records")
            closest = float(np.min(np.abs(q - threshold)))
            minimum_distances[threshold] = min(minimum_distances[threshold], closest)
            if closest <= THRESHOLD_TOLERANCE:
                raise ValueError("An intermediate classifier score requires an original-score recheck")
            event = q >= threshold
            numerator = float(np.logaddexp.reduce(reverse[event], initial=-np.inf))
            start, ref = int(r["crude_pool_start"]), float(r["p_reference"])
            cost = float(r["crude_calibration_seconds_per_path"])
            count = math.floor(seconds / cost)
            exact_count = native // 33
            if not 0 < min(count, exact_count) <= max(count, exact_count) <= int(r["crude_n"]):
                raise ValueError("Checkpoint MC prefix falls outside the original budget")
            hits = int(np.count_nonzero(pool[start:start + count] >= threshold))
            exact_hits = int(np.count_nonzero(pool[start:start + exact_count] >= threshold))
            row = dict(
                configuration=r["configuration"], seed=seed, replicate=int(r["replicate"]), N=n,
                **{"lambda": tilt}, threshold=threshold, p_reference=ref, primary_tilt=False,
                parent_lambda_final=2.5, checkpoint_resampled=current["resampled"],
                log_z_hat=log_z, smc_ess=current["ess_after_resampling"],
                event_particles=int(event.sum()), crude_hits=hits, crude_n=count, crude_pool_start=start,
                smc_seconds=seconds, smc_seconds_measured=False,
                smc_terminal_seconds=terminal["runtime_seconds"],
                crude_calibration_seconds_per_path=cost, smc_native_transitions=native,
                smc_terminal_native_transitions=full_native,
                crude_native_hits=exact_hits, crude_native_n=exact_count,
                population_source=file.relative_to(root).as_posix(),
                population_sha256=hashlib.sha256(raw).hexdigest(),
            )
            row.update(estimate_fields("smc", log_z + numerator, ref))
            row.update(estimate_fields("snis", numerator - reverse_mass, ref))
            row.update(estimate_fields("crude", math.log(hits / count) if hits else -math.inf, ref))
            row.update(estimate_fields("crude_native", math.log(exact_hits / exact_count) if exact_hits else -math.inf, ref))
            rows.append(row)
    if len(rows) != len(originals) or len({(r["seed"], r["threshold"]) for r in rows}) != 192:
        raise ValueError(f"Missing or duplicated lambda={tilt:g} estimates")
    return rows, minimum_distances


def summarize_checkpoint(rows):
    summary = summarize(rows)
    native_rows = [dict(row, crude=row["crude_native"], crude_n=row["crude_native_n"],
                        log_crude=row["log_crude_native"], crude_ratio=row["crude_native_ratio"],
                        crude_log_relative_error=row["crude_native_log_relative_error"])
                   for row in rows]
    summary += [dict(row, method="crude_native") for row in summarize(native_rows) if row["method"] == "crude"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    root = args.results
    # Exclude the distinct runs that stopped at 0.5 from this same-ladder comparison.
    terminal = [r for r in read_csv(root / "runs.csv") if float(r["lambda"]) == 2.5]
    terminal_summary = [r for r in read_csv(root / "summary.csv") if float(r["lambda"]) == 2.5]
    comparisons = tuple((event, tilt) for event in EVENTS for tilt in (*CHECKPOINT_TILTS, 2.5))
    checkpoint_rows, checkpoint_summaries, audit = [], [], {}
    for tilt in CHECKPOINT_TILTS:
        timing_errors = verify_historical_timing(root.parent / "camera_ready_same_run_logit", tilt)
        rows, distances = reconstruct(root, terminal, tilt)
        summary = summarize_checkpoint(rows)
        stem = f"lambda{tilt:g}".replace(".", "p")
        write_csv(root / f"{stem}_runs.csv", rows)
        write_csv(root / f"{stem}_summary.csv", summary)
        checkpoint_rows.extend(rows)
        checkpoint_summaries.extend(summary)
        audit[tilt] = dict(
            checkpoint_runs=len(rows) // len(EVENTS), event_estimates=len(rows),
            historical_clock_validation_runs=len(timing_errors),
            historical_clock_relative_error_median=statistics.median(timing_errors),
            historical_clock_relative_error_max_absolute=max(abs(x) for x in timing_errors),
            nearest_score_distance=distances,
        )
    groups, summaries = group_results(terminal + checkpoint_rows, terminal_summary + checkpoint_summaries, comparisons)
    for event, tilt, n in groups:
        terminal_group = groups[event, 2.5, n]
        identity = lambda rows: {(int(r["seed"]), int(r["replicate"]), int(r["crude_pool_start"])) for r in rows}
        if identity(groups[event, tilt, n]) != identity(terminal_group):
            raise ValueError("Tilt comparison does not use the same seeds and crude blocks")
    options = dict(comparisons=comparisons, prefix="tilt_comparison_", estimated_tilts=CHECKPOINT_TILTS)
    boxplots(root, groups, summaries, "ratio", **options)
    boxplots(root, groups, summaries, "log_relative_error", **options)
    relative_sd(root, summaries, **options)
    print(json.dumps(dict(checkpoints=audit, new_simulations=False), indent=2))


if __name__ == "__main__":
    main()
