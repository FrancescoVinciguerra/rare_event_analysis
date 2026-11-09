"""Compare saved independent SMC averages with single runs at matched GPU cost."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT / "src"))

from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs
from diffusion_toxicity_smc_experiment.crude_tail_reference import validated_scores
from diffusion_toxicity_smc_experiment.particle_scaling import (
    checkpoint_log_estimates, log_relative_error, terminal_log_estimates, validated_pool,
)
from diffusion_toxicity_smc_experiment.same_run_reuse import (
    calibration_for, fixed_mc_block, native_clock_count, write_csv,
)


CASES = ((.995, 1., 32), (.995, 1., 48), (.999, 2.5, 128))
METHODS = ("smc_pair", "smc_large", "crude_pair", "crude_large")
SEED = 995999
MAX_BUDGET_GAP = .03


def read_json(path):
    return json.loads(path.read_bytes())


def crude_pool(root):
    extreme, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_extreme_tail_scaling.yaml")
    pilot, _ = study_configs(EXPERIMENT / "configs/mdlm_owt_low_particle_budget.yaml")
    recent = root / "camera_ready_crude_tail_reference"
    manifest = read_json(recent / "manifest.json")
    status = read_json(recent / "report_status.json")
    reported = read_json(recent / "reference.json")
    if not status["complete"] or any(
        manifest[key] != extreme["reference"][key] for key in ("decoder_id", "evaluator_id")
    ):
        raise ValueError("The recent MC campaign is incomplete or uses another model")
    streams, chunks = set(), []
    for config in (extreme, pilot):
        scores, sources = validated_pool(root, config)
        seeds = {entry["seed"] for entry in sources}
        if seeds & streams:
            raise ValueError("Duplicate MC random stream")
        streams.update(seeds)
        chunks.append(scores)
    recent_chunks = []
    for worker, batches in enumerate(manifest["planned_batches"]):
        for seed, n in batches:
            if seed in streams:
                raise ValueError("Duplicate MC random stream")
            streams.add(seed)
            path = recent / f"workers/worker_{worker}/baselines/seed{seed}.json"
            recent_chunks.append(validated_scores(read_json(path), seed, n))
    scores = np.concatenate(recent_chunks)
    if len(scores) != reported["samples"] or (scores >= reported["threshold"]).sum() != reported["hits"]:
        raise ValueError("The recent MC pool does not match its report")
    return np.random.default_rng(SEED).permutation(np.concatenate([*chunks, scores]))


def source_rows(root):
    path = root / "camera_ready_low_particle_budget/combined_compute_scaling_runs.csv"
    groups = defaultdict(list)
    keys = {(q, tilt, size) for q, tilt, n in CASES for size in (n, 2 * n)}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = float(row["threshold"]), float(row["lambda"]), int(row["N"])
            if key in keys:
                groups[key].append(row)
    for key in keys:
        rows = sorted(groups[key], key=lambda row: int(row["seed"]))
        if len(rows) != 24 or len({row["seed"] for row in rows}) != 24:
            raise ValueError(f"Expected 24 distinct runs for {key}")
        groups[key] = [audit_source(root, key, row) for row in rows]
    return groups


def audit_source(root, key, row):
    q, tilt, n = key
    seed = int(row["seed"])
    folder = root / row["campaign"]
    path, = folder.glob(f"workers/worker_*/populations/*seed{seed}.json")
    data = read_json(path)
    summary = data["summary"]
    level, = (level for level in data["level_diagnostics"] if level["eta"] == tilt)
    if level["terminal"]:
        estimate, _, hits = terminal_log_estimates(data, q)
        seconds = summary["runtime_seconds"]
    else:
        cp, = (cp for cp in data["population_checkpoints"] if cp["lambda"] == tilt and cp["sweep"] == 4)
        estimate, _, hits = checkpoint_log_estimates(cp, q)
        seconds = cp["runtime_seconds_cumulative"]
    if (row["smc_seconds_measured"] != "True" or summary["seed"] != seed
            or summary["num_particles"] != n or data["phi"]["name"] != "logit"
            or not math.isclose(seconds, float(row["smc_seconds"]), rel_tol=1e-12)
            or not math.isclose(estimate, float(row["log_smc"]), rel_tol=1e-12)
            or hits != int(row["event_particles"])):
        raise ValueError(f"Saved estimate or clock could not be reproduced for seed {seed}")
    _, calibration = calibration_for(folder, data)
    rate = calibration["seconds_per_path"]
    if native_clock_count(seconds, rate) != int(row["crude_n"]):
        raise ValueError(f"Saved MC calibration could not be reproduced for seed {seed}")
    return dict(seed=seed, seconds=seconds, rate=rate, log_p=estimate, hits=hits,
                terminal=level["terminal"], resampled=level["resampled"], ref=float(row["p_reference"]))


def matched_controls(pair_time, large_time):
    gap = np.abs(pair_time[:, None] / large_time - 1)
    # Selection only uses measured time. Large-run controls are never reused.
    pairs, controls = linear_sum_assignment(np.where(gap <= MAX_BUDGET_GAP, gap, 1e6))
    if np.any(gap[pairs, controls] > MAX_BUDGET_GAP):
        return None
    return controls, gap[pairs, controls]


def compare(groups, scores):
    result = []
    for case_index, (q, tilt, n) in enumerate(CASES):
        small, large = groups[q, tilt, n], groups[q, tilt, 2 * n]
        if {r["seed"] for r in small} & {r["seed"] for r in large}:
            raise ValueError("The two SMC cohorts must be independent")
        if len({(r["terminal"], r["ref"]) for r in [*small, *large]}) != 1:
            raise ValueError("Different checkpoint policies or references within one comparison")
        pairs = list(zip(small[::2], small[1::2], strict=True))
        pair_time = np.array([a["seconds"] + b["seconds"] for a, b in pairs])
        large_time = np.array([r["seconds"] for r in large])
        matched = matched_controls(pair_time, large_time)
        if matched is None:
            raise ValueError("Cannot match all SMC pairs within the 3% clock tolerance")
        controls, gaps = matched
        for i, j in enumerate(controls):
            a, b = pairs[i]
            control = large[j]
            ref = control["ref"]
            n_pair = math.floor(a["seconds"] / a["rate"] + b["seconds"] / b["rate"])
            n_large = native_clock_count(control["seconds"], control["rate"])
            prefix, start, _ = fixed_mc_block(scores, seed_rank=case_index * len(pairs) + int(i),
                                             total_seeds=len(CASES) * len(pairs), required=max(n_pair, n_large))
            hits_pair, hits_large = (int(np.count_nonzero(prefix[:count] >= q)) for count in (n_pair, n_large))
            log_pair = float(np.logaddexp(a["log_p"], b["log_p"])) - math.log(2)
            record = dict(threshold=q, tilt=tilt, small_n=n, pair=int(i) + 1, p_reference=ref,
                          terminal=control["terminal"], seed_a=a["seed"], seed_b=b["seed"],
                          seed_large=control["seed"], smc_pair_seconds=pair_time[i],
                          smc_large_seconds=large_time[j], relative_budget_gap=gaps[i],
                          smc_pair_hits=a["hits"] + b["hits"], smc_large_hits=control["hits"],
                          a_resampled=a["resampled"], b_resampled=b["resampled"],
                          large_resampled=control["resampled"], crude_pool_start=start,
                          crude_pair_n=n_pair, crude_large_n=n_large, crude_pair_hits=hits_pair,
                          crude_large_hits=hits_large)
            for method, log_estimate in zip(METHODS, (log_pair, control["log_p"],
                    math.log(hits_pair / n_pair) if hits_pair else -math.inf,
                    math.log(hits_large / n_large) if hits_large else -math.inf), strict=True):
                delta = log_estimate - math.log(ref)
                record.update({method: math.exp(log_estimate), f"{method}_ratio": math.exp(delta),
                               f"{method}_log_error": log_relative_error(delta)})
            result.append(record)
    return result


def summarize(rows):
    result = []
    rng = np.random.default_rng(SEED + 1)
    for q, tilt, n in CASES:
        group = [r for r in rows if (r["threshold"], r["tilt"], r["small_n"]) == (q, tilt, n)]
        draws = rng.integers(len(group), size=(20000, len(group)))
        for method in METHODS:
            budget = method.rsplit("_", 1)[-1]
            ratios = np.array([r[f"{method}_ratio"] for r in group])
            errors = [r[f"{method}_log_error"] for r in group]
            positive = ratios[ratios > 0]
            lo, hi = np.quantile(np.std(ratios[draws], axis=1, ddof=1), [.025, .975])
            row = dict(threshold=q, tilt=tilt, small_n=n, method=method, estimates=len(group),
                       positive=len(positive), zeros=len(group) - len(positive),
                       mean_gpu_seconds=np.mean([r[f"smc_{budget}_seconds"] for r in group]),
                       mean_crude_n=np.mean([r[f"crude_{budget}_n"] for r in group]),
                       mean_ratio=ratios.mean(), median_ratio=np.median(ratios),
                       median_positive_ratio=np.median(positive) if len(positive) else None,
                       relative_sd=ratios.std(ddof=1), relative_sd_lower_95=lo, relative_sd_upper_95=hi,
                       median_log_error=np.median(errors), mean_log_error=np.mean(errors),
                       within_factor_ten=int(np.count_nonzero((ratios >= .1) & (ratios <= 10))),
                       max_budget_gap=max(r["relative_budget_gap"] for r in group))
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT / "artifacts/current")
    args = parser.parse_args()
    scores = crude_pool(args.root)
    rows = compare(source_rows(args.root), scores)
    summary = summarize(rows)
    output = args.root / "camera_ready_low_particle_budget"
    write_csv(output / "paired_runs_matched_gpu.csv", rows)
    write_csv(output / "paired_runs_summary.csv", summary)
    sys.path.insert(0, str(EXPERIMENT / "reproduction/plots"))
    from plot_averaged_runs import plot

    plot(output / "plots/paired_runs_matched_gpu.png", rows)
    print(f"{len(rows)} matched comparisons, {len(scores)} saved MC scores")


if __name__ == "__main__":
    main()
