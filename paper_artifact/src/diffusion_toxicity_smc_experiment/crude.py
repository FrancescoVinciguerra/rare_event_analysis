"""Native crude score bundles and independent timing calibration."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import platform
import socket
import time
from statistics import median

import numpy as np
import torch

from .configuration import decoder_id, evaluator_id, fingerprint
from .storage import atomic_write_json, file_hash, initialize_manifest, read_json


@torch.inference_mode()
def sample_batch(context, seed, thresholds, *, count, policy="matched_transitions", target_seconds=None):
    """Draw and score a fixed-size batch of independent native trajectories."""
    if count < 1:
        raise ValueError("Empty crude budget")
    generator = torch.Generator(device=context.device.type).manual_seed(int(seed))
    functions = context.functions
    decoder = context.config["decoder"]
    smc = context.config["smc"]

    context.sampler.reset_nfe()
    context.synchronize()
    started = time.perf_counter()
    paths, _ = context.sampler.sample_trajectory(
        batch_size=count, sequence_length=decoder["sequence_length"],
        num_steps=decoder["denoising_steps"], fixed_prefix_token_ids=None, device=context.device,
        generator=generator, microbatch_size=smc["microbatch_size"],
    )
    functions.reward(paths[:, -1])
    functions.event(paths[:, -1])
    scores = [float(row["toxic_probability"]) for row in functions.last_event_rows]
    estimates = {str(q): sum(score >= q for score in scores) / len(scores) for q in thresholds}
    context.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "scores": scores, "seed": int(seed), "n": len(scores),
        "native_transitions": (decoder["denoising_steps"] + 1) * len(scores),
        "runtime_seconds": elapsed, "nfe": context.sampler.nfe, "estimates": estimates,
        "target_seconds": target_seconds,
        "overshoot_seconds": max(0.0, elapsed - target_seconds) if target_seconds is not None else None,
        "policy": policy, "sample_count_chosen_before_generation": True,
    }


def native_clock_count(seconds, seconds_per_path):
    if seconds <= 0 or seconds_per_path <= 0:
        raise ValueError("Invalid measured cost")
    return max(1, math.floor(seconds / seconds_per_path))


def fixed_mc_block(scores, *, seed_rank, total_seeds, required):
    """Blocks fixed by planned seed rank, never by successes or available SMC runs."""
    size = len(scores) // total_seeds
    if not 0 <= seed_rank < total_seeds or not 1 <= required <= size:
        raise ValueError("Saved independent MC pool does not cover the fixed seed block")
    start = seed_rank * size
    return scores[start:start + required], start, size


@dataclass(frozen=True)
class CrudeData:
    matched_scores: np.ndarray
    samples: int
    hits: dict
    cohorts: list
    manifest_sha256: str


def planned_batches(settings):
    mc = settings["mc"]
    size = mc["batch_size"]
    if type(size) is not int or size < 1:
        raise ValueError("MC batch_size must be a positive integer")
    batches = []
    for cohort in mc["cohorts"]:
        for index, offset in enumerate(range(0, cohort["samples"], size)):
            batches.append({"cohort": cohort["name"], "seed": cohort["seed_start"] + index,
                            "samples": min(size, cohort["samples"] - offset), "matched": cohort["matched"]})
    return batches


def check_batches(settings, plan, batches):
    expected = [(c["name"], c["samples"], c["matched"]) for c in settings["mc"]["cohorts"]]
    totals = {c: 0 for c, _, _ in expected}
    match = {c: matched for c, _, matched in expected}
    if len(totals) != len(expected) or not any(match.values()):
        raise ValueError("Declare distinct crude cohorts including a matched cohort")
    seeds = set()
    for batch in batches:
        cohort, seed, count = (batch[key] for key in ("cohort", "seed", "samples"))
        if (cohort not in totals or batch["matched"] != match[cohort] or type(count) is not int or count < 1
                or type(seed) is not int or not 0 <= seed < 2**63 or seed in seeds):
            raise ValueError(f"Invalid or repeated crude batch: {batch}")
        seeds.add(seed)
        totals[cohort] += count
    if list(totals.items()) != [(name, count) for name, count, _ in expected]:
        raise ValueError("Crude cohort sample totals differ from the YAML")
    calibration = settings["mc"]["calibration"]
    if any(type(calibration[key]) is not int or calibration[key] <= 0
           for key in ("paths", "repetitions", "warmup_paths")):
        raise ValueError("Calibration sizes and repetitions must be positive integers")
    start = calibration["seed_start"]
    timing_seeds = [*range(start, start + 1 + calibration["repetitions"]), calibration["smc_warmup_seed"]]
    if (any(type(seed) is not int or not 0 <= seed < 2**63 for seed in timing_seeds)
            or len({run.seed for run in plan} | seeds | set(timing_seeds)) != len(plan) + len(seeds) + len(timing_seeds)):
        raise ValueError("SMC, crude and calibration require disjoint seeds")


def batch_path(root, batch):
    return root / "batches" / f"seed{batch['seed']}.json"


def batch_scores(data, batch):
    scores = np.asarray(data["scores"], dtype="<f8")
    if (data["seed"] != batch["seed"] or data["n"] != batch["samples"]
            or scores.shape != (batch["samples"],) or data["sample_count_chosen_before_generation"] is not True
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError(f"Invalid crude scores for seed {batch['seed']}")
    return scores


def model_ids(config):
    return {"decoder_id": decoder_id(config), "evaluator_id": evaluator_id(config)}


def generate_crude(batches, root, context):
    protocol = {**model_ids(context.config), "batches": batches, "protocol": "native_trajectory_scores_v1"}
    initialize_manifest(root, protocol)
    manifest_path = root / "scores_manifest.json"
    if manifest_path.exists():
        return
    files = {}
    for batch in batches:
        path = batch_path(root, batch)
        if path.exists():
            batch_scores(read_json(path), batch)
        else:
            data = sample_batch(context, batch["seed"], (), count=batch["samples"], policy="fixed_size_reference")
            batch_scores(data, batch)
            atomic_write_json(path, data)
        files[str(batch["seed"])] = file_hash(path)
    atomic_write_json(manifest_path, {**protocol, "files_sha256": files})


def load_crude(settings, plan, root, config):
    manifest_path = root / "scores_manifest.json"
    manifest = read_json(manifest_path)
    if any(manifest[key] != value for key, value in model_ids(config).items()):
        raise ValueError("Crude scores belong to a different decoder or evaluator")
    batches = manifest["batches"]
    check_batches(settings, plan, batches)
    if (set(manifest["files_sha256"]) != {str(b["seed"]) for b in batches}
            or {p.name for p in (root / "batches").glob("*.json")} != {batch_path(root, b).name for b in batches}):
        raise ValueError("The crude inventory differs from its manifest")
    thresholds = sorted({row["threshold"] for row in settings["ladder"]["observe"]})
    counts = {c["name"]: dict(name=c["name"], samples=c["samples"], hits=[0] * len(thresholds))
              for c in settings["mc"]["cohorts"]}
    matched = []
    for batch in batches:
        path = batch_path(root, batch)
        if file_hash(path) != manifest["files_sha256"][str(batch["seed"])]:
            raise ValueError(f"Crude score hash mismatch: {path}")
        scores = batch_scores(read_json(path), batch)
        hits = counts[batch["cohort"]]["hits"]
        for i, threshold in enumerate(thresholds):
            hits[i] += int(np.count_nonzero(scores >= threshold))
        if batch["matched"]:
            matched.append(scores)
    sample_count = sum(c["samples"] for c in counts.values())
    return CrudeData(np.concatenate(matched), sample_count,
                     {q: sum(c["hits"][i] for c in counts.values()) for i, q in enumerate(thresholds)},
                     list(counts.values()), file_hash(manifest_path))


def calibrate(settings, context):
    calibration = settings["mc"]["calibration"]
    start = calibration["seed_start"]
    sample_batch(context, start, (), count=calibration["warmup_paths"], policy="excluded_warmup")
    measurements = []
    for seed in range(start + 1, start + 1 + calibration["repetitions"]):
        draw = sample_batch(context, seed, (), count=calibration["paths"], policy="excluded_calibration")
        measurements.append({key: draw[key] for key in ("seed", "n", "runtime_seconds")})
    seconds_per_path = median(row["runtime_seconds"] / row["n"] for row in measurements)
    if not math.isfinite(seconds_per_path) or seconds_per_path <= 0:
        raise ValueError("Invalid independent crude timing calibration")
    device = {"hostname": socket.gethostname(), "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
              "torch": str(torch.__version__), "device": str(context.device)}
    if context.device.type == "cuda":
        gpu = torch.cuda.get_device_properties(context.device)
        device.update(gpu=gpu.name, cuda=torch.version.cuda, device_index=torch.cuda.current_device(),
                      visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return {"origin": "measured", "group": "measured_" + fingerprint(device)[:12],
            "hardware": device, "seconds_per_path": seconds_per_path,
            "measurements": measurements, "excluded_from_estimates_and_measured_run_times": True}
