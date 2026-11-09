"""Measure full native MC throughput and one fixed-tilt RollBack SMC run on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
from statistics import median

import torch

from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs
from diffusion_toxicity_smc_experiment.camera_ready_gpu import Context, crude
from diffusion_toxicity_smc_experiment.exploratory import audit_population
from diffusion_toxicity_smc_experiment.exploratory_population import population
from diffusion_toxicity_smc_experiment.same_run_reuse import native_clock_count
from diffusion_toxicity_smc_experiment.storage import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    torch.set_num_threads(cores)
    torch.set_num_interop_threads(1)
    study, configs = study_configs(args.study)
    config = next(config for config in configs if config["smc"]["num_particles"] == 64)
    config["runtime"]["device"] = "cpu"
    config["smc"]["tilted_lambdas"] = [1.]
    config["smc"]["tilted_sampler"]["num_tempering_levels"] = 2
    config["_exploration"]["schedule"] = [.5, 1.]
    study["thresholds"] = [.995, .999]
    context = Context(config)
    if context.device.type != "cpu":
        raise RuntimeError("Expected CPU for both SMC and crude MC")

    report = dict(host=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"],
                  cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_cores_requested=cores,
                  torch_threads=torch.get_num_threads(), torch_version=str(torch.__version__),
                  device=context.device.type, decoder_dtype=str(next(context.generator.model.parameters()).dtype),
                  classifier_dtype=str(next(context.evaluator_runtime.classifier.parameters()).dtype),
                  mc_trials=[], mc_warmup_paths=16, smc_particles=64, smc_tilts=[.5, 1.],
                  smc_moves_per_level=4)
    path = args.output / "timing.json"
    atomic_write_json(path, report)

    crude(context, 8126000000, study["thresholds"], count=16, policy="excluded_cpu_warmup")
    for i in range(3):
        trial = crude(context, 8126000001 + i, study["thresholds"], count=32,
                      policy="fixed_count_cpu_timing")
        if trial["n"] != 32 or trial["runtime_seconds"] <= 0:
            raise ValueError("Incomplete calibration batch")
        record = {key: trial[key] for key in ("seed", "n", "runtime_seconds", "native_transitions")}
        record["seconds_per_path"] = trial["runtime_seconds"] / trial["n"]
        report["mc_trials"].append(record)
        atomic_write_json(path, report)
        print(f"MC_CALIBRATION {record}", flush=True)

    rate = median(trial["seconds_per_path"] for trial in report["mc_trials"])
    report["mc_seconds_per_path"] = rate
    report["mc_paths_per_second"] = 1. / rate
    atomic_write_json(path, report)

    # Initialize CPU RollBack/MH paths before measuring the full SMC run.
    warm = {**config, "smc": {**config["smc"], "num_particles": 8}}
    population(context, warm, {**study, "record_population_checkpoints": False}, 8126000004)
    result = population(context, config, study, 8126000005)
    audit_population(result, config)
    elapsed = result["summary"]["runtime_seconds"]
    report["smc_runtime_seconds"] = elapsed
    report["smc_matched_mc_paths"] = native_clock_count(elapsed, rate)
    report["smc_native_transitions"] = result["summary"]["total_decoder_particle_steps"]
    report["smc_seed"] = result["summary"]["seed"]
    atomic_write_json(path, report)
    atomic_write_json(args.output / "smc_population.json", result)
    print(f"SMC_SECONDS={elapsed:.3f} MATCHED_MC_PATHS={report['smc_matched_mc_paths']}", flush=True)


if __name__ == "__main__":
    main()
