"""Configuration, run identity and persistence for the portable paper driver."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import yaml

from .configuration import decoder_id, evaluator_id, fingerprint
from .pipeline import audit_population, component_config
from .paper_analysis import checkpoint_targets, select_populations
from .storage import atomic_write_json, file_hash, read_json


@dataclass(frozen=True)
class Run:
    particles: int
    replicate: int
    seed: int

    @property
    def filename(self):
        return f"N{self.particles}_seed{self.seed}.json"


def scientific_sources():
    names = ("ancestral", "configuration", "evaluation", "exploratory_kernel", "exploratory_population",
             "pipeline", "population_checkpoints", "potentials", "runtime", "smc")
    return {name: file_hash(Path(__file__).with_name(name + ".py")) for name in names}


def plan_runs(settings):
    grid = settings["grid"]
    count, stride = grid["replicates"], grid["seed_stride"]
    sizes = grid["seed_start"]
    if type(count) is not int or count < 1 or type(stride) is not int or stride < 1:
        raise ValueError("replicates and seed_stride must be positive integers")
    if not sizes or any(type(n) is not int or n < 1 or type(s) is not int for n, s in sizes.items()):
        raise ValueError("seed_start must map positive particle counts to integer seeds")
    runs = [Run(n, rank, start + rank * stride) for rank in range(count) for n, start in sorted(sizes.items())]
    if len({run.seed for run in runs}) != len(runs) or any(not 0 <= run.seed < 2**63 for run in runs):
        raise ValueError("Every planned run needs a distinct valid seed")
    return runs


def resolved_config(settings, particles):
    config = deepcopy(settings["model"])
    levels = [float(level) for level in settings["ladder"]["levels"][1:]]
    moves = settings["ladder"]["moves_per_level"]
    config["smc"].update(num_particles=particles, tilted_lambdas=[levels[-1]],
                         tilted_component_weights_a=[1.0], num_independent_runs=settings["grid"]["replicates"])
    config["smc"]["tilted_sampler"].update(num_tempering_levels=len(levels), mcmc_steps_per_level=moves)
    config["_exploration"] = {"id": f"N{particles}", "schedule": levels, "moves_per_level": moves}
    return config


def load_settings(path, *, smoke=False):
    settings = yaml.safe_load(path.read_bytes())
    if not isinstance(settings, dict) or settings["schema_version"] != 1 or "extends" in settings:
        raise ValueError("Expected a self-contained paper YAML with schema_version 1")
    if smoke:
        profile = settings["smoke"]
        settings["grid"] = profile["grid"]
        for key in ("levels", "moves_per_level", "observe", "recorded_thresholds"):
            settings["ladder"][key] = profile[key]
        settings["model"]["runtime"]["device"] = "cpu"
        settings["model"]["decoder"].update(sequence_length=3, mask_token_id=2)
        settings["mc"]["cohorts"] = [{**settings["mc"]["cohorts"][0], "name": "smoke", "matched": True,
                                      "samples": profile["matched_samples"]}]
        settings["mc"]["batch_size"] = profile["batch_size"]
        settings["figures"].update(usetex=False, bootstrap_resamples=profile["bootstrap_resamples"])
    runs = plan_runs(settings)
    ladder = settings["ladder"]
    levels, observe = ladder["levels"], ladder["observe"]
    if (len(levels) < 2 or levels[0] != 0 or any(not math.isfinite(x) for x in levels)
            or any(a >= b for a, b in zip(levels, levels[1:]))):
        raise ValueError("The tempering ladder must increase from zero")
    moves = ladder["moves_per_level"]
    if type(moves) is not int or moves < 1:
        raise ValueError("moves_per_level must be a positive integer")
    keys = {(item["threshold"], item["tilt"]) for item in observe}
    if (not keys or len(keys) != len(observe)
            or any(not 0 < q <= 1 or tilt not in levels[1:] for q, tilt in keys)
            or any(item["sweep"] != "last" for item in observe)):
        raise ValueError("Observation requires a unique threshold and existing post-level checkpoint")
    recorded = ladder["recorded_thresholds"]
    if not recorded or len(set(recorded)) != len(recorded) or any(not 0 < q <= 1 for q in recorded):
        raise ValueError("Invalid recorder thresholds")
    config = resolved_config(settings, runs[0].particles)
    if (config["smc"]["final_resampling"] or config["smc"]["ess_threshold"] != .5
            or config["smc"]["resampling_method"] != "systematic"
            or config["smc"]["tilted_sampler"]["type"] != "trajectory_resample_move"
            or config["smc"]["tilted_sampler"]["mutation_policy"] != "every_level"):
        raise ValueError("The validated paper sampler requires its original RollBack/resampling policy")
    decoder_id(config)  # Validates the pinned model and component configuration.
    component_config(config, runs[0].seed)
    if settings["mc"]["comparison"] != "calibrated_wall_clock":
        raise ValueError("Choose calibrated_wall_clock for the paper comparison")
    return settings


def run_signature(config, run, recorded):
    return fingerprint({"decoder": decoder_id(config), "evaluator": evaluator_id(config),
                        "component": asdict(component_config(config, run.seed)),
                        "schedule": config["_exploration"]["schedule"], "phi": config["smc"]["phi"],
                        "recorded_thresholds": recorded})


def make_record(settings, run, population, timing):
    config = resolved_config(settings, run.particles)
    recorded = settings["ladder"]["recorded_thresholds"]
    record = {"run": asdict(run), "signature": run_signature(config, run, recorded),
              "population": population, "timing": timing}
    validate_record(settings, run, record)
    return record


def validate_record(settings, run, record):
    config = resolved_config(settings, run.particles)
    recorded = settings["ladder"]["recorded_thresholds"]
    population = record["population"]
    summary = population["summary"]
    levels = population["level_diagnostics"]
    checkpoints = population["population_checkpoints"]
    if (record["run"] != asdict(run) or record["signature"] != run_signature(config, run, recorded)
            or (summary["num_particles"], summary["seed"], summary["lambda"]) !=
               (run.particles, run.seed, settings["ladder"]["levels"][-1])
            or population["phi"]["name"] != config["smc"]["phi"]
            or [row["eta"] for row in levels] != config["_exploration"]["schedule"]
            or [row["terminal"] for row in levels] != [False] * (len(levels) - 1) + [True]
            or len(checkpoints) != 1 + len(levels) * settings["ladder"]["moves_per_level"]
            or set(population["corrections"]) != {str(q) for q in recorded}):
        raise ValueError(f"Population identity or checkpoints differ from the plan for {run.filename}")
    timing = record["timing"]
    if (timing["excluded_from_estimates_and_measured_run_times"] is not True
            or not math.isfinite(timing["seconds_per_path"]) or timing["seconds_per_path"] <= 0
            or any(not math.isfinite(cp["runtime_seconds_cumulative"]) for cp in checkpoints)):
        raise ValueError(f"Invalid independent calibration or checkpoint timings for {run.filename}")
    audit_population(population, config)
    targets = checkpoint_targets(settings)
    # Reuse the historical checkpoint ledger validation before a run can be committed or imported.
    list(select_populations(population, targets, dict.fromkeys(targets),
                              sweeps_per_level=settings["ladder"]["moves_per_level"],
                              denoising_steps=config["decoder"]["denoising_steps"]))
    return record


def saved_records(settings, root, runs):
    folder = root / "runs"
    expected = {run.filename for run in runs}
    extra = {path.name for path in folder.glob("*.json")} - expected
    if extra:
        raise ValueError(f"Unexpected run artifacts at {folder}: {sorted(extra)}")
    return {run: validate_record(settings, run, read_json(folder / run.filename))
            for run in runs if (folder / run.filename).is_file()}


def execute_runs(settings, root, pending, context, timing):
    from .exploratory_population import population

    study = {"phi": settings["model"]["smc"]["phi"], "record_population_checkpoints": True,
             "thresholds": settings["ladder"]["recorded_thresholds"]}
    warmup = settings["mc"]["calibration"]
    config = resolved_config(settings, warmup["warmup_paths"])
    context.config = config
    population(context, config, {**study, "record_population_checkpoints": False}, warmup["smc_warmup_seed"])
    timing = {**timing, "smc_warmup_seed": warmup["smc_warmup_seed"]}
    for run in pending:
        destination = root / "runs" / run.filename
        if destination.exists():
            raise FileExistsError(destination)
        config = resolved_config(settings, run.particles)
        context.config = config
        result = population(context, config, study, run.seed)
        atomic_write_json(destination, make_record(settings, run, result, timing))
        print(f"Saved {run.filename}", flush=True)
