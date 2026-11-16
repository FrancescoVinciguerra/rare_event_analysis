"""Run or analyse the MDLM paper experiment from a single YAML."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from diffusion_toxicity_smc_experiment.paper_analysis import METRICS, analyse, write_reference
from diffusion_toxicity_smc_experiment.crude import calibrate, check_batches, generate_crude, load_crude, planned_batches
from diffusion_toxicity_smc_experiment.paper_experiment import (
    execute_runs, load_settings, plan_runs, resolved_config, saved_records, scientific_sources,
)
from diffusion_toxicity_smc_experiment.storage import initialize_manifest, output_lock


def smoke_context(config):
    import torch

    from diffusion_toxicity_smc_experiment.ancestral import NativeMDLMSampler
    from diffusion_toxicity_smc_experiment.potentials import potential_from_config

    torch.set_num_threads(1)

    def logits(*, input_ids, timesteps, return_dict):
        return (torch.zeros((*input_ids.shape, 3), device=input_ids.device),)

    class Scores:
        potential = potential_from_config(config)

        def reward(self, tokens):
            values = torch.where(tokens[:, -1] == 1, 4.0, -4.0).double()
            self.last_event_rows = [{"toxic_probability": score} for score in values.sigmoid().tolist()]
            return values

        def event(self, tokens):
            self.reward(tokens)
            return tokens[:, -1] == 1

    return SimpleNamespace(config=config, device=torch.device("cpu"), synchronize=lambda: None,
                           sampler=NativeMDLMSampler(logits, mask_token_id=2), functions=Scores())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "plan", "resume", "analysis", "crude", "smoke"), default="run")
    parser.add_argument("--output", type=Path, help="Directory for this experiment and its derived reports")
    parser.add_argument("--crude-root", type=Path, help="Path to a portable crude score bundle")
    parser.add_argument("--data", type=Path, help="Read an existing runs/ and crude/ bundle in analysis mode")
    parser.add_argument("--allow-network", action="store_true", help="Permit pinned model downloads")
    args = parser.parse_args(argv)
    if args.data and args.mode != "analysis":
        parser.error("--data requires --mode analysis")
    smoke = args.mode == "smoke"
    config_path = args.config.expanduser().resolve(strict=True)
    settings = load_settings(config_path, smoke=smoke)
    runs = plan_runs(settings)
    batches = planned_batches(settings)
    check_batches(settings, runs, batches)
    figures = settings["figures"]
    metrics = [*figures["metrics"], *figures["summary_only_metrics"]]
    lower, upper = figures["bootstrap_quantiles"]
    if (not figures["metrics"] or len(set(metrics)) != len(metrics) or set(metrics) - METRICS.keys()
            or figures["bootstrap_resamples"] < 1 or not 0 <= lower < upper <= 1):
        raise ValueError("Invalid figure metrics or bootstrap settings")
    if args.mode == "plan":
        print(json.dumps({"planned_runs": len(runs), "runs": [asdict(run) for run in runs]}, indent=2))
        return

    output = "smoke_directory" if smoke else "directory"
    root = (args.output or config_path.parent / settings["outputs"][output]).resolve()
    data = args.data.expanduser().resolve() if args.data else root
    if args.data and not data.is_dir():
        parser.error(f"Analysis data directory is missing: {data}")
    if args.data and root.is_relative_to(data):
        parser.error("The output directory must be outside the analysis data bundle")
    if args.crude_root and root.is_relative_to(args.crude_root.resolve()):
        raise ValueError("The output directory must be outside the input score bundle")
    crude_root = args.crude_root or data / "crude"
    representative = resolved_config(settings, runs[0].particles)
    with output_lock(root):
        records = {} if args.mode == "crude" else saved_records(settings, data, runs)
        pending = [run for run in runs if run not in records]
        if args.mode == "analysis" and pending:
            raise ValueError(f"Analysis requires {len(runs)} complete ladders; {len(pending)} are missing")
        native_run = args.mode in {"run", "resume", "smoke"}
        if native_run:
            initialize_manifest(root, {"schema_version": settings["schema_version"], "model": settings["model"],
                                "ladder": {k: v for k, v in settings["ladder"].items() if k != "observe"},
                                "plan": [asdict(run) for run in runs], "synthetic_smoke": smoke,
                                "scientific_source_sha256": scientific_sources()})
        needs_scores = args.crude_root is None and not (crude_root / "scores_manifest.json").is_file()
        context = None
        if (native_run and pending) or (args.mode != "analysis" and needs_scores):
            if smoke:
                context = smoke_context(representative)
            else:
                import torch
                from diffusion_toxicity_smc_experiment.pipeline import Context

                runtime = representative["runtime"]
                if hf_cache := runtime.get("hf_cache"):
                    runtime["hf_cache"] = str((config_path.parent / Path(hf_cache).expanduser()).resolve())
                torch.set_num_threads(runtime["cpu_threads"])
                device = torch.device(runtime["device"])
                if device.type == "cuda" and device.index is not None:
                    torch.cuda.set_device(device)
                context = Context(representative, local_files_only=not args.allow_network)
        if needs_scores and args.mode != "analysis":
            generate_crude(batches, crude_root, context)
        crude = load_crude(settings, runs, crude_root, representative)
        if args.mode == "crude":
            write_reference(root, crude)
            return

        if native_run and pending:
            execute_runs(settings, root, pending, context, calibrate(settings, context))
            records = saved_records(settings, root, runs)
        analyse(settings, root, records, crude)
        print(f"Analysed {len(records)} ladders in {root}")


if __name__ == "__main__":
    main()
