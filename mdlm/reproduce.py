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
from diffusion_toxicity_smc_experiment.paper_reuse import audit_archive, check_archive_crude
from diffusion_toxicity_smc_experiment.storage import atomic_write_json, initialize_manifest, output_lock


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
    parser.add_argument("--mode", choices=("run", "plan", "resume", "analysis", "crude", "smoke", "audit"), default="run")
    parser.add_argument("--output", type=Path, help="Directory for this experiment and its derived reports")
    parser.add_argument("--crude-root", type=Path, help="Path to a portable crude score bundle")
    parser.add_argument("--archive", type=Path, help="Unpacked historical experiment to audit or reuse")
    parser.add_argument("--frozen-source", type=Path, help="Approved source bundle for --archive")
    parser.add_argument("--allow-network", action="store_true", help="Permit pinned model downloads")
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.frozen_source):
        parser.error("--archive and --frozen-source must be set together")
    if args.archive and args.mode not in {"audit", "run", "resume", "analysis"}:
        parser.error("Archive input is only used by audit, run, resume and analysis")
    smoke = args.mode == "smoke"
    settings = load_settings(args.config, smoke=smoke)
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
    root = (args.output or args.config.parent / settings["outputs"][output]).resolve()
    if args.archive and root.is_relative_to(args.archive.resolve()):
        raise ValueError("The output directory must be outside the historical archive")
    if args.crude_root and root.is_relative_to(args.crude_root.resolve()):
        raise ValueError("The output directory must be outside the input score bundle")
    crude_root = args.crude_root or root / "crude"
    representative = resolved_config(settings, runs[0].particles)
    with output_lock(root):
        imported = {}
        if args.archive or args.mode == "audit":
            audit, imported = audit_archive(settings, runs, args.archive, args.frozen_source)
            atomic_write_json(root / "compatibility.json", audit)
            print(json.dumps(audit["counts"]))
        if args.mode == "audit":
            return

        records = {} if args.mode == "crude" else saved_records(settings, root, runs)
        overlap = records.keys() & imported.keys()
        if any(records[run].get("archive") != imported[run]["archive"] for run in overlap):
            raise ValueError("The archive and the output directory contain different results for the same run")
        pending = [run for run in runs if run not in records and run not in imported]
        if args.mode == "analysis" and pending:
            raise ValueError(f"Analysis requires {len(runs)} complete ladders; {len(pending)} are missing")
        archived_cost = imported or any(r["timing"]["origin"] == "archive" for r in records.values())
        if archived_cost and args.crude_root is None:
            raise ValueError("Supply the original paper MC bundle with --crude-root for archived timing")

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
                torch.set_num_threads(runtime["cpu_threads"])
                device = torch.device(runtime["device"])
                if device.type == "cuda" and device.index is not None:
                    torch.cuda.set_device(device)
                context = Context(representative, local_files_only=not args.allow_network)
        if needs_scores and args.mode != "analysis":
            generate_crude(batches, crude_root, context)
        crude = load_crude(settings, runs, crude_root, representative)
        if archived_cost:
            check_archive_crude(settings, crude)
        if args.mode == "crude":
            write_reference(root, crude)
            return

        if native_run:
            for run in imported.keys() - overlap:
                atomic_write_json(root / "runs" / run.filename, imported[run])
            if pending:
                execute_runs(settings, root, pending, context, calibrate(settings, context))
            records = saved_records(settings, root, runs)
        else:
            records.update(imported)
        analyse(settings, root, records, crude)
        print(f"Analysed {len(records)} ladders in {root}")


if __name__ == "__main__":
    main()
