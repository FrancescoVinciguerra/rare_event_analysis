"""Estimate base-model ARI tails from final suffix direct-MH SMC particles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rea_llms.benchmarking import configuration_fingerprint, git_commit  # noqa: E402
from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
)
from rea_llms.observables import create_observable  # noqa: E402
from rea_llms.tail_estimation import estimate_base_tails_from_final_samples  # noqa: E402
from suffix_balanced_smc_experiment.smc_sampler import (  # noqa: E402
    SuffixBalancedSMCConfig,
    metropolis_fast_path_enabled,
    parse_float_list,
    parse_float_schedule,
    parse_int_list,
    parse_int_schedule,
    run_suffix_balanced_smc,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run suffix direct-MH SMC and estimate base-model tail probabilities "
            "from its final weighted population."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-label", default="suffix_smc_tail_estimation")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--seeds", default="200")
    parser.add_argument("--num-particles", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lambdas", default="0,-0.25,-0.5,-0.75,-1,-1.5,-2")
    parser.add_argument("--mcmc-steps-per-level", type=int, default=5)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mask-mode",
        choices=["stochastic", "deterministic"],
        default="stochastic",
        help="Notebook-compatible name for the state-independent suffix-size mode.",
    )
    parser.add_argument(
        "--mask-sizes",
        default="1,2,4,8,16,32,64,100",
        help=(
            "Stochastic suffix support, or a deterministic suffix-size schedule. "
            "The name is retained for notebook parameter compatibility."
        ),
    )
    parser.add_argument(
        "--mask-rho-schedule",
        default="0.20,0.18,0.16,0.14,0.12,0.10,0.06,0.04",
        help="Semicolon-separated suffix-size probability rows.",
    )
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tail-thresholds", default="6,8,10,12,15,20")
    return parser.parse_args(argv)


def _config_for_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
) -> SuffixBalancedSMCConfig:
    common: dict[str, Any] = {
        "num_particles": args.num_particles,
        "max_new_tokens": args.max_new_tokens,
        "lambdas": lambdas,
        "mcmc_steps_per_level": args.mcmc_steps_per_level,
        "ess_threshold": args.ess_threshold,
        "resampling": True,
        "suffix_mode": args.mask_mode,
        "balancing_function": "metropolis",
        "metropolis_fast_path": True,
        "proposal_temperature": args.proposal_temperature,
        "batch_size": args.batch_size,
        "implementation": "parallel",
        "seed": seed,
        "tail_thresholds": thresholds,
    }
    if args.mask_mode == "stochastic":
        common.update(
            suffix_sizes=parse_int_list(args.mask_sizes),
            suffix_rho_schedule=parse_float_schedule(args.mask_rho_schedule),
        )
    else:
        common.update(
            suffix_sizes=tuple(
                sorted(
                    {
                        value
                        for row in parse_int_schedule(args.mask_sizes)
                        for value in row
                    }
                )
            ),
            suffix_size_schedule=parse_int_schedule(args.mask_sizes),
        )
    config = SuffixBalancedSMCConfig(**common)
    if not metropolis_fast_path_enabled(config):
        raise RuntimeError("tail estimation requires the exact direct-Metropolis fast path")
    return config


def _run_action(
    run_path: Path,
    fingerprint: str,
    *,
    skip_completed: bool,
    overwrite: bool,
) -> str:
    if not run_path.exists():
        return "run"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    existing = payload.get("configuration_fingerprint")
    if existing == fingerprint and skip_completed:
        return "skip"
    if existing != fingerprint and not overwrite:
        raise FileExistsError(
            f"Incompatible completed run at {run_path.parent}; choose another "
            "experiment label or pass --overwrite."
        )
    if existing == fingerprint and not overwrite:
        raise FileExistsError(
            f"Completed run already exists at {run_path.parent}; pass "
            "--skip-completed or --overwrite."
        )
    return "run"


def render_tail_estimation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Suffix direct-MH SMC tail estimation",
        "",
        "The base-model probabilities use a consistent, finite-particle-biased "
        "self-normalized importance-sampling correction of the final SMC population.",
        "",
        f"- Runtime: {summary['runtime_seconds']:.3f} s",
        f"- Final lambda: {summary['lambda_final']:.6g}",
        f"- Particles: {summary['num_particles']}",
        f"- SMC-weighted mean phi: {summary['mean_phi']:.6f}",
        f"- Maximum phi: {summary['max_phi']:.6f}",
        f"- Final SMC ESS/N: {summary['smc_ess_over_n']:.6f}",
        f"- Corrected importance ESS/N: {summary['corrected_importance_ess_over_n']:.6f}",
        "",
        "| threshold | raw count | raw tilted fraction | estimated probability under p_M |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["tails"]:
        lines.append(
            f"| {row['threshold']:.6g} | {row['raw_count']} | "
            f"{row['raw_tilted_fraction']:.6f} | "
            f"{row['estimated_probability_under_p_m']:.8g} |"
        )
    return "\n".join(lines) + "\n"


def _run_seed(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
    repository_commit: str,
) -> Path:
    config = _config_for_seed(
        args,
        seed=seed,
        lambdas=lambdas,
        thresholds=thresholds,
    )
    essential_config = {
        **asdict(config),
        "algorithm": "suffix_direct_mh_smc",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "ari_mode": "raw",
        "observable_backend": "exact_cached",
        "requested_device": args.device,
        "model_device": str(next(model.parameters()).device),
    }
    fingerprint = configuration_fingerprint(
        {**essential_config, "repository_commit": repository_commit}
    )
    output_dir = args.output_root / args.experiment_label / f"seed_{seed}"
    run_path = output_dir / "run.json"
    action = _run_action(
        run_path,
        fingerprint,
        skip_completed=args.skip_completed,
        overwrite=args.overwrite,
    )
    if action == "skip":
        print(f"Reusing completed run: {run_path}", flush=True)
        return run_path

    observable = create_observable("ari", mode="raw", backend="exact_cached")
    progress = tqdm(total=max(0, len(lambdas) - 1), desc=f"SMC seed {seed}")

    def update_progress(row: dict[str, Any]) -> None:
        progress.set_postfix(
            lambda_value=f"{row['lambda']:g}",
            ess=f"{row['ess_over_n_before_resampling']:.3f}",
        )
        progress.update(1)

    try:
        result = run_suffix_balanced_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=config,
            progress_callback=update_progress,
        )
    finally:
        progress.close()
    if result.summary.get("resample_final") is not False:
        raise RuntimeError("the final weighted SMC population was unexpectedly resampled")
    if len(result.samples) != config.num_particles:
        raise RuntimeError("final SMC sample count does not match num_particles")

    estimates = estimate_base_tails_from_final_samples(
        result.samples,
        lambda_final=float(lambdas[-1]),
        thresholds=thresholds,
    ).to_dict()
    summary = {
        "runtime_seconds": float(result.summary["runtime_total_seconds"]),
        "lambda_final": float(lambdas[-1]),
        "num_particles": config.num_particles,
        "mean_phi": float(result.summary["weighted_mean_ari"]),
        "max_phi": max(float(sample["phi"]) for sample in result.samples),
        "smc_ess_over_n": float(result.summary["final_ess_over_n"]),
        **estimates,
    }
    payload = {
        "schema_version": 1,
        "algorithm": "suffix_direct_mh_smc",
        "estimator": "self_normalized_importance_sampling",
        "finite_particle_bias": "generally biased; consistent as particle count increases",
        "seed": seed,
        "configuration_fingerprint": fingerprint,
        "repository_commit": repository_commit,
        "summary": summary,
        "config": essential_config,
        "samples": result.samples,
    }
    write_json(run_path, payload)
    (output_dir / "report.md").write_text(
        render_tail_estimation_report(summary),
        encoding="utf-8",
    )
    print(run_path.resolve(), flush=True)
    return run_path


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct values")
    lambdas = parse_float_list(args.lambdas)
    if not math.isclose(lambdas[0], 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("--lambdas must start at zero")
    thresholds = parse_float_list(args.tail_thresholds)
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("--tail-thresholds must be sorted in non-decreasing order")
    if args.skip_completed and args.overwrite:
        raise ValueError("use either --skip-completed or --overwrite, not both")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    model_device = next(model.parameters()).device
    if args.device == "cuda" and model_device.type != "cuda":
        raise RuntimeError("--device cuda was requested, but the model is not on CUDA")
    print(f"model device: {model_device}", flush=True)
    repository_commit = git_commit(ROOT)
    for seed in seeds:
        _run_seed(
            args,
            model,
            tokenizer,
            seed=seed,
            lambdas=lambdas,
            thresholds=thresholds,
            repository_commit=repository_commit,
        )


if __name__ == "__main__":
    main()
