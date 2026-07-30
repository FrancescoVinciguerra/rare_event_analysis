"""Command-line entry point for suffix direct-MH SMC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
)
from rea_llms.observables import create_observable  # noqa: E402
from suffix_direct_mh_smc_experiment.smc_sampler import (  # noqa: E402
    SuffixDirectMHSMCConfig,
    parse_float_list,
    parse_float_schedule,
    parse_int_list,
    parse_int_schedule,
    render_report,
    run_suffix_direct_mh_smc,
    write_json,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one independent SMC run configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run full-completion SMC with exact temperature-one LM suffix "
            "proposals and direct Metropolis-Hastings acceptance."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-label", default="suffix_direct_mh_smc")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lambdas", default="0,-0.25,-0.5,-1")
    parser.add_argument("--mcmc-steps-per-level", type=int, default=5)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--disable-resampling", action="store_true")

    parser.add_argument(
        "--suffix-mode",
        choices=["stochastic", "deterministic"],
        default="stochastic",
    )
    parser.add_argument("--suffix-sizes", default=None)
    parser.add_argument("--suffix-rhos", default=None)
    parser.add_argument("--suffix-size-schedule", default=None)
    parser.add_argument("--suffix-rho-schedule", default=None)
    parser.add_argument(
        "--allow-local-suffix-support",
        action="store_true",
        help="Allow a suffix-size law without positive probability for m=T.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tail-thresholds", default="6,8,10,12,15")
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="raw")
    parser.add_argument("--observable-backend", default="exact_cached")
    parser.add_argument("--disable-score-cache", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SuffixDirectMHSMCConfig:
    """Build the exact direct-MH configuration represented by CLI arguments."""

    return SuffixDirectMHSMCConfig(
        num_particles=args.num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=parse_float_list(args.lambdas),
        mcmc_steps_per_level=args.mcmc_steps_per_level,
        ess_threshold=args.ess_threshold,
        resampling=not args.disable_resampling,
        suffix_mode=args.suffix_mode,
        suffix_sizes=parse_int_list(args.suffix_sizes) if args.suffix_sizes else None,
        suffix_rhos=parse_float_list(args.suffix_rhos) if args.suffix_rhos else None,
        suffix_size_schedule=(
            parse_int_schedule(args.suffix_size_schedule)
            if args.suffix_size_schedule
            else None
        ),
        suffix_rho_schedule=(
            parse_float_schedule(args.suffix_rho_schedule)
            if args.suffix_rho_schedule
            else None
        ),
        require_global_irreducibility=not args.allow_local_suffix_support,
        batch_size=args.batch_size,
        seed=args.seed,
        tail_thresholds=parse_float_list(args.tail_thresholds),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = args.output_dir or (
        args.output_root / args.experiment_label / f"seed_{config.seed}"
    )
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"A completed run already exists at {output_dir}; choose another "
            "experiment label/seed or pass --overwrite."
        )

    observable = create_observable(
        "ari",
        mode=args.ari_mode,
        cap=args.ari_cap,
        backend=args.observable_backend,
        cache_enabled=not args.disable_score_cache,
    )
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

    progress = tqdm(total=max(0, len(config.lambdas) - 1), desc=f"SMC seed {config.seed}")

    def update_progress(row: dict) -> None:
        progress.set_postfix(
            lambda_value=f"{row['lambda']:g}",
            ess=f"{row['ess_over_n_before_resampling']:.3f}",
        )
        progress.update(1)

    try:
        result = run_suffix_direct_mh_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=config,
            progress_callback=update_progress,
        )
    finally:
        progress.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **result.config,
        "experiment_label": args.experiment_label,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "requested_device": args.device,
        "model_device": str(model_device),
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap,
        "observable_backend": args.observable_backend,
    }
    payload = {
        **result.summary,
        "diagnostics": result.diagnostics,
        "config": run_config,
    }
    write_json(output_dir / "summary.json", payload)
    write_jsonl(output_dir / "samples.jsonl", result.samples)
    (output_dir / "report.md").write_text(
        render_report(result.summary, result.diagnostics),
        encoding="utf-8",
    )
    print((output_dir / "summary.json").resolve(), flush=True)


if __name__ == "__main__":
    main()
