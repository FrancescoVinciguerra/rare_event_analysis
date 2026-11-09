"""Command-line entry point for suffix direct-MH SMC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[1]
for directory in (ROOT, EXPERIMENT_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    TORCH_DTYPES,
    load_model_and_tokenizer,
)
from rea_llms.observables import (  # noqa: E402
    ARI_CONVENTIONS,
    OBSERVABLE_NAMES,
    create_observable,
)
from rea_llms.observables.base import observable_configuration  # noqa: E402
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
    parser.add_argument("--model-dtype", choices=TORCH_DTYPES, default="auto")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "outputs")
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
        "--resample-final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resample the terminal population to uniform weights. Disabled by "
            "default because retaining terminal SMC weights avoids extra variance."
        ),
    )

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
    parser.add_argument("--observable", choices=OBSERVABLE_NAMES, default="ari")
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render --prompt as one user turn before generator tokenization.",
    )
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="raw")
    parser.add_argument("--ari-convention", choices=ARI_CONVENTIONS, default="current")
    parser.add_argument("--observable-backend", default="exact_cached")
    parser.add_argument("--disable-score-cache", action="store_true")
    parser.add_argument(
        "--guardrail-model",
        default="Qwen/Qwen3Guard-Gen-0.6B",
    )
    parser.add_argument(
        "--guardrail-revision",
        default="fada3b2f655b89601929198343c94cd2f64d93cc",
    )
    parser.add_argument("--guardrail-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--guardrail-dtype",
        choices=TORCH_DTYPES,
        default="float16",
    )
    parser.add_argument(
        "--guardrail-phi",
        choices=["unsafe_probability", "unsafe_log_odds"],
        default="unsafe_log_odds",
    )
    parser.add_argument("--guardrail-log-odds-clip", type=float, default=20.0)
    parser.add_argument("--guardrail-batch-size", type=int, default=16)
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
        resample_final=args.resample_final,
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

    if args.observable == "ari":
        observable = create_observable(
            "ari",
            mode=args.ari_mode,
            cap=args.ari_cap,
            backend=args.observable_backend,
            convention=args.ari_convention,
            cache_enabled=not args.disable_score_cache,
        )
    else:
        observable = create_observable(
            "qwen_guard_gen",
            model_name=args.guardrail_model,
            revision=args.guardrail_revision,
            device=args.guardrail_device,
            dtype=args.guardrail_dtype,
            phi_definition=args.guardrail_phi,
            log_odds_clip=args.guardrail_log_odds_clip,
            batch_size=args.guardrail_batch_size,
            cache_enabled=not args.disable_score_cache,
            local_files_only=args.local_files_only,
        )
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
        dtype=args.model_dtype,
    )
    model_device = next(model.parameters()).device
    if args.device == "cuda" and model_device.type != "cuda":
        raise RuntimeError("--device cuda was requested, but the model is not on CUDA")
    print(f"model device: {model_device}", flush=True)
    generator_prompt = args.prompt
    if args.apply_chat_template:
        generator_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

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
            prompt=generator_prompt,
            observable_prompt=args.prompt,
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
        "generator_prompt": generator_prompt,
        "requested_device": args.device,
        "model_device": str(model_device),
        "model_dtype": args.model_dtype,
        "observable": args.observable,
        "observable_configuration": observable_configuration(observable),
        "tilt_convention": "gamma_lambda(x) = p_M(x|c) * exp(-lambda * phi(x))",
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap,
        "ari_convention": args.ari_convention,
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
