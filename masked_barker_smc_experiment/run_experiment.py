from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from masked_barker_smc_experiment.smc_sampler import (  # noqa: E402
    MaskedBarkerSMCConfig,
    parse_explicit_masks,
    parse_float_list,
    parse_float_schedule,
    parse_int_list,
    parse_int_schedule,
    render_report,
    run_masked_barker_smc,
    write_json,
    write_jsonl,
)
from rea_llms.experiment_log import (  # noqa: E402
    append_experiment_log,
    compute_ari_tail_summary,
    publish_selected_result,
    render_experiment_log_entry,
)
from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
)
from rea_llms.observables import create_observable  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-sequence SMC with state-independent token masks, "
            "an exactly symmetrized masked-LM proposal, and Barker mutation."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "masked_barker_smc_run",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-log", type=Path, default=None)
    parser.add_argument("--publish-result-dir", type=Path, default=None)

    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lambdas", default="0,-0.25,-0.5,-1.0")
    parser.add_argument("--mcmc-steps-per-level", type=int, default=5)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--resampling", choices=["systematic"], default="systematic")
    parser.add_argument("--disable-resampling", action="store_true")

    parser.add_argument("--mask-mode", choices=["stochastic", "deterministic"], default="stochastic")
    parser.add_argument("--mask-sizes", default="1,2,4,8")
    parser.add_argument(
        "--mask-rho-schedule",
        default=None,
        help="Comma-separated probabilities, or semicolon-separated rows by SMC transition.",
    )
    parser.add_argument(
        "--deterministic-mask-size-schedule",
        default=None,
        help="One size per level, or semicolon-separated rows with comma-separated MCMC-step sizes.",
    )
    parser.add_argument(
        "--mask-policy",
        choices=["uniform_noncontiguous", "contiguous", "explicit"],
        default="uniform_noncontiguous",
    )
    parser.add_argument(
        "--explicit-masks",
        default=None,
        help=(
            "JSON with zero-based completion indices and shape [level][step][index]. "
            "A single [index, ...] mask is repeated at every level and step."
        ),
    )
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--score-token-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tail-thresholds", default="6,7,8")

    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--observable-backend", default="exact_cached")
    parser.add_argument("--disable-score-cache", action="store_true")
    return parser.parse_args(argv)


def _progress(row: dict) -> None:
    acceptance = row["acceptance_rate"]
    acceptance_text = "n/a" if acceptance is None else f"{acceptance:.3f}"
    print(
        f"level {row['level']}: lambda={row['lambda']:g}, "
        f"ESS/N={row['ess_over_n_before_resampling']:.3f}, "
        f"resampled={row['resampled']}, acceptance={acceptance_text}, "
        f"weighted ARI pre={row['weighted_mean_ari_before_resampling']:.3f}, "
        f"unweighted ARI post={row['unweighted_mean_ari_after_resampling']:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = MaskedBarkerSMCConfig(
        num_particles=args.num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=parse_float_list(args.lambdas),
        mcmc_steps_per_level=args.mcmc_steps_per_level,
        ess_threshold=args.ess_threshold,
        resampling=not args.disable_resampling,
        resampling_method=args.resampling,
        mask_mode=args.mask_mode,
        mask_sizes=parse_int_list(args.mask_sizes),
        mask_rho_schedule=(
            parse_float_schedule(args.mask_rho_schedule)
            if args.mask_rho_schedule is not None
            else None
        ),
        deterministic_mask_size_schedule=(
            parse_int_schedule(args.deterministic_mask_size_schedule)
            if args.deterministic_mask_size_schedule is not None
            else None
        ),
        mask_policy=args.mask_policy,
        explicit_masks=(
            parse_explicit_masks(args.explicit_masks) if args.explicit_masks is not None else None
        ),
        proposal_temperature=args.proposal_temperature,
        batch_size=args.batch_size,
        score_token_chunk_size=args.score_token_chunk_size,
        seed=args.seed,
        tail_thresholds=parse_float_list(args.tail_thresholds),
    )
    observable = create_observable(
        "ari",
        mode="capped",
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
    print(f"model device: {model_device}", flush=True)
    if args.device == "cuda" and model_device.type != "cuda":
        raise RuntimeError("--device cuda was requested, but the model is not on CUDA")

    result = run_masked_barker_smc(
        model,
        tokenizer,
        observable,
        prompt=args.prompt,
        config=config,
        progress_callback=_progress,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.name
    run_config = {
        **result.config,
        "run_name": run_name,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "requested_device": args.device,
        "model_device": str(model_device),
        "ari_mode": "capped",
        "ari_cap": args.ari_cap,
        "observable_backend": args.observable_backend,
    }
    ari_tail_summary = compute_ari_tail_summary(
        result.samples,
        thresholds=config.tail_thresholds,
    )
    ari_tail_summary["unweighted_mean_ari"] = ari_tail_summary.pop("mean_ari")
    ari_tail_summary["weighted_mean_ari"] = result.summary["weighted_mean_ari"]
    ari_tail_summary["raw_tail_proportions"] = ari_tail_summary.pop("tail_probabilities")
    ari_tail_summary["weighted_tail_probabilities"] = result.summary[
        "weighted_tail_probabilities"
    ]
    enriched_summary = {
        **result.summary,
        "run_name": run_name,
        "algorithm": "masked_barker_smc",
        "ari_tail_summary": ari_tail_summary,
    }
    write_json(
        args.output_dir / "summary.json",
        {
            **enriched_summary,
            "diagnostics": result.diagnostics,
            "config": run_config,
        },
    )
    write_jsonl(args.output_dir / "samples.jsonl", result.samples)
    (args.output_dir / "report.md").write_text(
        render_report(enriched_summary, result.diagnostics),
        encoding="utf-8",
    )
    if args.experiment_log is not None:
        append_experiment_log(
            args.experiment_log,
            render_experiment_log_entry(
                run_name=run_name,
                algorithm="masked_barker_smc",
                summary=enriched_summary,
                config=run_config,
            ),
        )
    if args.publish_result_dir is not None:
        publish_selected_result(
            output_dir=args.output_dir,
            publish_root=args.publish_result_dir,
            run_name=run_name,
            samples=result.samples,
        )
    print((args.output_dir / "summary.json").resolve())


if __name__ == "__main__":
    main()
