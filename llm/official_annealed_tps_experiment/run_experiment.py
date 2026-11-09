from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parent
for directory in (ROOT / "shared", EXPERIMENT_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from official_annealed_tps_experiment.annealed_tps import (  # noqa: E402
    OfficialAnnealedTPSConfig,
    render_tps_report,
    run_official_annealed_tps,
)
from rea_llms.benchmarking import (  # noqa: E402
    completed_run_action,
    configuration_fingerprint,
    git_commit,
)
from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
    resolve_device,
)
from rea_llms.observables import ARI_CONVENTIONS, create_observable  # noqa: E402
from suffix_direct_mh_smc_experiment.smc_sampler import (  # noqa: E402
    parse_float_list,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official annealed TPS adapter.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lambdas", default="0,-0.25,-0.5,-1.0")
    parser.add_argument("--steps-per-bias", type=int, default=1000)
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="raw")
    parser.add_argument("--ari-convention", choices=ARI_CONVENTIONS, default="current")
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--tail-thresholds", default="10,12,15,18")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "outputs")
    parser.add_argument("--experiment-label", default="official_annealed_tps")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--official-code-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "external" / "rea_of_llms_official",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    repository_commit = git_commit(ROOT)
    official_commit = git_commit(args.official_code_dir)
    selected_device = resolve_device(args.device)
    config = OfficialAnnealedTPSConfig(
        max_new_tokens=args.max_new_tokens,
        lambdas=parse_float_list(args.lambdas),
        steps_per_bias=args.steps_per_bias,
        proposal_temperature=args.proposal_temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        ari_mode=args.ari_mode,
        ari_cap=args.ari_cap,
        tail_thresholds=parse_float_list(args.tail_thresholds),
    )
    fingerprint_payload = {
        "algorithm": "official_annealed_tps",
        "seed": args.seed,
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "prompt": args.prompt,
        "completion_length": args.max_new_tokens,
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap if args.ari_mode == "capped" else None,
        "ari_convention": args.ari_convention,
        "lambda_schedule": list(config.lambdas),
        "steps_per_bias": args.steps_per_bias,
        "suffix_configuration": "official_uniform_cut_indices_prompt_to_max_length_minus_1",
        "temperature": args.proposal_temperature,
        "tail_thresholds": list(config.tail_thresholds),
        "observable_backend": "exact_cached",
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "model_device": str(selected_device),
        "repository_commit": repository_commit,
        "official_tps_commit": official_commit,
    }
    fingerprint = configuration_fingerprint(fingerprint_payload)
    output_dir = args.output_dir or (
        args.output_root / args.experiment_label / "official_annealed_tps" / f"seed_{args.seed}"
    )
    action = completed_run_action(
        output_dir,
        fingerprint,
        skip_completed=args.skip_completed,
        overwrite=args.overwrite,
    )
    if action == "skip":
        print(f"Skipping compatible completed TPS run: {output_dir}")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    observable = create_observable(
        "ari",
        mode=args.ari_mode,
        cap=args.ari_cap,
        backend="exact_cached",
        convention=args.ari_convention,
    )
    progress = tqdm(total=len(config.lambdas), desc=f"TPS seed {args.seed}")

    def update(row: dict) -> None:
        progress.set_postfix(
            lambda_value=f"{row['lambda']:g}",
            acceptance=f"{row['acceptance_rate']:.3f}",
        )
        progress.update(1)

    try:
        result = run_official_annealed_tps(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=config,
            progress_callback=update,
        )
    finally:
        progress.close()
    summary = {
        **result.summary,
        "configuration_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "repository_commit": repository_commit,
        "official_commit": official_commit,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "ari_convention": args.ari_convention,
        "model_device": str(next(model.parameters()).device),
        "config": result.config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    np.savez_compressed(output_dir / "trajectory.npz", **result.trajectory)
    (output_dir / "report.md").write_text(render_tps_report(summary), encoding="utf-8")
    print((output_dir / "summary.json").resolve())


if __name__ == "__main__":
    main()
