from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from official_annealed_tps_experiment.annealed_tps import (  # noqa: E402
    OfficialAnnealedTPSConfig,
    render_tps_report,
    run_official_annealed_tps,
)
from rea_llms.benchmarking import (  # noqa: E402
    completed_run_action,
    configuration_fingerprint,
    git_commit,
    split_rhat,
    summarize_independent_estimates,
)
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
    parse_int_list,
    render_report,
    run_suffix_direct_mh_smc,
    write_json,
    write_jsonl,
)


ALGORITHMS = {"suffix_direct_mh_smc", "official_annealed_tps"}


def _parse_string_list(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("expected at least one comma-separated value")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw-ARI suffix direct-MH SMC with official annealed TPS."
    )
    parser.add_argument(
        "--algorithms",
        default="suffix_direct_mh_smc,official_annealed_tps",
    )
    parser.add_argument("--seeds", default="200,201,202")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--lambdas",
        default=(
            "0,-0.25,-0.5,-0.75,-1,-1.25,-1.5,-1.75,-2,-2.25,-2.5,"
            "-2.75,-3,-3.25,-3.5,-3.75,-4,-4.25,-4.5,-4.75,-5"
        ),
    )
    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="raw")
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--tail-thresholds", default="10,12,15,18")

    parser.add_argument("--smc-num-particles", type=int, default=256)
    parser.add_argument("--smc-mcmc-steps-per-level", type=int, default=10)
    parser.add_argument("--smc-ess-threshold", type=float, default=0.5)
    parser.add_argument("--smc-suffix-sizes", default="1,2,4,8,16,32,64,100")
    parser.add_argument(
        "--smc-suffix-rhos",
        default="0.20,0.18,0.16,0.14,0.12,0.10,0.06,0.04",
    )
    parser.add_argument("--smc-batch-size", type=int, default=32)
    parser.add_argument("--tps-steps-per-bias", type=int, default=2000)

    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-label", default="raw_ari_smc_vs_official_tps")
    parser.add_argument("--configuration-label", default="selected")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reference-summary-path", type=Path, default=None)
    parser.add_argument(
        "--official-code-dir",
        type=Path,
        default=ROOT / "external" / "rea_of_llms_official",
    )
    return parser.parse_args(argv)


def _fingerprint_payload(
    args: argparse.Namespace,
    *,
    algorithm: str,
    seed: int,
    lambdas: tuple[float, ...],
    repository_commit: str,
    official_commit: str,
    model_device: str,
) -> dict[str, Any]:
    common = {
        "algorithm": algorithm,
        "seed": seed,
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "prompt": args.prompt,
        "completion_length": args.max_new_tokens,
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap if args.ari_mode == "capped" else None,
        "lambda_schedule": list(lambdas),
        "temperature": 1.0,
        "tail_thresholds": list(parse_float_list(args.tail_thresholds)),
        "observable_backend": "exact_cached",
        "requested_device": args.device,
        "model_device": model_device,
        "repository_commit": repository_commit,
        "official_tps_commit": official_commit,
    }
    if algorithm == "suffix_direct_mh_smc":
        return {
            **common,
            "num_particles": args.smc_num_particles,
            "mcmc_steps_per_level": args.smc_mcmc_steps_per_level,
            "suffix_sizes": list(parse_int_list(args.smc_suffix_sizes)),
            "suffix_rhos": list(parse_float_list(args.smc_suffix_rhos)),
            "ess_threshold": args.smc_ess_threshold,
            "resampling": "systematic_before_final_only",
            "batch_size": args.smc_batch_size,
            "mutation_kernel": "suffix_direct_mh",
        }
    return {
        **common,
        "steps_per_bias": args.tps_steps_per_bias,
        "suffix_configuration": "official_uniform_cut_indices_prompt_to_max_length_minus_1",
        "num_chains": 1,
        "resampling": False,
        "importance_weights": False,
        "batch_size": 1,
    }


def _run_smc(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
    output_dir: Path,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    repository_commit: str,
    official_commit: str,
) -> dict[str, Any]:
    action = completed_run_action(
        output_dir,
        fingerprint,
        skip_completed=args.skip_completed,
        overwrite=args.overwrite,
    )
    if action == "skip":
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    observable = create_observable(
        "ari", mode=args.ari_mode, cap=args.ari_cap, backend="exact_cached"
    )
    config = SuffixDirectMHSMCConfig(
        num_particles=args.smc_num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=lambdas,
        mcmc_steps_per_level=args.smc_mcmc_steps_per_level,
        ess_threshold=args.smc_ess_threshold,
        suffix_mode="stochastic",
        suffix_sizes=parse_int_list(args.smc_suffix_sizes),
        suffix_rhos=parse_float_list(args.smc_suffix_rhos),
        batch_size=args.smc_batch_size,
        seed=seed,
        tail_thresholds=thresholds,
    )
    progress = tqdm(total=max(0, len(lambdas) - 1), desc=f"SMC seed {seed}")

    def update(row: dict) -> None:
        progress.set_postfix(lambda_value=f"{row['lambda']:g}")
        progress.update(1)

    try:
        result = run_suffix_direct_mh_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=config,
            progress_callback=update,
        )
    finally:
        progress.close()
    if result.diagnostics["level_diagnostics"][-1]["lambda"] != lambdas[-1]:
        raise RuntimeError("SMC did not complete the full lambda schedule")
    summary = {
        **result.summary,
        "algorithm": "suffix_direct_mh_smc",
        "core_algorithm": "suffix_direct_mh_smc",
        "seed": seed,
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap if args.ari_mode == "capped" else None,
        "completed_full_schedule": True,
        "mean_raw_ari_final_target": result.summary["weighted_mean_ari"],
        "max_raw_ari_final_target": result.summary["max_ari"],
        "max_raw_ari_discovered": result.summary["top_discovered_completions"][0][
            "raw_ari"
        ],
        "tail_probabilities_final_target": result.summary[
            "weighted_tail_probabilities"
        ],
        "acceptance_rate": result.summary["mean_mcmc_acceptance_rate"],
        "smc_weight_ess": result.summary["final_ess_over_n"] * args.smc_num_particles,
        "smc_weight_ess_over_n": result.summary["final_ess_over_n"],
        "tps_temporal_ess": None,
        "configuration_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "repository_commit": repository_commit,
        "official_commit": official_commit,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "model_device": str(next(model.parameters()).device),
        "prompt": args.prompt,
        "diagnostics": result.diagnostics,
        "config": result.config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "samples.jsonl", result.samples)
    (output_dir / "report.md").write_text(
        render_report(summary, result.diagnostics), encoding="utf-8"
    )
    return summary


def _run_tps(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
    output_dir: Path,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    repository_commit: str,
    official_commit: str,
) -> dict[str, Any]:
    action = completed_run_action(
        output_dir,
        fingerprint,
        skip_completed=args.skip_completed,
        overwrite=args.overwrite,
    )
    if action == "skip":
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    observable = create_observable(
        "ari", mode=args.ari_mode, cap=args.ari_cap, backend="exact_cached"
    )
    config = OfficialAnnealedTPSConfig(
        max_new_tokens=args.max_new_tokens,
        lambdas=lambdas,
        steps_per_bias=args.tps_steps_per_bias,
        proposal_temperature=1.0,
        batch_size=1,
        seed=seed,
        ari_mode=args.ari_mode,
        ari_cap=args.ari_cap,
        tail_thresholds=thresholds,
    )
    progress = tqdm(total=len(lambdas), desc=f"TPS seed {seed}")

    def update(row: dict) -> None:
        progress.set_postfix(lambda_value=f"{row['lambda']:g}")
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
    if not result.summary["completed_full_schedule"]:
        raise RuntimeError("TPS did not complete every step of the full bias schedule")
    summary = {
        **result.summary,
        "smc_weight_ess": None,
        "smc_weight_ess_over_n": None,
        "configuration_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "repository_commit": repository_commit,
        "official_commit": official_commit,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "model_device": str(next(model.parameters()).device),
        "prompt": args.prompt,
        "config": result.config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    np.savez_compressed(output_dir / "trajectory.npz", **result.trajectory)
    (output_dir / "report.md").write_text(render_tps_report(summary), encoding="utf-8")
    return summary


def _run_record(summary: dict[str, Any], configuration_label: str) -> dict[str, Any]:
    tails = summary["tail_probabilities_final_target"]
    return {
        "algorithm": summary["algorithm"],
        "configuration": configuration_label,
        "seed": summary["seed"],
        "completed_full_schedule": summary["completed_full_schedule"],
        "runtime_seconds": summary["runtime_total_seconds"],
        "total_proposals": summary["total_proposals"],
        "total_regenerated_tokens": summary["total_regenerated_tokens"],
        "acceptance_rate": summary["acceptance_rate"],
        "mean_raw_ari_final_target": summary["mean_raw_ari_final_target"],
        "max_raw_ari": summary["max_raw_ari_discovered"],
        "tail_15": tails.get("ari_ge_15"),
        "tail_18": tails.get("ari_ge_18"),
        "smc_weight_ess": summary.get("smc_weight_ess"),
        "smc_weight_ess_over_n": summary.get("smc_weight_ess_over_n"),
        "tps_temporal_ess": summary.get("tps_temporal_ess"),
        "configuration_fingerprint": summary["configuration_fingerprint"],
    }


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in sorted({row["algorithm"] for row in records}):
        selected = [row for row in records if row["algorithm"] == algorithm]
        runtimes = [float(row["runtime_seconds"]) for row in selected]
        for estimator in ("mean_raw_ari_final_target", "tail_15", "tail_18"):
            values = [float(row[estimator]) for row in selected]
            rows.append(
                {
                    "algorithm": algorithm,
                    "estimator": estimator,
                    **summarize_independent_estimates(values, runtimes),
                }
            )
    return rows


def _render_comparison_report(payload: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        return "NaN" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# Raw-ARI SMC vs Official Annealed TPS",
        "",
        "Every row is a completed full schedule. SMC estimates use final normalized "
        "importance weights; TPS estimates use only the final-bias time series.",
        "",
        "No exact ground truth is available. Agreement, between-seed stability, temporal "
        "ESS, weight ESS, and empirical estimator efficiency are operational diagnostics, "
        "not a proof of convergence.",
        "",
        f"- Repository commit: `{payload['repository_commit']}`",
        f"- Official TPS commit: `{payload['official_commit']}`",
        f"- ARI mode: `{payload['ari_mode']}`",
        f"- Final target lambda: {payload['target_lambda']:g}",
        "",
        "## Runs",
        "",
        "| algorithm | configuration | seed | full schedule | runtime (s) | proposals | regenerated tokens | acceptance | mean raw ARI | max raw ARI | tail 15 | tail 18 | SMC weight ESS/N | TPS temporal ESS |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['algorithm']} | {row['configuration']} | {row['seed']} | "
            f"{str(row['completed_full_schedule']).lower()} | {row['runtime_seconds']:.3f} | "
            f"{row['total_proposals']} | {row['total_regenerated_tokens']} | "
            f"{fmt(row['acceptance_rate'])} | {fmt(row['mean_raw_ari_final_target'])} | "
            f"{fmt(row['max_raw_ari'])} | {fmt(row['tail_15'])} | {fmt(row['tail_18'])} | "
            f"{fmt(row['smc_weight_ess_over_n'])} | {fmt(row['tps_temporal_ess'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Independent-Seed Estimators",
            "",
            "| algorithm | estimator | mean | standard deviation | CI95 low | CI95 high | mean runtime (s) | empirical estimator efficiency |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["aggregate_estimators"]:
        lines.append(
            f"| {row['algorithm']} | {row['estimator']} | {fmt(row['mean'])} | "
            f"{fmt(row['standard_deviation'])} | {fmt(row['ci95_low'])} | "
            f"{fmt(row['ci95_high'])} | {fmt(row['mean_runtime_seconds'], 3)} | "
            f"{fmt(row['empirical_estimator_efficiency'])} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- TPS split-R-hat across independent seed chains: {fmt(payload['tps_split_rhat'])}",
            "- SMC weight ESS and TPS temporal ESS are different diagnostics and are not compared as interchangeable quantities.",
            "- Tail-discovery maxima describe evaluated proposals, not final-target estimator accuracy.",
            "- No ground truth is used unless an independent reference summary is supplied.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    algorithms = _parse_string_list(args.algorithms)
    unknown = set(algorithms) - ALGORITHMS
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("algorithms must not contain duplicates")
    seeds = parse_int_list(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be distinct")
    if args.ari_mode != "raw":
        raise ValueError("the SMC-vs-official-TPS benchmark requires --ari-mode raw")
    lambdas = parse_float_list(args.lambdas)
    if not math.isclose(float(lambdas[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the comparison lambda schedule must start at 0")
    thresholds = parse_float_list(args.tail_thresholds)
    if not {15.0, 18.0}.issubset(set(thresholds)):
        raise ValueError("--tail-thresholds must include 15 and 18 for this benchmark")
    repository_commit = git_commit(ROOT)
    official_commit = git_commit(args.official_code_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    model_device = str(next(model.parameters()).device)
    configuration_root = (
        args.output_root / args.experiment_label / args.configuration_label
    )
    summaries: list[dict[str, Any]] = []
    trajectory_paths: list[Path] = []
    for algorithm in algorithms:
        for seed in seeds:
            fingerprint_payload = _fingerprint_payload(
                args,
                algorithm=algorithm,
                seed=seed,
                lambdas=lambdas,
                repository_commit=repository_commit,
                official_commit=official_commit,
                model_device=model_device,
            )
            fingerprint = configuration_fingerprint(fingerprint_payload)
            output_dir = configuration_root / algorithm / f"seed_{seed}"
            if algorithm == "suffix_direct_mh_smc":
                summary = _run_smc(
                    args,
                    model,
                    tokenizer,
                    seed=seed,
                    lambdas=lambdas,
                    thresholds=thresholds,
                    output_dir=output_dir,
                    fingerprint=fingerprint,
                    fingerprint_payload=fingerprint_payload,
                    repository_commit=repository_commit,
                    official_commit=official_commit,
                )
            else:
                summary = _run_tps(
                    args,
                    model,
                    tokenizer,
                    seed=seed,
                    lambdas=lambdas,
                    thresholds=thresholds,
                    output_dir=output_dir,
                    fingerprint=fingerprint,
                    fingerprint_payload=fingerprint_payload,
                    repository_commit=repository_commit,
                    official_commit=official_commit,
                )
                trajectory_paths.append(output_dir / "trajectory.npz")
            summaries.append(summary)

    records = [_run_record(summary, args.configuration_label) for summary in summaries]
    tps_chains = []
    for path in trajectory_paths:
        with np.load(path) as trajectory:
            final_lambda_index = int(trajectory["lambda_index"].max())
            tps_chains.append(
                trajectory["raw_ari"][trajectory["lambda_index"] == final_lambda_index]
            )
    aggregate = _aggregate_records(records)
    reference = None
    reference_mse = None
    if args.reference_summary_path is not None:
        reference = json.loads(args.reference_summary_path.read_text(encoding="utf-8"))
        reference_values = reference.get("reference_estimators", reference)
        reference_mse = []
        for algorithm in sorted({row["algorithm"] for row in records}):
            selected = [row for row in records if row["algorithm"] == algorithm]
            for estimator in ("mean_raw_ari_final_target", "tail_15", "tail_18"):
                if estimator not in reference_values:
                    continue
                target = float(reference_values[estimator])
                mse = float(
                    np.mean([(float(row[estimator]) - target) ** 2 for row in selected])
                )
                reference_mse.append(
                    {"algorithm": algorithm, "estimator": estimator, "mse": mse}
                )
    payload = {
        "experiment_label": args.experiment_label,
        "configuration_label": args.configuration_label,
        "ari_mode": args.ari_mode,
        "target_lambda": lambdas[-1],
        "runs": records,
        "aggregate_estimators": aggregate,
        "tps_split_rhat": split_rhat(tps_chains),
        "reference_summary_path": (
            str(args.reference_summary_path) if args.reference_summary_path else None
        ),
        "reference": reference,
        "reference_mse": reference_mse,
        "ground_truth_available": reference is not None,
        "repository_commit": repository_commit,
        "official_commit": official_commit,
    }
    write_json(configuration_root / "comparison_summary.json", payload)
    (configuration_root / "comparison_report.md").write_text(
        _render_comparison_report(payload), encoding="utf-8"
    )
    print((configuration_root / "comparison_summary.json").resolve())


if __name__ == "__main__":
    main()
