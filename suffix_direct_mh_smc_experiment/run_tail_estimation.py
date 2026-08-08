"""Estimate base-model ARI tails from final suffix direct-MH SMC particles."""

from __future__ import annotations

import argparse
import json
import math
import re
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
from rea_llms.observables import ARI_CONVENTIONS, create_observable  # noqa: E402
from rea_llms.tail_estimation import (  # noqa: E402
    aggregate_terminal_lambda_runs,
    estimate_base_tails_from_final_samples,
    overlap_is_usable,
)
from suffix_direct_mh_smc_experiment.smc_sampler import (  # noqa: E402
    SuffixDirectMHSMCConfig,
    parse_float_list,
    parse_float_schedule,
    parse_int_list,
    parse_int_schedule,
    run_suffix_direct_mh_smc,
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
    parser.add_argument(
        "--lambda-schedules-json",
        help=(
            "JSON object mapping configuration labels to independent lambda "
            "schedules. When set, --lambdas is ignored."
        ),
    )
    parser.add_argument("--mcmc-steps-per-level", type=int, default=5)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument(
        "--resample-final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resample the terminal population to uniform weights. Disabled by "
            "default so the SNIS estimator uses the terminal SMC weights directly."
        ),
    )
    parser.add_argument(
        "--suffix-mode",
        choices=["stochastic", "deterministic"],
        default="stochastic",
        help="State-independent suffix-size selection mode.",
    )
    parser.add_argument(
        "--suffix-sizes",
        default="1,2,4,8,16,32,64,100",
        help=(
            "Stochastic suffix support, or a deterministic suffix-size schedule. "
            "Use semicolon-separated rows for deterministic schedules."
        ),
    )
    parser.add_argument(
        "--suffix-rho-schedule",
        default="0.20,0.18,0.16,0.14,0.12,0.10,0.06,0.04",
        help="Semicolon-separated suffix-size probability rows.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tail-thresholds", default="6,8,10,12,15,20")
    parser.add_argument("--ari-convention", choices=ARI_CONVENTIONS, default="current")
    parser.add_argument("--min-corrected-ess-over-n", type=float, default=0.10)
    parser.add_argument("--min-raw-event-fraction", type=float, default=0.05)
    parser.add_argument("--max-raw-event-fraction", type=float, default=0.95)
    return parser.parse_args(argv)


_CONFIGURATION_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def tail_run_directory(
    output_root: Path,
    experiment_label: str,
    seed: int,
    *,
    configuration_label: str | None = None,
) -> Path:
    root = output_root / experiment_label
    if configuration_label is not None:
        if not _CONFIGURATION_LABEL_RE.fullmatch(configuration_label):
            raise ValueError(
                "configuration labels may contain only letters, numbers, '.', '-', and '_'"
            )
        root /= configuration_label
    return root / f"seed_{seed}"


def parse_lambda_schedules(value: str) -> dict[str, tuple[float, ...]]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--lambda-schedules-json must be valid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--lambda-schedules-json must be a non-empty JSON object")
    schedules: dict[str, tuple[float, ...]] = {}
    for raw_label, raw_schedule in raw.items():
        label = str(raw_label)
        if not _CONFIGURATION_LABEL_RE.fullmatch(label):
            raise ValueError(
                "configuration labels may contain only letters, numbers, '.', '-', and '_'"
            )
        if isinstance(raw_schedule, str):
            schedule = parse_float_list(raw_schedule)
        elif isinstance(raw_schedule, list):
            schedule = tuple(float(item) for item in raw_schedule)
        else:
            raise ValueError(f"lambda schedule {label!r} must be a string or JSON list")
        if not schedule:
            raise ValueError(f"lambda schedule {label!r} is empty")
        if not math.isclose(schedule[0], 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"lambda schedule {label!r} must start at zero")
        if any(not math.isfinite(item) for item in schedule):
            raise ValueError(f"lambda schedule {label!r} must be finite")
        schedules[label] = schedule
    return schedules


def _config_for_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
) -> SuffixDirectMHSMCConfig:
    common: dict[str, Any] = {
        "num_particles": args.num_particles,
        "max_new_tokens": args.max_new_tokens,
        "lambdas": lambdas,
        "mcmc_steps_per_level": args.mcmc_steps_per_level,
        "ess_threshold": args.ess_threshold,
        "resampling": True,
        "resample_final": args.resample_final,
        "suffix_mode": args.suffix_mode,
        "batch_size": args.batch_size,
        "seed": seed,
        "tail_thresholds": thresholds,
    }
    if args.suffix_mode == "stochastic":
        common.update(
            suffix_sizes=parse_int_list(args.suffix_sizes),
            suffix_rho_schedule=parse_float_schedule(args.suffix_rho_schedule),
        )
    else:
        common.update(
            suffix_sizes=tuple(
                sorted(
                    {
                        value
                        for row in parse_int_schedule(args.suffix_sizes)
                        for value in row
                    }
                )
            ),
            suffix_size_schedule=parse_int_schedule(args.suffix_sizes),
        )
    return SuffixDirectMHSMCConfig(**common)


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
        "| threshold | raw count | raw tilted fraction | "
        "estimated probability under p_M | overlap usable |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["tails"]:
        lines.append(
            f"| {row['threshold']:.6g} | {row['raw_count']} | "
            f"{row['raw_tilted_fraction']:.6f} | "
            f"{row['estimated_probability_under_p_m']:.8g} | "
            f"{str(row['overlap_usable']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def render_terminal_lambda_sweep_report(aggregate: dict[str, Any]) -> str:
    diagnostics = aggregate["diagnostic_thresholds"]
    lines = [
        "# Terminal-lambda SMC tail sweep",
        "",
        "Each row aggregates independent seeds from one SMC configuration. "
        "Different terminal lambdas are not combined. Estimates use only each "
        "run's final weighted population.",
        "",
        "The SNIS correction is consistent but generally biased at finite particle count.",
        "",
        "Overlap is a diagnostic flag only: corrected ESS/N >= "
        f"{diagnostics['min_corrected_ess_over_n']:.3g} and raw event fraction in "
        f"[{diagnostics['min_raw_event_fraction']:.3g}, "
        f"{diagnostics['max_raw_event_fraction']:.3g}].",
        "",
        "| configuration | terminal lambda | threshold | seeds | mean raw count | "
        "mean raw fraction | mean SNIS P under p_M | SD across seeds | min ESS/N | "
        "mean ESS/N | usable seeds | overlap usable |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate["aggregate_rows"]:
        std = row["std_estimated_probability_under_p_m"]
        std_text = "NA" if std is None else f"{std:.8g}"
        lines.append(
            f"| {row['configuration']} | {row['lambda_final']:.6g} | "
            f"{row['threshold']:.6g} | {row['completed_seeds']} | "
            f"{row['mean_raw_event_count']:.4g} | "
            f"{row['mean_raw_event_fraction']:.6f} | "
            f"{row['mean_estimated_probability_under_p_m']:.8g} | {std_text} | "
            f"{row['minimum_corrected_ess_over_n']:.6f} | "
            f"{row['mean_corrected_ess_over_n']:.6f} | "
            f"{row['usable_seed_count']} | {str(row['overlap_usable']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def write_terminal_lambda_sweep_figure(
    aggregate: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    configuration_rows = aggregate["configuration_rows"]
    aggregate_rows = aggregate["aggregate_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    x = [float(row["lambda_final"]) for row in configuration_rows]
    mean_ess = [float(row["mean_corrected_ess_over_n"]) for row in configuration_rows]
    min_ess = [float(row["minimum_corrected_ess_over_n"]) for row in configuration_rows]
    axes[0].plot(x, mean_ess, marker="o", label="mean across seeds")
    axes[0].plot(x, min_ess, marker="x", linestyle="--", label="minimum across seeds")
    axes[0].axhline(
        float(aggregate["diagnostic_thresholds"]["min_corrected_ess_over_n"]),
        color="0.45",
        linestyle=":",
        label="diagnostic minimum",
    )
    axes[0].set_xlabel("terminal lambda")
    axes[0].set_ylabel("corrected ESS/N")
    axes[0].set_title("Base-correction overlap")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    thresholds = sorted({float(row["threshold"]) for row in aggregate_rows})
    positive = [
        float(row["mean_estimated_probability_under_p_m"])
        for row in aggregate_rows
        if float(row["mean_estimated_probability_under_p_m"]) > 0.0
    ]
    use_log = bool(positive and max(positive) / min(positive) >= 100.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for threshold_index, threshold in enumerate(thresholds):
        color = colors[threshold_index % len(colors)]
        rows = sorted(
            (row for row in aggregate_rows if float(row["threshold"]) == threshold),
            key=lambda row: float(row["lambda_final"]),
        )
        values = [float(row["mean_estimated_probability_under_p_m"]) for row in rows]
        lambda_values = [float(row["lambda_final"]) for row in rows]
        if use_log:
            positive_pairs = [
                (lambda_value, value)
                for lambda_value, value in zip(lambda_values, values)
                if value > 0.0
            ]
            if positive_pairs:
                axes[1].plot(
                    [pair[0] for pair in positive_pairs],
                    [pair[1] for pair in positive_pairs],
                    marker="o",
                    color=color,
                    label=f"ARI >= {threshold:g}",
                )
            zero_x = [
                lambda_value
                for lambda_value, value in zip(lambda_values, values)
                if value == 0.0
            ]
            if zero_x:
                axes[1].scatter(
                    zero_x,
                    [0.025] * len(zero_x),
                    marker="x",
                    transform=axes[1].get_xaxis_transform(),
                    clip_on=False,
                    color=color,
                    s=24,
                    label=(f"ARI >= {threshold:g}" if not positive_pairs else None),
                )
        else:
            axes[1].plot(
                lambda_values,
                values,
                marker="o",
                color=color,
                label=f"ARI >= {threshold:g}",
            )
    if use_log:
        axes[1].set_yscale("log")
        axes[1].text(
            0.01,
            0.02,
            "x at baseline: exact zero (omitted from log transform)",
            transform=axes[1].transAxes,
            fontsize=7,
            va="bottom",
        )
    axes[1].set_xlabel("terminal lambda")
    axes[1].set_ylabel("estimated P under p_M")
    axes[1].set_title("Final-population SNIS tail estimates")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _run_seed(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    seed: int,
    lambdas: tuple[float, ...],
    thresholds: tuple[float, ...],
    repository_commit: str,
    configuration_label: str | None = None,
    min_corrected_ess_over_n: float,
    min_raw_event_fraction: float,
    max_raw_event_fraction: float,
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
        "ari_convention": args.ari_convention,
        "observable_backend": "exact_cached",
        "requested_device": args.device,
        "model_device": str(next(model.parameters()).device),
        "overlap_diagnostic": {
            "min_corrected_ess_over_n": min_corrected_ess_over_n,
            "min_raw_event_fraction": min_raw_event_fraction,
            "max_raw_event_fraction": max_raw_event_fraction,
        },
    }
    fingerprint = configuration_fingerprint(
        {**essential_config, "repository_commit": repository_commit}
    )
    output_dir = tail_run_directory(
        args.output_root,
        args.experiment_label,
        seed,
        configuration_label=configuration_label,
    )
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

    observable = create_observable(
        "ari",
        mode="raw",
        backend="exact_cached",
        convention=args.ari_convention,
    )
    progress = tqdm(total=max(0, len(lambdas) - 1), desc=f"SMC seed {seed}")

    def update_progress(row: dict[str, Any]) -> None:
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
    if bool(result.summary.get("resample_final")) != bool(args.resample_final):
        raise RuntimeError("final-resampling mode does not match the requested configuration")
    if len(result.samples) != config.num_particles:
        raise RuntimeError("final SMC sample count does not match num_particles")

    estimates = estimate_base_tails_from_final_samples(
        result.samples,
        lambda_final=float(lambdas[-1]),
        thresholds=thresholds,
    ).to_dict()
    for row in estimates["tails"]:
        row["overlap_usable"] = overlap_is_usable(
            float(estimates["corrected_importance_ess_over_n"]),
            float(row["raw_tilted_fraction"]),
            min_corrected_ess_over_n=min_corrected_ess_over_n,
            min_raw_event_fraction=min_raw_event_fraction,
            max_raw_event_fraction=max_raw_event_fraction,
        )
    summary = {
        "runtime_seconds": float(result.summary["runtime_total_seconds"]),
        "lambda_final": float(lambdas[-1]),
        "num_particles": config.num_particles,
        "mean_phi": float(result.summary["weighted_mean_ari"]),
        "max_phi": max(float(sample["phi"]) for sample in result.samples),
        "smc_ess_over_n": float(
            result.summary.get(
                "final_weight_ess_over_n_before_resampling",
                result.summary["final_ess_over_n"],
            )
        ),
        **estimates,
    }
    payload = {
        "schema_version": 2,
        "algorithm": "suffix_direct_mh_smc",
        "estimator": "self_normalized_importance_sampling",
        "finite_particle_bias": "generally biased; consistent as particle count increases",
        "seed": seed,
        "configuration_label": configuration_label,
        "configuration_fingerprint": fingerprint,
        "repository_commit": repository_commit,
        "summary": summary,
        "config": essential_config,
        "samples": result.samples,
    }
    write_json(run_path, payload)
    if configuration_label is None:
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
    if args.lambda_schedules_json:
        lambda_schedules = parse_lambda_schedules(args.lambda_schedules_json)
    else:
        lambdas = parse_float_list(args.lambdas)
        if not math.isclose(lambdas[0], 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("--lambdas must start at zero")
        lambda_schedules = {"": lambdas}
    thresholds = parse_float_list(args.tail_thresholds)
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("--tail-thresholds must be sorted in non-decreasing order")
    if args.skip_completed and args.overwrite:
        raise ValueError("use either --skip-completed or --overwrite, not both")
    overlap_is_usable(
        args.min_corrected_ess_over_n,
        args.min_raw_event_fraction,
        min_corrected_ess_over_n=args.min_corrected_ess_over_n,
        min_raw_event_fraction=args.min_raw_event_fraction,
        max_raw_event_fraction=args.max_raw_event_fraction,
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
    repository_commit = git_commit(ROOT)
    run_paths: list[Path] = []
    is_sweep = bool(args.lambda_schedules_json)
    for configuration_label, schedule in lambda_schedules.items():
        visible_label = configuration_label or "single schedule"
        print(
            f"configuration: {visible_label} (terminal lambda {schedule[-1]:g})",
            flush=True,
        )
        for seed in seeds:
            run_paths.append(
                _run_seed(
                    args,
                    model,
                    tokenizer,
                    seed=seed,
                    lambdas=schedule,
                    thresholds=thresholds,
                    repository_commit=repository_commit,
                    configuration_label=configuration_label if is_sweep else None,
                    min_corrected_ess_over_n=args.min_corrected_ess_over_n,
                    min_raw_event_fraction=args.min_raw_event_fraction,
                    max_raw_event_fraction=args.max_raw_event_fraction,
                )
            )

    if is_sweep:
        run_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in run_paths
        ]
        aggregate = {
            "schema_version": 1,
            "algorithm": "suffix_direct_mh_smc",
            "experiment_label": args.experiment_label,
            "estimator": "self_normalized_importance_sampling",
            "finite_particle_bias": (
                "generally biased; consistent as particle count increases"
            ),
            "repository_commit": repository_commit,
            **aggregate_terminal_lambda_runs(
                run_payloads,
                min_corrected_ess_over_n=args.min_corrected_ess_over_n,
                min_raw_event_fraction=args.min_raw_event_fraction,
                max_raw_event_fraction=args.max_raw_event_fraction,
            ),
        }
        experiment_dir = args.output_root / args.experiment_label
        aggregate_path = experiment_dir / "aggregate.json"
        write_json(aggregate_path, aggregate)
        (experiment_dir / "report.md").write_text(
            render_terminal_lambda_sweep_report(aggregate),
            encoding="utf-8",
        )
        write_terminal_lambda_sweep_figure(
            aggregate,
            experiment_dir / "terminal_lambda_sweep.png",
        )
        print(aggregate_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
