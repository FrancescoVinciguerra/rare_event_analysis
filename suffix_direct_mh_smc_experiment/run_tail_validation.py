"""Validate deterministic-mixture SMC tail estimates against official direct data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from rea_llms.official_direct_data import boxplot_statistics  # noqa: E402
from rea_llms.tail_estimation import (  # noqa: E402
    MixtureComponentPopulation,
    estimate_base_tails_from_deterministic_mixture,
)
from suffix_direct_mh_smc_experiment.smc_sampler import (  # noqa: E402
    ALGORITHM_VERSION,
    SuffixDirectMHSMCConfig,
    parse_float_list,
    parse_int_list,
    run_suffix_direct_mh_smc,
)


DEFAULT_COMPONENTS = (
    {"cap": 6.0, "lambda": -0.5},
    {"cap": 6.0, "lambda": -1.0},
    {"cap": 6.0, "lambda": -1.5},
    {"cap": 7.0, "lambda": -1.0},
    {"cap": 7.0, "lambda": -1.5},
    {"cap": 7.0, "lambda": -2.0},
    {"cap": 8.0, "lambda": -1.5},
    {"cap": 8.0, "lambda": -2.0},
    {"cap": 8.0, "lambda": -2.5},
    {"cap": 10.0, "lambda": -2.0},
    {"cap": 10.0, "lambda": -2.5},
    {"cap": 10.0, "lambda": -3.0},
    {"cap": 10.0, "lambda": -3.5},
)
CACHE_SCHEMA_VERSION = 1
SEED_STRIDE = 100_000


@dataclass(frozen=True)
class ComponentSpec:
    cap: float
    lambda_final: float
    rho: float | None = None


@dataclass(frozen=True)
class CompletedComponent:
    spec: ComponentSpec
    seed_group: int
    seed: int
    run_id: str
    raw_ari: np.ndarray
    phi: np.ndarray
    final_log_weights: np.ndarray
    log_normalizing_constant: float
    runtime_seconds: float
    reused: bool
    path: Path


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def configuration_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def component_seed(seed_group: int, component_index: int) -> int:
    """Map a non-negative seed group and component index injectively to one seed."""

    if seed_group < 0:
        raise ValueError("seed groups must be non-negative")
    if not 0 <= component_index < SEED_STRIDE:
        raise ValueError(f"component_index must be in [0, {SEED_STRIDE})")
    seed = seed_group * SEED_STRIDE + component_index
    if seed >= 2**63:
        raise ValueError("derived component seed exceeds the supported PyTorch range")
    return seed


def build_lambda_schedule(lambda_final: float, step: float) -> tuple[float, ...]:
    """Build a non-increasing schedule from zero to an exact terminal lambda."""

    terminal = float(lambda_final)
    step = float(step)
    if not math.isfinite(terminal) or terminal > 0.0:
        raise ValueError("terminal lambda must be finite and non-positive")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("lambda step must be finite and positive")
    if math.isclose(terminal, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return (0.0,)
    full_steps = int(math.floor(abs(terminal) / step + 1e-12))
    values = [0.0] + [-step * index for index in range(1, full_steps + 1)]
    if not math.isclose(values[-1], terminal, rel_tol=0.0, abs_tol=1e-12):
        values.append(terminal)
    else:
        values[-1] = terminal
    if any(current > previous + 1e-12 for previous, current in zip(values, values[1:])):
        raise AssertionError("constructed lambda schedule is not non-increasing")
    return tuple(float(value) for value in values)


def parse_components(value: str) -> tuple[ComponentSpec, ...]:
    try:
        rows = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--components-json must be a JSON list") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("--components-json must contain a non-empty list")
    components: list[ComponentSpec] = []
    for row in rows:
        if not isinstance(row, dict) or "cap" not in row or "lambda" not in row:
            raise ValueError("each component requires cap and lambda")
        cap = float(row["cap"])
        lambda_final = float(row["lambda"])
        rho = None if row.get("rho") is None else float(row["rho"])
        if not math.isfinite(cap):
            raise ValueError("component caps must be finite")
        if not math.isfinite(lambda_final) or lambda_final > 0.0:
            raise ValueError("component lambdas must be finite and non-positive")
        if rho is not None and (not math.isfinite(rho) or rho <= 0.0):
            raise ValueError("explicit component rho values must be positive")
        components.append(ComponentSpec(cap=cap, lambda_final=lambda_final, rho=rho))
    has_rho = [component.rho is not None for component in components]
    if any(has_rho) and not all(has_rho):
        raise ValueError("specify rho for every component or for none")
    return tuple(components)


def _component_metadata(
    args: argparse.Namespace,
    spec: ComponentSpec,
    *,
    seed_group: int,
    component_index: int,
    seed: int,
    schedule: Sequence[float],
) -> dict[str, Any]:
    suffix_sizes = parse_int_list(args.suffix_sizes)
    suffix_rhos = parse_float_list(args.suffix_rhos) if args.suffix_rhos else None
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "algorithm": "suffix_direct_mh_smc",
        "algorithm_version": ALGORITHM_VERSION,
        "mutation_kernel": "suffix_direct_mh",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "proposal_temperature": 1.0,
        "generation_temperature": 1.0,
        "fixed_completion_length": True,
        "ari_mode": "capped",
        "ari_cap": float(spec.cap),
        "final_evaluation": "raw_ari",
        "observable_backend": "exact_cached",
        "lambdas": [float(value) for value in schedule],
        "lambda_final": float(spec.lambda_final),
        "num_particles": int(args.num_particles),
        "mcmc_steps_per_level": int(args.mcmc_steps_per_level),
        "ess_threshold": float(args.ess_threshold),
        "resampling": True,
        "resampling_method": "systematic",
        "resample_final": False,
        "suffix_mode": "stochastic",
        "suffix_sizes": list(suffix_sizes),
        "suffix_rhos": list(suffix_rhos) if suffix_rhos is not None else None,
        "require_global_irreducibility": True,
        "batch_size": int(args.batch_size),
        "track_discovery_diagnostics": False,
        "seed_group": int(seed_group),
        "component_index": int(component_index),
        "seed": int(seed),
    }


def component_run_id(metadata: dict[str, Any]) -> str:
    fingerprint = configuration_fingerprint(metadata)
    cap = f"{float(metadata['ari_cap']):g}".replace(".", "p")
    lam = f"{abs(float(metadata['lambda_final'])):g}".replace(".", "p")
    return (
        f"group_{metadata['seed_group']}_component_{metadata['component_index']:03d}_"
        f"cap_{cap}_lambda_m{lam}_seed_{metadata['seed']}_{fingerprint[:12]}"
    )


def cache_metadata_matches(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    return canonical_json(stored) == canonical_json(expected)


def _save_component_archive(
    path: Path,
    *,
    metadata: dict[str, Any],
    raw_ari: np.ndarray,
    phi: np.ndarray,
    final_log_weights: np.ndarray,
    log_normalizing_constant: float,
    runtime_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(canonical_json(metadata)),
            raw_ari=np.asarray(raw_ari, dtype=np.float64),
            phi=np.asarray(phi, dtype=np.float64),
            final_log_weights=np.asarray(final_log_weights, dtype=np.float64),
            log_normalizing_constant=np.asarray(log_normalizing_constant, dtype=np.float64),
            runtime_seconds=np.asarray(runtime_seconds, dtype=np.float64),
        )
    os.replace(temporary, path)


def _load_component_archive(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
    spec: ComponentSpec,
    reused: bool,
) -> CompletedComponent:
    with np.load(path, allow_pickle=False) as archive:
        stored_metadata = json.loads(str(archive["metadata_json"].item()))
        if not cache_metadata_matches(stored_metadata, expected_metadata):
            raise ValueError(f"cached component metadata mismatch: {path}")
        raw_ari = np.asarray(archive["raw_ari"], dtype=np.float64)
        phi = np.asarray(archive["phi"], dtype=np.float64)
        final_log_weights = np.asarray(
            archive["final_log_weights"], dtype=np.float64
        )
        log_z = float(archive["log_normalizing_constant"].item())
        runtime = float(archive["runtime_seconds"].item())
    expected_count = int(expected_metadata["num_particles"])
    if raw_ari.shape != (expected_count,) or phi.shape != raw_ari.shape:
        raise ValueError(f"cached component has an invalid particle shape: {path}")
    if final_log_weights.shape != raw_ari.shape:
        raise ValueError(f"cached component has invalid final weights: {path}")
    if not np.isfinite(raw_ari).all() or not np.isfinite(phi).all():
        raise ValueError(f"cached component contains non-finite ARI: {path}")
    if (
        not np.isfinite(final_log_weights).any()
        or np.isnan(final_log_weights).any()
        or np.isposinf(final_log_weights).any()
    ):
        raise ValueError(f"cached component contains invalid log weights: {path}")
    if not np.allclose(phi, np.minimum(raw_ari, spec.cap), atol=1e-6, rtol=0.0):
        raise ValueError(f"cached capped potential does not equal min(raw_ari, cap): {path}")
    if not math.isfinite(log_z) or not math.isfinite(runtime):
        raise ValueError(f"cached component has invalid scalar diagnostics: {path}")
    return CompletedComponent(
        spec=spec,
        seed_group=int(expected_metadata["seed_group"]),
        seed=int(expected_metadata["seed"]),
        run_id=path.stem,
        raw_ari=raw_ari,
        phi=phi,
        final_log_weights=final_log_weights,
        log_normalizing_constant=log_z,
        runtime_seconds=runtime,
        reused=reused,
        path=path,
    )


def _run_component(
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    spec: ComponentSpec,
    schedule: tuple[float, ...],
    seed_group: int,
    component_index: int,
    seed: int,
    metadata: dict[str, Any],
    output_path: Path,
) -> CompletedComponent:
    observable = create_observable(
        "ari",
        mode="capped",
        cap=spec.cap,
        backend="exact_cached",
    )
    config = SuffixDirectMHSMCConfig(
        num_particles=args.num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=schedule,
        mcmc_steps_per_level=args.mcmc_steps_per_level,
        ess_threshold=args.ess_threshold,
        resampling=True,
        suffix_mode="stochastic",
        suffix_sizes=parse_int_list(args.suffix_sizes),
        suffix_rhos=(parse_float_list(args.suffix_rhos) if args.suffix_rhos else None),
        batch_size=args.batch_size,
        seed=seed,
        tail_thresholds=parse_float_list(args.thresholds),
    )
    progress = tqdm(
        total=max(0, len(schedule) - 1),
        desc=f"cap={spec.cap:g}, lambda={spec.lambda_final:g}, seed={seed}",
        leave=False,
    )

    def update_progress(row: dict[str, Any]) -> None:
        progress.set_postfix(
            level_lambda=f"{row['lambda']:g}",
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
            track_discovery_diagnostics=False,
        )
    finally:
        progress.close()
    if result.summary.get("resample_final") is not False:
        raise RuntimeError("suffix SMC unexpectedly resampled its final population")
    if result.summary.get("algorithm_version") != ALGORITHM_VERSION:
        raise RuntimeError("suffix SMC algorithm version does not match cache metadata")
    raw_ari = np.asarray([row["raw_ari"] for row in result.samples], dtype=np.float64)
    phi = np.asarray([row["phi"] for row in result.samples], dtype=np.float64)
    final_log_weights = np.asarray(
        [row["log_weight"] for row in result.samples], dtype=np.float64
    )
    log_z = float(result.summary["log_normalizing_constant_estimate"])
    runtime = float(result.summary["runtime_total_seconds"])
    _save_component_archive(
        output_path,
        metadata=metadata,
        raw_ari=raw_ari,
        phi=phi,
        final_log_weights=final_log_weights,
        log_normalizing_constant=log_z,
        runtime_seconds=runtime,
    )
    return _load_component_archive(
        output_path,
        expected_metadata=metadata,
        spec=spec,
        reused=False,
    )


def _reference_summary_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    matches = sorted(ROOT.glob("results/**/reference_direct_sampling_summary.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one reference_direct_sampling_summary.json; "
            f"found {matches}"
        )
    return matches[0]


def _load_direct_reference(
    path: Path,
    thresholds: Sequence[float],
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    direct = payload.get("direct_ari", {})
    if direct.get("ari_representation") != "raw":
        raise ValueError("the direct reference must contain raw ARI values")
    rows = {float(row["threshold"]): row for row in direct.get("tails", [])}
    missing = [float(threshold) for threshold in thresholds if float(threshold) not in rows]
    if missing:
        raise ValueError(
            f"direct reference is missing thresholds {missing}; update its single summary first"
        )
    return payload, rows


def _mixture_coefficients(
    specs: Sequence[ComponentSpec],
    completed: Sequence[CompletedComponent],
) -> list[float] | None:
    if all(spec.rho is None for spec in specs):
        return [float(component.raw_ari.size) for component in completed]
    return [float(spec.rho) for spec in specs if spec.rho is not None]


def _estimate_seed_group(
    components: Sequence[CompletedComponent],
    specs: Sequence[ComponentSpec],
    thresholds: Sequence[float],
):
    populations = [
        MixtureComponentPopulation(
            run_id=component.run_id,
            seed=component.seed,
            cap=component.spec.cap,
            lambda_final=component.spec.lambda_final,
            log_normalizing_constant=component.log_normalizing_constant,
            raw_ari=component.raw_ari,
            final_log_weights=component.final_log_weights,
        )
        for component in components
    ]
    return estimate_base_tails_from_deterministic_mixture(
        populations,
        thresholds=thresholds,
        mixture_coefficients=_mixture_coefficients(specs, components),
    )


def _comparison_rows(
    grouped_components: dict[int, list[CompletedComponent]],
    specs: Sequence[ComponentSpec],
    thresholds: Sequence[float],
    direct_rows: dict[float, dict[str, Any]],
    direct_n: int,
) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    rows: list[dict[str, Any]] = []
    estimates_by_group: dict[int, Any] = {}
    for seed_group, components in sorted(grouped_components.items()):
        estimate = _estimate_seed_group(components, specs, thresholds)
        estimates_by_group[seed_group] = estimate
        probabilities = estimate.estimated_probabilities.detach().cpu().tolist()
        ess = float(estimate.corrected_ess.detach().cpu())
        ess_over_n = float(estimate.corrected_ess_over_n.detach().cpu())
        runtime = float(sum(component.runtime_seconds for component in components))
        for threshold, probability in zip(thresholds, probabilities):
            reference = direct_rows[float(threshold)]
            direct_probability = float(reference["empirical_probability"])
            difference = float(probability - direct_probability)
            rows.append(
                {
                    "row_type": "seed_group",
                    "seed_group": seed_group,
                    "threshold": float(threshold),
                    "direct_sample_count": direct_n,
                    "direct_event_count": int(reference["count"]),
                    "direct_probability": direct_probability,
                    "direct_wilson_95_lower": float(reference["wilson_95_lower"]),
                    "direct_wilson_95_upper": float(reference["wilson_95_upper"]),
                    "mixture_probability": float(probability),
                    "mixture_probability_std": None,
                    "difference_hat_minus_direct": difference,
                    "relative_difference": (
                        difference / direct_probability if direct_probability > 0.0 else None
                    ),
                    "weight_ess": ess,
                    "weight_ess_over_n": ess_over_n,
                    "runtime_seconds": runtime,
                }
            )
    for threshold in thresholds:
        threshold_rows = [
            row
            for row in rows
            if row["row_type"] == "seed_group"
            and math.isclose(float(row["threshold"]), float(threshold))
        ]
        probabilities = [float(row["mixture_probability"]) for row in threshold_rows]
        direct_probability = float(threshold_rows[0]["direct_probability"])
        mean_probability = float(statistics.fmean(probabilities))
        difference = mean_probability - direct_probability
        rows.append(
            {
                "row_type": "aggregate",
                "seed_group": None,
                "threshold": float(threshold),
                "direct_sample_count": direct_n,
                "direct_event_count": int(threshold_rows[0]["direct_event_count"]),
                "direct_probability": direct_probability,
                "direct_wilson_95_lower": float(
                    threshold_rows[0]["direct_wilson_95_lower"]
                ),
                "direct_wilson_95_upper": float(
                    threshold_rows[0]["direct_wilson_95_upper"]
                ),
                "mixture_probability": mean_probability,
                "mixture_probability_std": (
                    float(statistics.stdev(probabilities))
                    if len(probabilities) > 1
                    else None
                ),
                "difference_hat_minus_direct": difference,
                "relative_difference": (
                    difference / direct_probability if direct_probability > 0.0 else None
                ),
                "weight_ess": float(
                    statistics.fmean(float(row["weight_ess"]) for row in threshold_rows)
                ),
                "weight_ess_over_n": float(
                    statistics.fmean(
                        float(row["weight_ess_over_n"]) for row in threshold_rows
                    )
                ),
                "runtime_seconds": float(
                    statistics.fmean(
                        float(row["runtime_seconds"]) for row in threshold_rows
                    )
                ),
            }
        )
    return rows, estimates_by_group


def _weighted_box(values: np.ndarray, log_weights: np.ndarray, label: str) -> dict[str, Any]:
    weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
    stats = boxplot_statistics(values, weights=weights)
    return {
        "label": label,
        "q1": float(stats["q1"]),
        "med": float(stats["median"]),
        "q3": float(stats["q3"]),
        "whislo": float(stats["whisker_low"]),
        "whishi": float(stats["whisker_high"]),
        "fliers": [],
    }


def _write_ari_boxplots(
    output_path: Path,
    direct_summary: dict[str, Any],
    grouped_components: dict[int, list[CompletedComponent]],
    specs: Sequence[ComponentSpec],
) -> None:
    direct = direct_summary["direct_ari"]
    direct_box = direct["boxplot"]
    boxes = [
        {
            "label": "direct",
            "q1": float(direct_box["q1"]),
            "med": float(direct_box["median"]),
            "q3": float(direct_box["q3"]),
            "whislo": float(direct_box["whisker_low"]),
            "whishi": float(direct_box["whisker_high"]),
            "fliers": [],
        }
    ]
    for index, spec in enumerate(specs):
        component_rows = [rows[index] for rows in grouped_components.values()]
        values = np.concatenate([row.raw_ari for row in component_rows])
        log_weights = np.concatenate(
            [
                row.final_log_weights - math.log(len(component_rows))
                for row in component_rows
            ]
        )
        boxes.append(
            _weighted_box(
                values,
                log_weights,
                f"c={spec.cap:g}\nlambda={spec.lambda_final:g}",
            )
        )
    fig, ax = plt.subplots(figsize=(max(10.0, 0.78 * len(boxes)), 5.2))
    artists = ax.bxp(boxes, showfliers=False, patch_artist=True)
    artists["boxes"][0].set_facecolor("#4C78A8")
    for box in artists["boxes"][1:]:
        box.set_facecolor("#F58518")
    ax.set_ylabel("raw ARI")
    ax.set_title("Official direct reference and weighted terminal tilted populations")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.01,
        "Tilted boxes use final SMC weights and show proposal exploration, not reconstructed p_M.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_probability_plot(
    output_path: Path,
    aggregate_rows: Sequence[dict[str, Any]],
) -> None:
    rows = sorted(aggregate_rows, key=lambda row: float(row["threshold"]))
    x = np.asarray([float(row["threshold"]) for row in rows])
    direct = np.asarray([float(row["direct_probability"]) for row in rows])
    direct_low = np.asarray([float(row["direct_wilson_95_lower"]) for row in rows])
    direct_high = np.asarray([float(row["direct_wilson_95_upper"]) for row in rows])
    mixture = np.asarray([float(row["mixture_probability"]) for row in rows])
    mixture_std = np.asarray(
        [
            0.0 if row["mixture_probability_std"] is None else float(row["mixture_probability_std"])
            for row in rows
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        x - 0.05,
        direct,
        yerr=np.vstack((direct - direct_low, direct_high - direct)),
        marker="o",
        capsize=3,
        label="official direct (Wilson 95%)",
        color="#4C78A8",
    )
    positive = mixture > 0.0
    if np.any(positive):
        lower = np.minimum(mixture_std[positive], mixture[positive] * 0.999999)
        ax.errorbar(
            x[positive] + 0.05,
            mixture[positive],
            yerr=np.vstack((lower, mixture_std[positive])),
            marker="s",
            capsize=3,
            label="SMC mixture (seed-group SD)",
            color="#F58518",
        )
    if np.any(~positive):
        zero_floor = float(direct_low.min() / 3.0)
        ax.scatter(
            x[~positive] + 0.05,
            np.full(np.count_nonzero(~positive), zero_floor),
            marker="v",
            color="#F58518",
            label="SMC mixture: zero observed estimate",
        )
        ax.set_ylim(bottom=zero_floor / 1.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xlabel("raw ARI threshold")
    ax.set_ylabel("P under base language model")
    ax.set_title("Tail-probability validation")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_report(
    args: argparse.Namespace,
    specs: Sequence[ComponentSpec],
    grouped_components: dict[int, list[CompletedComponent]],
    aggregate_rows: Sequence[dict[str, Any]],
    reference_path: Path,
) -> str:
    total_runs = sum(len(rows) for rows in grouped_components.values())
    reused_runs = sum(row.reused for rows in grouped_components.values() for row in rows)
    lines = [
        "# Tail-probability validation",
        "",
        "The reference is empirical direct sampling, not exact analytic ground truth.",
        "The SMC result is a plug-in deterministic-mixture SNIS estimate because the",
        "component normalizing constants are estimated by SMC.",
        "",
        "## Configuration",
        "",
        f"- Direct reference: `{reference_path}`",
        f"- Direct observations: `{aggregate_rows[0]['direct_sample_count']:,}`",
        f"- Seed groups: `{sorted(grouped_components)}`",
        f"- Particles per component: `{args.num_particles}`",
        f"- MCMC steps per level: `{args.mcmc_steps_per_level}`",
        f"- Lambda step: `{args.lambda_step}`",
        f"- ESS resampling threshold: `{args.ess_threshold}`",
        f"- Independent terminal component runs: `{total_runs}`",
        f"- Reused compatible component runs: `{reused_runs}`",
        "- Final resampling: `false`",
        "",
        "Components per seed group:",
        "",
        "| cap | terminal lambda | mixture rho | reused seed groups |",
        "|---:|---:|---:|:---|",
    ]
    for index, spec in enumerate(specs):
        rho = "n_j / sum(n)" if spec.rho is None else f"{spec.rho:.6g}"
        reused_groups = [
            str(seed_group)
            for seed_group, rows in sorted(grouped_components.items())
            if rows[index].reused
        ]
        reused = ", ".join(reused_groups) if reused_groups else "none"
        lines.append(
            f"| {spec.cap:g} | {spec.lambda_final:g} | {rho} | {reused} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            "| threshold | direct events | direct | Wilson 95% | mixture mean | seed SD | difference (hat-direct) | ESS | ESS/N | mean runtime (s) |",
            "|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        sd = row["mixture_probability_std"]
        lines.append(
            f"| {row['threshold']:g} | {row['direct_event_count']} | "
            f"{row['direct_probability']:.6g} | "
            f"[{row['direct_wilson_95_lower']:.6g}, {row['direct_wilson_95_upper']:.6g}] | "
            f"{row['mixture_probability']:.6g} | "
            f"{('-' if sd is None else f'{sd:.6g}')} | "
            f"{row['difference_hat_minus_direct']:.6g} | "
            f"{row['weight_ess']:.2f} | {row['weight_ess_over_n']:.4f} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Each seed group is the replication unit. Particles within one SMC run",
            "are not treated as independent replicates. The ARI boxplots use final",
            "SMC weights for tilted targets and do not represent reconstructed p_M.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate suffix direct-MH SMC mixture tail estimates."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--components-json",
        default=canonical_json(list(DEFAULT_COMPONENTS)),
    )
    parser.add_argument("--seed-groups", default="200,300,400")
    parser.add_argument("--thresholds", default="6,7,8,10")
    parser.add_argument("--lambda-step", type=float, default=0.25)
    parser.add_argument("--num-particles", type=int, default=256)
    parser.add_argument("--mcmc-steps-per-level", type=int, default=10)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--suffix-sizes", default="1,2,4,8,16,32,64,100")
    parser.add_argument("--suffix-rhos", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/tail_probability_validation"),
    )
    parser.add_argument("--reference-summary", type=Path, default=None)
    parser.add_argument(
        "--skip-completed-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = parse_components(args.components_json)
    seed_groups = parse_int_list(args.seed_groups)
    thresholds = parse_float_list(args.thresholds)
    if len(seed_groups) != len(set(seed_groups)):
        raise ValueError("seed groups must be distinct")
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("thresholds must be sorted")
    if args.num_particles <= 0 or args.max_new_tokens <= 0 or args.batch_size <= 0:
        raise ValueError("particle, token, and batch counts must be positive")
    suffix_sizes = parse_int_list(args.suffix_sizes)
    if any(size > args.max_new_tokens for size in suffix_sizes):
        raise ValueError("suffix sizes cannot exceed max_new_tokens")
    if args.max_new_tokens not in suffix_sizes:
        raise ValueError("suffix sizes must include max_new_tokens for global support")

    reference_path = _reference_summary_path(args.reference_summary)
    direct_summary, direct_rows = _load_direct_reference(reference_path, thresholds)
    output_root = args.output_root
    component_root = output_root / "component_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    plans: list[tuple[int, int, ComponentSpec, int, tuple[float, ...], dict[str, Any], Path]] = []
    for seed_group in seed_groups:
        for component_index, spec in enumerate(specs):
            seed = component_seed(seed_group, component_index)
            schedule = build_lambda_schedule(spec.lambda_final, args.lambda_step)
            metadata = _component_metadata(
                args,
                spec,
                seed_group=seed_group,
                component_index=component_index,
                seed=seed,
                schedule=schedule,
            )
            run_id = component_run_id(metadata)
            plans.append(
                (
                    seed_group,
                    component_index,
                    spec,
                    seed,
                    schedule,
                    metadata,
                    component_root / f"{run_id}.npz",
                )
            )
    if len({item[3] for item in plans}) != len(plans):
        raise AssertionError("derived component seeds are not unique")
    if len({item[6].stem for item in plans}) != len(plans):
        raise AssertionError("derived component run IDs are not unique")

    completed: list[CompletedComponent] = []
    missing = []
    for plan in plans:
        seed_group, _, spec, _, _, metadata, path = plan
        if path.exists() and args.skip_completed_runs:
            completed.append(
                _load_component_archive(
                    path,
                    expected_metadata=metadata,
                    spec=spec,
                    reused=True,
                )
            )
        else:
            missing.append(plan)
    model = tokenizer = None
    if missing:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            args.tokenizer_name,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        model_device = next(model.parameters()).device
        if args.device == "cuda" and model_device.type != "cuda":
            raise RuntimeError("--device cuda requested, but model is not on CUDA")
        print(f"model device: {model_device}", flush=True)
    started = time.perf_counter()
    for plan in tqdm(missing, desc="Independent SMC components"):
        seed_group, component_index, spec, seed, schedule, metadata, path = plan
        completed.append(
            _run_component(
                args,
                model,
                tokenizer,
                spec=spec,
                schedule=schedule,
                seed_group=seed_group,
                component_index=component_index,
                seed=seed,
                metadata=metadata,
                output_path=path,
            )
        )
    grouped: dict[int, list[CompletedComponent]] = {}
    index_by_spec = {
        (component_seed(seed_group, index), seed_group): index
        for seed_group in seed_groups
        for index in range(len(specs))
    }
    for component in completed:
        grouped.setdefault(component.seed_group, []).append(component)
    for seed_group, rows in grouped.items():
        rows.sort(key=lambda row: index_by_spec[(row.seed, seed_group)])
        if len(rows) != len(specs):
            raise RuntimeError(f"seed group {seed_group} is incomplete")

    comparison_rows, _ = _comparison_rows(
        grouped,
        specs,
        thresholds,
        direct_rows,
        int(direct_summary["direct_ari"]["count"]),
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_root / "comparison.csv", index=False)
    aggregate_rows = [
        row for row in comparison_rows if row["row_type"] == "aggregate"
    ]
    _write_ari_boxplots(output_root / "ari_boxplots.png", direct_summary, grouped, specs)
    _write_probability_plot(output_root / "probability_comparison.png", aggregate_rows)
    (output_root / "REPORT.md").write_text(
        _render_report(args, specs, grouped, aggregate_rows, reference_path),
        encoding="utf-8",
    )
    print(
        f"completed {len(completed)} components ({len(completed) - len(missing)} reused) "
        f"in {time.perf_counter() - started:.1f} s",
        flush=True,
    )
    print((output_root / "comparison.csv").resolve(), flush=True)


if __name__ == "__main__":
    main()
