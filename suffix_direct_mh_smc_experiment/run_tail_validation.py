"""Validate component-averaged SMC tails against reproducible IID direct data."""

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
from rea_llms.data_reuse_audit import (  # noqa: E402
    ArchiveAuditRecord,
    classify_component_archives,
    discover_component_archives,
    prompt_sha256,
    selected_archive_paths,
)
from rea_llms.current_direct_reference import (  # noqa: E402
    load_scientifically_compatible_summary,
)
from rea_llms.observables import (  # noqa: E402
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTIONS,
    create_observable,
)
from rea_llms.scoring import ari_decoding_description  # noqa: E402
from rea_llms.tail_estimation import (  # noqa: E402
    MixtureComponentPopulation,
    diagnose_component_reverse_estimator,
    estimate_base_tails_by_component_average,
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
    generated_token_count: int | None
    reused: bool
    path: Path


@dataclass
class TailValidationResult:
    """In-memory validation data used by the Colab notebook."""

    comparison: pd.DataFrame
    component_diagnostics: pd.DataFrame
    component_status: pd.DataFrame
    complete_grouped: dict[int, list[CompletedComponent]]
    specs: tuple[ComponentSpec, ...]
    thresholds: tuple[float, ...]
    direct_summary: dict[str, Any] | None
    generated_component_count: int
    reusable_component_count: int
    planned_missing_count: int


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
    metadata = {
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
        "resample_final": True,
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
    # Existing current-convention component archives predate this explicit
    # field. Preserve their fingerprints while separating every new convention.
    if args.ari_convention != ARI_CONVENTION_CURRENT:
        metadata.update(
            {
                "ari_convention": args.ari_convention,
                "prompt_sha256": prompt_sha256(args.prompt),
                "decoding_convention": ari_decoding_description(
                    args.ari_convention
                ),
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": (
                    "tokenizer default; not passed explicitly"
                ),
                "eos_stopping": False,
                "top_k": None,
                "top_p": None,
            }
        )
    return metadata


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
    generated_token_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        arrays = {
            "metadata_json": np.asarray(canonical_json(metadata)),
            "raw_ari": np.asarray(raw_ari, dtype=np.float64),
            "phi": np.asarray(phi, dtype=np.float64),
            "final_log_weights": np.asarray(final_log_weights, dtype=np.float64),
            "log_normalizing_constant": np.asarray(
                log_normalizing_constant,
                dtype=np.float64,
            ),
            "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        }
        if generated_token_count is not None:
            arrays["generated_token_count"] = np.asarray(
                generated_token_count,
                dtype=np.int64,
            )
        np.savez_compressed(
            handle,
            **arrays,
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
        generated_token_count = (
            int(archive["generated_token_count"].item())
            if "generated_token_count" in archive
            else None
        )
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
    if expected_metadata.get("resample_final") is True:
        uniform_log_weight = -math.log(expected_count)
        if not np.allclose(
            final_log_weights,
            uniform_log_weight,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(
                f"cached final-resampled component is not uniformly weighted: {path}"
            )
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
        generated_token_count=generated_token_count,
        reused=reused,
        path=path,
    )


def _audit_component_cache(
    component_root: Path,
    plans: Sequence[
        tuple[int, int, ComponentSpec, int, tuple[float, ...], dict[str, Any], Path]
    ],
) -> list[ArchiveAuditRecord]:
    expected = {path.stem: metadata for *_, metadata, path in plans}
    return classify_component_archives(
        discover_component_archives(component_root),
        expected,
    )


def _load_selected_audit_components(
    records: Sequence[ArchiveAuditRecord],
    plans: Sequence[
        tuple[int, int, ComponentSpec, int, tuple[float, ...], dict[str, Any], Path]
    ],
    *,
    reused_run_ids: set[str] | None = None,
) -> list[CompletedComponent]:
    selected = selected_archive_paths(records)
    plan_by_run_id = {
        path.stem: (spec, metadata)
        for _, _, spec, _, _, metadata, path in plans
    }
    completed: list[CompletedComponent] = []
    reused_ids = reused_run_ids or set()
    for run_id, archive_path in sorted(selected.items()):
        spec, expected_metadata = plan_by_run_id[run_id]
        completed.append(
            _load_component_archive(
                archive_path,
                expected_metadata=expected_metadata,
                spec=spec,
                reused=run_id in reused_ids,
            )
        )
    return completed


def _load_expected_components(
    plans: Sequence[
        tuple[int, int, ComponentSpec, int, tuple[float, ...], dict[str, Any], Path]
    ],
) -> list[CompletedComponent]:
    """Load only exact expected cache paths without scanning a remote directory."""

    completed: list[CompletedComponent] = []
    for _, _, spec, _, _, metadata, path in plans:
        if path.is_file():
            completed.append(
                _load_component_archive(
                    path,
                    expected_metadata=metadata,
                    spec=spec,
                    reused=True,
                )
            )
    return completed


def _component_status_frame(
    plans: Sequence[
        tuple[int, int, ComponentSpec, int, tuple[float, ...], dict[str, Any], Path]
    ],
    completed: Sequence[CompletedComponent],
    complete_grouped: dict[int, list[CompletedComponent]],
) -> pd.DataFrame:
    completed_by_id = {component.run_id: component for component in completed}
    included_ids = {
        component.run_id
        for components in complete_grouped.values()
        for component in components
    }
    rows = []
    for seed_group, _, spec, seed, _, _, path in plans:
        component = completed_by_id.get(path.stem)
        rows.append(
            {
                "seed_group": seed_group,
                "cap": spec.cap,
                "lambda_final": spec.lambda_final,
                "seed": seed,
                "status": (
                    "missing"
                    if component is None
                    else "reused"
                    if component.reused
                    else "generated"
                ),
                "included_in_mixture": path.stem in included_ids,
            }
        )
    return pd.DataFrame(rows)


def _write_audit_csv(
    path: Path,
    records: Sequence[ArchiveAuditRecord],
    *,
    included_paths: set[Path] | None = None,
) -> pd.DataFrame:
    rows = [record.to_row() for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "path",
                "run_id",
                "expected_run_id",
                "classification",
                "selected_for_mixture",
                "reasons",
            ]
        )
    resolved_included = {
        candidate.resolve() for candidate in (included_paths or set())
    }
    frame["included_in_complete_mixture"] = [
        Path(value).resolve() in resolved_included for value in frame["path"]
    ]
    frame.to_csv(path, index=False)
    return frame


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
        convention=args.ari_convention,
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
    if result.summary.get("resample_final") is not True:
        raise RuntimeError("suffix SMC did not resample its final population")
    if result.summary.get("algorithm_version") != ALGORITHM_VERSION:
        raise RuntimeError("suffix SMC algorithm version does not match cache metadata")
    raw_ari = np.asarray([row["raw_ari"] for row in result.samples], dtype=np.float64)
    phi = np.asarray([row["phi"] for row in result.samples], dtype=np.float64)
    final_log_weights = np.asarray(
        [row["log_weight"] for row in result.samples], dtype=np.float64
    )
    log_z = float(result.summary["log_normalizing_constant_estimate"])
    runtime = float(result.summary["runtime_total_seconds"])
    generated_token_count = (
        args.num_particles * args.max_new_tokens
        + int(result.summary["total_regenerated_tokens"])
    )
    _save_component_archive(
        output_path,
        metadata=metadata,
        raw_ari=raw_ari,
        phi=phi,
        final_log_weights=final_log_weights,
        log_normalizing_constant=log_z,
        runtime_seconds=runtime,
        generated_token_count=generated_token_count,
    )
    return _load_component_archive(
        output_path,
        expected_metadata=metadata,
        spec=spec,
        reused=False,
    )


def _component_coefficients(
    specs: Sequence[ComponentSpec],
    _completed: Sequence[CompletedComponent],
) -> list[float] | None:
    if all(spec.rho is None for spec in specs):
        return None
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
    return estimate_base_tails_by_component_average(
        populations,
        thresholds=thresholds,
        component_coefficients=_component_coefficients(specs, components),
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
        generated_counts = [
            component.generated_token_count for component in components
        ]
        generated_token_count = (
            None
            if any(value is None for value in generated_counts)
            else int(
                sum(int(value) for value in generated_counts if value is not None)
            )
        )
        for threshold, probability in zip(thresholds, probabilities):
            direct = direct_rows[float(threshold)]
            direct_probability = float(direct["empirical_probability"])
            rows.append(
                {
                    "row_type": "seed_group",
                    "seed_group": seed_group,
                    "threshold": float(threshold),
                    "direct_sample_count": direct_n,
                    "direct_event_count": int(direct["count"]),
                    "direct_probability": direct_probability,
                    "direct_wilson_95_lower": float(direct["wilson_95_lower"]),
                    "direct_wilson_95_upper": float(direct["wilson_95_upper"]),
                    "mixture_probability": float(probability),
                    "mixture_probability_std": None,
                    "difference_mixture_minus_direct": float(
                        probability - direct_probability
                    ),
                    "weight_ess": ess,
                    "weight_ess_over_n": ess_over_n,
                    "runtime_seconds": runtime,
                    "generated_token_count": generated_token_count,
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
        mean_probability = float(statistics.fmean(probabilities))
        first = threshold_rows[0]
        direct_probability = float(first["direct_probability"])
        rows.append(
            {
                "row_type": "aggregate",
                "seed_group": None,
                "threshold": float(threshold),
                "direct_sample_count": direct_n,
                "direct_event_count": int(first["direct_event_count"]),
                "direct_probability": direct_probability,
                "direct_wilson_95_lower": float(first["direct_wilson_95_lower"]),
                "direct_wilson_95_upper": float(first["direct_wilson_95_upper"]),
                "mixture_probability": mean_probability,
                "mixture_probability_std": (
                    float(statistics.stdev(probabilities))
                    if len(probabilities) > 1
                    else None
                ),
                "difference_mixture_minus_direct": (
                    mean_probability - direct_probability
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
                "generated_token_count": first["generated_token_count"],
            }
        )
    return rows, estimates_by_group


def _component_diagnostic_rows(
    components: Sequence[CompletedComponent],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in sorted(
        components,
        key=lambda item: (item.seed_group, item.spec.cap, item.spec.lambda_final),
    ):
        compatible_thresholds = [
            float(threshold)
            for threshold in thresholds
            if float(threshold) <= component.spec.cap
        ]
        if not compatible_thresholds:
            continue
        diagnostic = diagnose_component_reverse_estimator(
            component.raw_ari,
            component.final_log_weights,
            cap=component.spec.cap,
            lambda_final=component.spec.lambda_final,
            thresholds=compatible_thresholds,
        )
        reverse_log_z = float(
            diagnostic.reverse_log_normalizing_constant.detach().cpu()
        )
        ess = float(diagnostic.corrected_ess.detach().cpu())
        ess_over_n = float(diagnostic.corrected_ess_over_n.detach().cpu())
        probabilities = diagnostic.estimated_probabilities.detach().cpu().tolist()
        counts = diagnostic.raw_counts.detach().cpu().tolist()
        for threshold, probability, count in zip(
            compatible_thresholds,
            probabilities,
            counts,
        ):
            rows.append(
                {
                    "run_id": component.run_id,
                    "seed_group": component.seed_group,
                    "cap": component.spec.cap,
                    "lambda_final": component.spec.lambda_final,
                    "threshold": threshold,
                    "component_probability": float(probability),
                    "corrected_ess": ess,
                    "corrected_ess_over_n": ess_over_n,
                    "raw_event_count": int(count),
                    "log_z_smc": component.log_normalizing_constant,
                    "log_z_reverse": reverse_log_z,
                    "delta_log_z_smc_minus_reverse": (
                        component.log_normalizing_constant - reverse_log_z
                    ),
                }
            )
    return rows


def _diagnostic_interpretation(
    diagnostic_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    seed_group_count: int,
    *,
    smoke_mode: bool = False,
) -> list[str]:
    messages: list[str] = []
    if seed_group_count == 1:
        messages.append(
            "Only one seed group is available, so between-run variance and stability "
            "cannot be assessed."
        )
    elif seed_group_count == 0:
        messages.append("No complete seed group is available for SMC estimation.")
    if smoke_mode:
        messages.append(
            "Smoke-mode estimates validate execution only; they are not used to "
            "attribute discrepancies or assess estimator accuracy."
        )
        return messages
    if not diagnostic_rows:
        messages.append(
            "No complete exactly reusable component set is available locally; the "
            "numerical source of the discrepancy remains unidentified."
        )
        return messages

    moderate = [
        row
        for row in diagnostic_rows
        if abs(float(row["lambda_final"])) <= 1.5
        and float(row["corrected_ess_over_n"]) >= 0.2
    ]
    suspect_log_z = [
        row
        for row in moderate
        if abs(float(row["delta_log_z_smc_minus_reverse"])) > 0.5
    ]
    if suspect_log_z:
        messages.append(
            "Some moderate-tilt components have corrected ESS/N >= 0.2 but "
            "|delta log Z| > 0.5; the SMC normalizing-constant estimate needs scrutiny."
        )

    overlap_warning = any(
        float(row["corrected_ess_over_n"]) < 0.05 for row in diagnostic_rows
    )
    if overlap_warning:
        messages.append(
            "At least one reverse correction has ESS/N < 0.05, indicating poor "
            "overlap; its component-wise and reverse-log-Z estimates are unstable."
        )

    component_disagreement = False
    by_threshold: dict[float, list[float]] = {}
    for row in diagnostic_rows:
        by_threshold.setdefault(float(row["threshold"]), []).append(
            float(row["component_probability"])
        )
    for values in by_threshold.values():
        positives = [value for value in values if value > 0.0]
        if len(positives) >= 2 and max(positives) / min(positives) > 3.0:
            component_disagreement = True
    if component_disagreement:
        messages.append(
            "Component-wise estimates differ by more than a factor of three at "
            "one or more thresholds, pointing to overlap or mixing limitations."
        )

    if not aggregate_rows:
        messages.append(
            "No aggregate SMC estimate is available for comparison with IID direct."
        )
        return messages

    relative_differences: list[float] = []
    for row in aggregate_rows:
        direct = float(row["direct_probability"])
        mixture = float(row["mixture_probability"])
        if direct > 0.0:
            relative_differences.append(mixture / direct - 1.0)
    messages.append(
        "SMC/direct relative differences by threshold: "
        + ", ".join(f"{value:+.1%}" for value in relative_differences)
        + "."
    )
    return messages


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate component-averaged suffix direct-MH SMC tail estimates."
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
    parser.add_argument(
        "--ari-convention",
        choices=ARI_CONVENTIONS,
        default=ARI_CONVENTION_CURRENT,
    )
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
    parser.add_argument(
        "--direct-summary",
        type=Path,
        default=None,
        help=(
            "Matching IID direct summary. Defaults to "
            "OUTPUT_ROOT/current_direct_reference/summary.json."
        ),
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Run integration checks without treating probability differences as validation.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Audit and analyze existing compatible archives without loading the model "
            "or generating missing components."
        ),
    )
    parser.add_argument(
        "--skip-completed-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def run_validation(
    args: argparse.Namespace,
    *,
    persist_analysis: bool = True,
    fast_cache: bool = False,
) -> TailValidationResult:
    """Run or load components and return the validation analysis in memory.

    ``fast_cache`` checks only the deterministic paths for the requested
    configurations. It is intended for high-latency Colab Drive mounts. The
    normal CLI keeps the broader archive audit for provenance work.
    """

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

    output_root = args.output_root
    component_root = output_root / "component_runs"
    output_root.mkdir(parents=True, exist_ok=True)
    direct_path = (
        args.direct_summary
        if args.direct_summary is not None
        else output_root / "current_direct_reference" / "summary.json"
    )
    direct_summary, direct_reasons = load_scientifically_compatible_summary(
        direct_path,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        thresholds=thresholds,
        ari_convention=args.ari_convention,
    )
    direct_rows = None
    direct_n = None
    if direct_summary is not None:
        direct_statistics = direct_summary["statistics"]
        direct_rows = {
            float(row["threshold"]): row for row in direct_statistics["tails"]
        }
        direct_n = int(direct_statistics["count"])
        print(
            f"matching {args.ari_convention} direct reference: compatible "
            f"N={direct_n}",
            flush=True,
        )
    else:
        print(
            f"matching {args.ari_convention} direct reference unavailable: "
            + "; ".join(direct_reasons),
            flush=True,
        )
    if persist_analysis or not fast_cache:
        print(
            "direct-reference preflight: "
            f"compatible={direct_summary is not None}, "
            f"smoke_mode={args.smoke_mode}",
            flush=True,
        )

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

    audit_records: list[ArchiveAuditRecord] = []
    cached_components: list[CompletedComponent] = []
    if fast_cache:
        cached_components = _load_expected_components(plans)
        selected_before_run = {
            component.run_id: component.path for component in cached_components
        }
    else:
        audit_records = _audit_component_cache(component_root, plans)
        selected_before_run = selected_archive_paths(audit_records)
    if args.audit_only:
        missing = []
    elif args.skip_completed_runs:
        missing = [plan for plan in plans if plan[6].stem not in selected_before_run]
    else:
        missing = list(plans)
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
    generated_components: list[CompletedComponent] = []
    progress = tqdm(missing, desc="Independent SMC components") if missing else missing
    for plan in progress:
        seed_group, component_index, spec, seed, schedule, metadata, path = plan
        generated_components.append(
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

    if fast_cache:
        completed = (
            cached_components + generated_components
            if args.skip_completed_runs
            else generated_components
        )
    else:
        audit_records = _audit_component_cache(component_root, plans)
        completed = _load_selected_audit_components(
            audit_records,
            plans,
            reused_run_ids=(
                set(selected_before_run) if args.skip_completed_runs else set()
            ),
        )
    grouped: dict[int, list[CompletedComponent]] = {}
    index_by_spec = {
        (component_seed(seed_group, index), seed_group): index
        for seed_group in seed_groups
        for index in range(len(specs))
    }
    for component in completed:
        grouped.setdefault(component.seed_group, []).append(component)
    complete_grouped: dict[int, list[CompletedComponent]] = {}
    for seed_group, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: index_by_spec[(row.seed, seed_group)])
        if len(rows) == len(specs):
            complete_grouped[seed_group] = rows
        elif not args.audit_only:
            raise RuntimeError(f"seed group {seed_group} is incomplete")
    included_paths = {
        component.path
        for components in complete_grouped.values()
        for component in components
    }
    if fast_cache:
        audit_frame = _component_status_frame(plans, completed, complete_grouped)
    elif persist_analysis:
        audit_frame = _write_audit_csv(
            output_root / "data_reuse_audit.csv",
            audit_records,
            included_paths=included_paths,
        )
    else:
        audit_frame = pd.DataFrame([record.to_row() for record in audit_records])

    comparison_rows: list[dict[str, Any]] = []
    if complete_grouped and direct_rows is not None and direct_n is not None:
        comparison_rows, _ = _comparison_rows(
            complete_grouped,
            specs,
            thresholds,
            direct_rows,
            direct_n,
        )
    comparison = pd.DataFrame(
        comparison_rows,
        columns=[
            "row_type",
            "seed_group",
            "threshold",
            "direct_sample_count",
            "direct_event_count",
            "direct_probability",
            "direct_wilson_95_lower",
            "direct_wilson_95_upper",
            "mixture_probability",
            "mixture_probability_std",
            "difference_mixture_minus_direct",
            "weight_ess",
            "weight_ess_over_n",
            "runtime_seconds",
            "generated_token_count",
        ],
    )
    if persist_analysis:
        comparison.to_csv(output_root / "comparison.csv", index=False)
    aggregate_rows = [
        row for row in comparison_rows if row["row_type"] == "aggregate"
    ]
    component_diagnostics = _component_diagnostic_rows(completed, thresholds)
    component_diagnostics_frame = pd.DataFrame(
        component_diagnostics,
        columns=[
            "run_id",
            "seed_group",
            "cap",
            "lambda_final",
            "threshold",
            "component_probability",
            "corrected_ess",
            "corrected_ess_over_n",
            "raw_event_count",
            "log_z_smc",
            "log_z_reverse",
            "delta_log_z_smc_minus_reverse",
        ],
    )
    if persist_analysis:
        component_diagnostics_frame.to_csv(
            output_root / "component_diagnostics.csv",
            index=False,
        )
    interpretation = _diagnostic_interpretation(
        component_diagnostics,
        aggregate_rows,
        len(complete_grouped),
        smoke_mode=args.smoke_mode,
    )
    if persist_analysis:
        for message in interpretation:
            print(f"diagnostic: {message}", flush=True)
    reusable_count = len(completed)
    planned_missing_count = len(plans) - reusable_count
    included_count = sum(len(rows) for rows in complete_grouped.values())
    cache_label = "requested components" if fast_cache else "archives"
    print(
        f"found {len(audit_frame)} {cache_label}; {reusable_count} available; "
        f"{included_count} included in complete mixtures; {planned_missing_count} "
        f"planned components missing; generated {len(missing)} in "
        f"{time.perf_counter() - started:.1f} s",
        flush=True,
    )
    if persist_analysis:
        print((output_root / "comparison.csv").resolve(), flush=True)
    return TailValidationResult(
        comparison=comparison,
        component_diagnostics=component_diagnostics_frame,
        component_status=audit_frame,
        complete_grouped=complete_grouped,
        specs=specs,
        thresholds=thresholds,
        direct_summary=direct_summary,
        generated_component_count=len(missing),
        reusable_component_count=reusable_count,
        planned_missing_count=planned_missing_count,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_validation(parse_args(argv))


if __name__ == "__main__":
    main()
