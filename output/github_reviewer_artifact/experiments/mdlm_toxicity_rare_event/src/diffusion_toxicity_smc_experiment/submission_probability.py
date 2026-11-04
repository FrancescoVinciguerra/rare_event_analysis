"""Rare-threshold and pilot-weight analysis for the submission experiment.

The pilot command reads completed direct and SMC artifacts only.  It freezes
component weights before any test population is inspected.  The report command
then applies both uniform and frozen pilot weights to an independent test suite.
Neither command loads a model or generates a completion.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from rea_llms.statistics import wilson_interval

from .configuration import (
    canonical_json,
    decoder_id,
    evaluator_id,
    fingerprint,
    load_yaml,
)
from .generate_direct import direct_run_directory
from .run_smc import smc_run_directory
from .serious_comparison import (
    _boxplot_svg,
    comparison_manifest,
    load_suite,
    resolved_comparison_entries,
)
from .storage import read_jsonl, scan_direct_generations


STUDY_KIND = "mdlm_rare_probability_submission_study_v1"


def _config_path(parent: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else parent / candidate


def load_study(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        study = yaml.safe_load(handle)
    if not isinstance(study, dict):
        raise ValueError(f"submission study must be a mapping: {path}")
    if int(study.get("schema_version", 0)) != 1:
        raise ValueError("submission study schema_version must be 1")
    thresholds = study.get("rare_events", {}).get("toxic_probability_thresholds")
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
            for value in thresholds
        )
    ):
        raise ValueError("rare-event thresholds must be finite values in (0, 1)")
    numeric_thresholds = [float(value) for value in thresholds]
    if numeric_thresholds != sorted(set(numeric_thresholds)):
        raise ValueError("rare-event thresholds must be unique and increasing")
    methods = study.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("study.methods must be a non-empty list")
    if "trajectory_waste_free" in methods:
        raise ValueError("the submission study deliberately excludes Waste-Free SMC")
    supported = {"fk_bootstrap", "trajectory_resample_move"}
    if not set(methods).issubset(supported):
        raise ValueError(f"unsupported submission-study methods: {methods}")
    pilot = study.get("pilot_weight_estimation")
    if not isinstance(pilot, dict):
        raise ValueError("pilot_weight_estimation must be a mapping")
    if pilot.get("policy") != "independent_pilot_inverse_variance":
        raise ValueError(
            "pilot_weight_estimation.policy must be "
            "independent_pilot_inverse_variance"
        )
    if pilot.get("include_pilot_runs_in_final_comparison") is not False:
        raise ValueError("pilot runs must be excluded from the final comparison")
    if pilot.get("freeze_before_test_report") is not True:
        raise ValueError("pilot weights must be frozen before the test report")
    shrinkage = float(pilot.get("variance_shrinkage", -1.0))
    floor_fraction = float(pilot.get("variance_floor_fraction", -1.0))
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("variance_shrinkage must be in [0, 1]")
    if not 0.0 <= floor_fraction <= 1.0:
        raise ValueError("variance_floor_fraction must be in [0, 1]")
    return study


def _suite_path(study: dict[str, Any], study_path: Path, key: str) -> Path:
    suites = study.get("suites")
    if not isinstance(suites, dict) or not isinstance(suites.get(key), str):
        raise ValueError(f"study.suites.{key} must name a suite config")
    return _config_path(study_path.parent, suites[key]).resolve()


def study_manifest(study: dict[str, Any], study_path: Path) -> dict[str, Any]:
    pilot_path = _suite_path(study, study_path, "pilot")
    test_path = _suite_path(study, study_path, "test")
    pilot_suite = load_suite(pilot_path)
    test_suite = load_suite(test_path)
    pilot_manifest = comparison_manifest(pilot_suite, pilot_path)
    test_manifest = comparison_manifest(test_suite, test_path)
    if pilot_manifest["decoder_id"] != test_manifest["decoder_id"]:
        raise ValueError("pilot and test suites use different decoders")
    if pilot_manifest["evaluator_id"] != test_manifest["evaluator_id"]:
        raise ValueError("pilot and test suites use different evaluators")
    expected_decoder = study.get("expected_decoder_id")
    if expected_decoder is not None and expected_decoder != pilot_manifest["decoder_id"]:
        raise ValueError("submission study has the wrong expected_decoder_id")
    expected_evaluator = study.get("expected_evaluator_id")
    if (
        expected_evaluator is not None
        and expected_evaluator != pilot_manifest["evaluator_id"]
    ):
        raise ValueError("submission study has the wrong expected_evaluator_id")
    return {
        "kind": STUDY_KIND,
        "schema_version": 1,
        "study_name": str(study.get("study_name", "")),
        "decoder_id": pilot_manifest["decoder_id"],
        "evaluator_id": pilot_manifest["evaluator_id"],
        "methods": list(study["methods"]),
        "rare_events": {
            "score_field": "event_toxic_probability",
            "definition": "1{event_toxic_probability >= threshold}",
            "toxic_probability_thresholds": [
                float(value)
                for value in study["rare_events"]["toxic_probability_thresholds"]
            ],
        },
        "pilot_weight_estimation": json.loads(
            canonical_json(study["pilot_weight_estimation"])
        ),
        "pilot_suite_path": str(pilot_path),
        "pilot_suite_id": fingerprint(pilot_manifest),
        "test_suite_path": str(test_path),
        "test_suite_id": fingerprint(test_manifest),
    }


def study_run_directory(
    study: dict[str, Any], study_path: Path, output_root: Path
) -> Path:
    manifest = study_manifest(study, study_path)
    return output_root / "mdlm_owt" / "submission_probability" / fingerprint(manifest)


def _write_text_compatible(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json_compatible(path: Path, payload: Any) -> None:
    _write_text_compatible(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _direct_configs(
    suite: dict[str, Any], suite_path: Path
) -> tuple[dict[str, Any], dict[str, Any], int]:
    direct = suite["direct"]
    generation = load_yaml(_config_path(suite_path.parent, direct["generation_config"]))
    evaluation = load_yaml(_config_path(suite_path.parent, direct["evaluation_config"]))
    if decoder_id(generation) != decoder_id(evaluation):
        raise ValueError("direct generation and evaluation decoder_id mismatch")
    return generation, evaluation, int(direct["target_samples"])


def _ordered_direct_scores(
    suite: dict[str, Any], suite_path: Path, output_root: Path
) -> list[dict[str, Any]]:
    generation, evaluation, target = _direct_configs(suite, suite_path)
    run_dir = direct_run_directory(generation, output_root)
    inventory = scan_direct_generations(run_dir, decoder_id(generation))
    by_index = {
        int(row["sample_index"]): row
        for row in inventory.rows
        if int(row["sample_index"]) < target
    }
    if set(by_index) != set(range(target)):
        raise ValueError(f"direct reference lacks indices 0..{target - 1}")
    eval_dir = run_dir / "evaluations" / evaluator_id(evaluation)
    by_sample_id: dict[str, dict[str, Any]] = {}
    for score_path in sorted(eval_dir.glob("scores_*.jsonl")):
        for row in read_jsonl(score_path):
            sample_id = str(row["sample_id"])
            if sample_id in by_sample_id:
                raise ValueError(f"duplicate direct score for {sample_id}")
            if row.get("decoder_id") != decoder_id(generation):
                raise ValueError(f"direct score has the wrong decoder_id: {sample_id}")
            if row.get("evaluator_id") != evaluator_id(evaluation):
                raise ValueError(f"direct score has the wrong evaluator_id: {sample_id}")
            score = float(row["event_toxic_probability"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise FloatingPointError(
                    f"direct toxic probability is not finite in [0, 1]: {sample_id}"
                )
            by_sample_id[sample_id] = row
    ordered = []
    for index in range(target):
        sample_id = str(by_index[index]["sample_id"])
        if sample_id not in by_sample_id:
            raise ValueError(f"direct sample {index} has not been scored")
        ordered.append(by_sample_id[sample_id])
    return ordered


def direct_threshold_summaries(
    scores: Sequence[dict[str, Any]], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    if not scores:
        raise ValueError("no direct scores")
    output = []
    total = len(scores)
    for threshold in thresholds:
        count = sum(
            float(row["event_toxic_probability"]) >= threshold for row in scores
        )
        probability = count / total
        low, high = wilson_interval(count, total)
        output.append(
            {
                "threshold": threshold,
                "samples": total,
                "event_discoveries": count,
                "empirical_probability": probability,
                "binomial_standard_error": math.sqrt(
                    probability * (1.0 - probability) / total
                ),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return output


def _selected_entries(
    suite: dict[str, Any],
    suite_path: Path,
    *,
    methods: set[str],
    budgets: set[int] | None,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in resolved_comparison_entries(suite, suite_path)
        if entry["method_key"] in methods
        and (budgets is None or int(entry["particle_budget"]) in budgets)
    ]
    if not entries:
        raise ValueError("no suite entries match the submission study")
    found_methods = {str(entry["method_key"]) for entry in entries}
    if found_methods != methods:
        raise ValueError(
            f"suite is missing requested methods: {sorted(methods - found_methods)}"
        )
    return entries


def planned_component_seeds(entries: Iterable[dict[str, Any]]) -> set[int]:
    seeds: set[int] = set()
    for entry in entries:
        smc = entry["config"]["smc"]
        runs = int(smc["independent_runs"])
        component_count = len(smc["tilted_lambdas"])
        for run_index in range(runs):
            for component_index in range(component_count):
                seed = int(smc["base_seed"]) + 1_000_000 * run_index + component_index
                if seed in seeds:
                    raise ValueError("suite reuses an SMC component seed")
                seeds.add(seed)
    return seeds


def component_threshold_estimate(
    particles: Sequence[dict[str, Any]], threshold: float
) -> tuple[float, int, float]:
    if not particles:
        raise ValueError("component has no particles")
    log_weights = [float(row["reverse_corrected_log_weight"]) for row in particles]
    if any(not math.isfinite(value) for value in log_weights):
        raise FloatingPointError("reverse-corrected log weights contain NaN or Inf")
    maximum = max(log_weights)
    raw_weights = [math.exp(value - maximum) for value in log_weights]
    total = sum(raw_weights)
    weights = [value / total for value in raw_weights]
    scores = [float(row["event_toxic_probability"]) for row in particles]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise FloatingPointError(
            "particle toxic probabilities must be finite values in [0, 1]"
        )
    events = [value >= threshold for value in scores]
    estimate = sum(weight * event for weight, event in zip(weights, events))
    ess = 1.0 / sum(weight * weight for weight in weights)
    return estimate, sum(events), ess


def suite_component_estimates(
    suite: dict[str, Any],
    suite_path: Path,
    *,
    output_root: Path,
    methods: set[str],
    budgets: set[int] | None,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in _selected_entries(
        suite, suite_path, methods=methods, budgets=budgets
    ):
        run_dir = smc_run_directory(entry["config"], output_root)
        if not (run_dir / "summary.json").exists():
            raise FileNotFoundError(f"missing completed SMC summary: {run_dir}")
        smc = entry["config"]["smc"]
        lambdas = [float(value) for value in smc["tilted_lambdas"]]
        for run_index in range(int(smc["independent_runs"])):
            for component_index, lambda_value in enumerate(lambdas):
                path = run_dir / "components" / (
                    f"run_{run_index:03d}_tilted_{component_index:03d}.json"
                )
                if not path.exists():
                    raise FileNotFoundError(f"missing SMC component: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("decoder_id") != entry["smc_manifest"]["decoder_id"]:
                    raise ValueError(f"component decoder_id mismatch: {path}")
                if payload.get("evaluator_id") != entry["smc_manifest"]["evaluator_id"]:
                    raise ValueError(f"component evaluator_id mismatch: {path}")
                summary = payload["summary"]
                if int(summary["run_index"]) != run_index or not math.isclose(
                    float(summary["lambda"]), lambda_value, abs_tol=1e-12
                ):
                    raise ValueError(f"component metadata mismatch: {path}")
                for threshold in thresholds:
                    estimate, discoveries, reverse_ess = component_threshold_estimate(
                        payload["particles"], threshold
                    )
                    rows.append(
                        {
                            "method_key": entry["method_key"],
                            "method_label": entry["method_label"],
                            "particle_budget": int(entry["particle_budget"]),
                            "run_index": run_index,
                            "lambda": lambda_value,
                            "seed": int(summary["seed"]),
                            "threshold": threshold,
                            "estimate": estimate,
                            "tilted_event_discoveries": discoveries,
                            "reverse_ess": reverse_ess,
                            "reverse_ess_over_n": reverse_ess / len(payload["particles"]),
                            "runtime_seconds": float(summary["runtime_seconds"]),
                            "decoder_particle_steps": int(
                                summary["total_decoder_particle_steps"]
                            ),
                            "classifier_texts_scored": int(
                                summary["classifier_texts_scored"]
                            ),
                            "distinct_root_ancestor_fraction": float(
                                summary["distinct_root_ancestor_fraction"]
                            ),
                            "peak_cuda_memory_allocated_bytes": int(
                                summary["peak_cuda_memory_allocated_bytes"]
                            ),
                        }
                    )
    return rows


def inverse_variance_weights(
    estimates_by_component: Sequence[Sequence[float]],
    *,
    shrinkage: float,
    floor_fraction: float,
) -> dict[str, Any]:
    if len(estimates_by_component) < 2:
        raise ValueError("at least two tilted components are required")
    if any(len(values) < 2 for values in estimates_by_component):
        raise ValueError("at least two pilot runs are required per component")
    variances = [statistics.variance(values) for values in estimates_by_component]
    mean_variance = sum(variances) / len(variances)
    if mean_variance <= 0.0:
        weights = [1.0 / len(variances)] * len(variances)
        adjusted = [0.0] * len(variances)
    else:
        floor = max(1e-18, floor_fraction * mean_variance)
        adjusted = [
            max((1.0 - shrinkage) * value + shrinkage * mean_variance, floor)
            for value in variances
        ]
        inverse = [1.0 / value for value in adjusted]
        normalizer = sum(inverse)
        weights = [value / normalizer for value in inverse]
    return {
        "raw_sample_variances": variances,
        "adjusted_variances": adjusted,
        "weights": weights,
        "sum_weights": sum(weights),
    }


def fit_pilot_weights(
    rows: Sequence[dict[str, Any]],
    *,
    shrinkage: float,
    floor_fraction: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method_key"]), int(row["particle_budget"]), float(row["threshold"]))
        ].append(row)
    output = []
    for (method, budget, threshold), selected in sorted(grouped.items()):
        lambdas = sorted({float(row["lambda"]) for row in selected})
        seen_pairs: set[tuple[float, int]] = set()
        run_sets: list[set[int]] = []
        for lambda_value in lambdas:
            run_indices: set[int] = set()
            for row in selected:
                if not math.isclose(
                    float(row["lambda"]), lambda_value, abs_tol=1e-12
                ):
                    continue
                pair = (lambda_value, int(row["run_index"]))
                if pair in seen_pairs:
                    raise ValueError("duplicate pilot component/run estimate")
                seen_pairs.add(pair)
                run_indices.add(int(row["run_index"]))
            run_sets.append(run_indices)
        if any(run_indices != run_sets[0] for run_indices in run_sets[1:]):
            raise ValueError("pilot lambda components do not share the same run set")
        estimates_by_lambda = [
            [
                float(row["estimate"])
                for row in selected
                if math.isclose(float(row["lambda"]), value, abs_tol=1e-12)
            ]
            for value in lambdas
        ]
        fitted = inverse_variance_weights(
            estimates_by_lambda,
            shrinkage=shrinkage,
            floor_fraction=floor_fraction,
        )
        output.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "threshold": threshold,
                "lambdas": lambdas,
                "pilot_runs_per_component": [
                    len(values) for values in estimates_by_lambda
                ],
                "uniform_weights": [1.0 / len(lambdas)] * len(lambdas),
                "optimized_weights": fitted["weights"],
                "raw_sample_variances": fitted["raw_sample_variances"],
                "adjusted_variances": fitted["adjusted_variances"],
            }
        )
    return output


def _pilot_budgets(study: dict[str, Any]) -> set[int]:
    values = study["pilot_weight_estimation"].get("particle_budgets")
    if not isinstance(values, list) or not values or any(
        not isinstance(value, int) or value <= 0 for value in values
    ):
        raise ValueError("pilot particle_budgets must be positive integers")
    return set(values)


def freeze_pilot(
    study: dict[str, Any], *, study_path: Path, output_root: Path
) -> Path:
    manifest = study_manifest(study, study_path)
    run_dir = study_run_directory(study, study_path, output_root)
    pilot_path = Path(manifest["pilot_suite_path"])
    test_path = Path(manifest["test_suite_path"])
    pilot_suite = load_suite(pilot_path)
    test_suite = load_suite(test_path)
    methods = set(study["methods"])
    budgets = _pilot_budgets(study)
    pilot_entries = _selected_entries(
        pilot_suite, pilot_path, methods=methods, budgets=budgets
    )
    test_entries = _selected_entries(
        test_suite, test_path, methods=methods, budgets=budgets
    )
    overlap = planned_component_seeds(pilot_entries).intersection(
        planned_component_seeds(test_entries)
    )
    if overlap:
        raise ValueError(f"pilot and test suites reuse component seeds: {sorted(overlap)}")
    thresholds = manifest["rare_events"]["toxic_probability_thresholds"]
    direct_scores = _ordered_direct_scores(pilot_suite, pilot_path, output_root)
    direct_rows = direct_threshold_summaries(direct_scores, thresholds)
    component_rows = suite_component_estimates(
        pilot_suite,
        pilot_path,
        output_root=output_root,
        methods=methods,
        budgets=budgets,
        thresholds=thresholds,
    )
    pilot_cfg = study["pilot_weight_estimation"]
    fitted = fit_pilot_weights(
        component_rows,
        shrinkage=float(pilot_cfg["variance_shrinkage"]),
        floor_fraction=float(pilot_cfg["variance_floor_fraction"]),
    )
    payload = {
        "kind": "mdlm_submission_pilot_weights_v1",
        "study_id": fingerprint(manifest),
        "manifest": manifest,
        "statistical_role": (
            "frozen pilot only; these runs must not be included in the final "
            "uniform-versus-optimized evaluation"
        ),
        "seed_overlap_with_planned_test": 0,
        "direct_thresholds": direct_rows,
        "fitted_component_weights": fitted,
    }
    _write_json_compatible(run_dir / "manifest.json", {"study_id": fingerprint(manifest), **manifest})
    _write_json_compatible(run_dir / "pilot_weights.json", payload)
    _write_text_compatible(run_dir / "direct_thresholds.csv", _csv_text(direct_rows))
    _write_text_compatible(
        run_dir / "pilot_component_estimates.csv", _csv_text(component_rows)
    )
    weight_rows = []
    for fit in fitted:
        for index, lambda_value in enumerate(fit["lambdas"]):
            weight_rows.append(
                {
                    "method_key": fit["method_key"],
                    "particle_budget": fit["particle_budget"],
                    "threshold": fit["threshold"],
                    "lambda": lambda_value,
                    "uniform_weight": fit["uniform_weights"][index],
                    "optimized_weight": fit["optimized_weights"][index],
                    "raw_sample_variance": fit["raw_sample_variances"][index],
                    "adjusted_variance": fit["adjusted_variances"][index],
                }
            )
    _write_text_compatible(run_dir / "pilot_weights.csv", _csv_text(weight_rows))
    return run_dir


def verify_frozen_pilot(
    study: dict[str, Any], *, study_path: Path, output_root: Path
) -> Path:
    manifest = study_manifest(study, study_path)
    path = study_run_directory(study, study_path, output_root) / "pilot_weights.json"
    if not path.exists():
        raise FileNotFoundError(f"frozen pilot weights do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != fingerprint(manifest):
        raise ValueError("frozen pilot weights have the wrong study_id")
    return path


def _metric_summary(values: Sequence[float], reference: float) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("at least two independent test runs are required")
    errors = [value - reference for value in values]
    return {
        "mean_estimate": statistics.mean(values),
        "estimate_standard_deviation": statistics.stdev(values),
        "bias_vs_direct_reference": statistics.mean(errors),
        "mae_vs_direct_reference": statistics.mean(abs(value) for value in errors),
        "rmse_vs_direct_reference": math.sqrt(
            statistics.mean(value * value for value in errors)
        ),
    }


def _threshold_tag(threshold: float) -> str:
    return format(threshold, ".12g").replace(".", "p")


def build_test_report(
    study: dict[str, Any], *, study_path: Path, output_root: Path
) -> Path:
    manifest = study_manifest(study, study_path)
    run_dir = study_run_directory(study, study_path, output_root)
    frozen_path = verify_frozen_pilot(
        study, study_path=study_path, output_root=output_root
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    test_path = Path(manifest["test_suite_path"])
    test_suite = load_suite(test_path)
    methods = set(study["methods"])
    budgets = _pilot_budgets(study)
    thresholds = manifest["rare_events"]["toxic_probability_thresholds"]
    direct_scores = _ordered_direct_scores(test_suite, test_path, output_root)
    direct_rows = direct_threshold_summaries(direct_scores, thresholds)
    direct_by_threshold = {
        float(row["threshold"]): float(row["empirical_probability"])
        for row in direct_rows
    }
    component_rows = suite_component_estimates(
        test_suite,
        test_path,
        output_root=output_root,
        methods=methods,
        budgets=budgets,
        thresholds=thresholds,
    )
    weights = {
        (
            str(row["method_key"]),
            int(row["particle_budget"]),
            float(row["threshold"]),
        ): row
        for row in frozen["fitted_component_weights"]
    }
    components_by_run: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        components_by_run[
            (
                str(row["method_key"]),
                int(row["particle_budget"]),
                int(row["run_index"]),
                float(row["threshold"]),
            )
        ].append(row)
    run_rows = []
    for (method, budget, run_index, threshold), selected in sorted(
        components_by_run.items()
    ):
        fit = weights[(method, budget, threshold)]
        lambdas = [float(value) for value in fit["lambdas"]]
        by_lambda = {float(row["lambda"]): float(row["estimate"]) for row in selected}
        if set(by_lambda) != set(lambdas):
            raise ValueError("test run is missing a tilted component")
        estimates = [by_lambda[value] for value in lambdas]
        uniform = sum(estimates) / len(estimates)
        optimized = sum(
            value * weight
            for value, weight in zip(estimates, fit["optimized_weights"])
        )
        reference = direct_by_threshold[threshold]
        run_rows.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "run_index": run_index,
                "threshold": threshold,
                "direct_reference": reference,
                "uniform_estimate": uniform,
                "optimized_estimate": optimized,
                "uniform_error": uniform - reference,
                "optimized_error": optimized - reference,
                "runtime_seconds": sum(
                    float(row["runtime_seconds"]) for row in selected
                ),
                "decoder_particle_steps": sum(
                    int(row["decoder_particle_steps"]) for row in selected
                ),
                "classifier_texts_scored": sum(
                    int(row["classifier_texts_scored"]) for row in selected
                ),
                "mean_reverse_ess_over_n": statistics.fmean(
                    float(row["reverse_ess_over_n"]) for row in selected
                ),
                "mean_distinct_root_ancestor_fraction": statistics.fmean(
                    float(row["distinct_root_ancestor_fraction"])
                    for row in selected
                ),
                "peak_cuda_memory_allocated_bytes": max(
                    int(row["peak_cuda_memory_allocated_bytes"])
                    for row in selected
                ),
            }
        )
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (str(row["method_key"]), int(row["particle_budget"]), float(row["threshold"]))
        ].append(row)
    summary_rows = []
    for (method, budget, threshold), selected in sorted(grouped.items()):
        reference = direct_by_threshold[threshold]
        uniform_metrics = _metric_summary(
            [float(row["uniform_estimate"]) for row in selected], reference
        )
        optimized_metrics = _metric_summary(
            [float(row["optimized_estimate"]) for row in selected], reference
        )
        uniform_rmse = uniform_metrics["rmse_vs_direct_reference"]
        optimized_rmse = optimized_metrics["rmse_vs_direct_reference"]
        paired_squared_error_improvements = [
            float(row["uniform_error"]) ** 2
            - float(row["optimized_error"]) ** 2
            for row in selected
        ]
        summary_rows.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "threshold": threshold,
                "independent_test_runs": len(selected),
                "direct_reference": reference,
                "uniform_mean": uniform_metrics["mean_estimate"],
                "uniform_sd": uniform_metrics["estimate_standard_deviation"],
                "uniform_rmse": uniform_rmse,
                "optimized_mean": optimized_metrics["mean_estimate"],
                "optimized_sd": optimized_metrics["estimate_standard_deviation"],
                "optimized_rmse": optimized_rmse,
                "relative_rmse_improvement": (
                    1.0 - optimized_rmse / uniform_rmse
                    if uniform_rmse > 0.0
                    else 0.0
                ),
                "uniform_variance": statistics.variance(
                    float(row["uniform_estimate"]) for row in selected
                ),
                "optimized_variance": statistics.variance(
                    float(row["optimized_estimate"]) for row in selected
                ),
                "optimized_lower_squared_error_fraction": statistics.fmean(
                    value > 0.0 for value in paired_squared_error_improvements
                ),
                "mean_paired_squared_error_improvement": statistics.fmean(
                    paired_squared_error_improvements
                ),
                "median_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in selected
                ),
                "median_decoder_particle_steps": statistics.median(
                    int(row["decoder_particle_steps"]) for row in selected
                ),
                "median_classifier_texts_scored": statistics.median(
                    int(row["classifier_texts_scored"]) for row in selected
                ),
                "median_reverse_ess_over_n": statistics.median(
                    float(row["mean_reverse_ess_over_n"]) for row in selected
                ),
                "median_distinct_root_ancestor_fraction": statistics.median(
                    float(row["mean_distinct_root_ancestor_fraction"])
                    for row in selected
                ),
                "peak_cuda_memory_allocated_bytes": max(
                    int(row["peak_cuda_memory_allocated_bytes"])
                    for row in selected
                ),
            }
        )
    report = {
        "kind": "mdlm_submission_probability_test_report_v1",
        "study_id": fingerprint(manifest),
        "pilot_weights_path": str(frozen_path),
        "pilot_and_test_populations_independent": True,
        "direct_thresholds": direct_rows,
        "summary": summary_rows,
        "finite_sample_note": (
            "Every component is a separately normalized reverse-SNIS ratio and "
            "is generally biased at finite N. Optimized weights minimize a "
            "pilot estimate of variance, not finite-sample MSE."
        ),
    }
    report_dir = run_dir / "test_report"
    _write_json_compatible(report_dir / "report.json", report)
    _write_text_compatible(report_dir / "run_estimates.csv", _csv_text(run_rows))
    _write_text_compatible(report_dir / "summary.csv", _csv_text(summary_rows))
    _write_text_compatible(
        report_dir / "component_estimates.csv", _csv_text(component_rows)
    )
    direct_intervals = {
        float(row["threshold"]): (
            float(row["wilson_95_low"]),
            float(row["wilson_95_high"]),
        )
        for row in direct_rows
    }
    method_labels = {
        "fk_bootstrap": "FK",
        "trajectory_resample_move": "Trajectory MH",
    }
    for threshold in thresholds:
        selected = [
            row for row in run_rows if float(row["threshold"]) == threshold
        ]
        estimate_groups = []
        error_groups = []
        for method in study["methods"]:
            method_rows = [
                row for row in selected if row["method_key"] == method
            ]
            label = method_labels.get(method, method)
            estimate_groups.extend(
                [
                    {
                        "label": f"{label} uniform",
                        "method_key": method,
                        "values": [
                            float(row["uniform_estimate"]) for row in method_rows
                        ],
                    },
                    {
                        "label": f"{label} pilot weights",
                        "method_key": method,
                        "values": [
                            float(row["optimized_estimate"]) for row in method_rows
                        ],
                    },
                ]
            )
            error_groups.extend(
                [
                    {
                        "label": f"{label} uniform",
                        "method_key": method,
                        "values": [
                            abs(float(row["uniform_error"])) for row in method_rows
                        ],
                    },
                    {
                        "label": f"{label} pilot weights",
                        "method_key": method,
                        "values": [
                            abs(float(row["optimized_error"])) for row in method_rows
                        ],
                    },
                ]
            )
        tag = _threshold_tag(threshold)
        _write_text_compatible(
            report_dir / f"estimate_threshold_{tag}_boxplot.svg",
            _boxplot_svg(
                title=f"Probability estimates: toxic score >= {threshold:g}",
                y_label="estimated probability",
                groups=estimate_groups,
                reference=direct_by_threshold[threshold],
                band=direct_intervals[threshold],
            ),
        )
        _write_text_compatible(
            report_dir / f"absolute_error_threshold_{tag}_boxplot.svg",
            _boxplot_svg(
                title=f"Absolute error: toxic score >= {threshold:g}",
                y_label="absolute error vs IID reference",
                groups=error_groups,
            ),
        )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "verify-pilot", "report"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    study_path = args.study.resolve()
    study = load_study(study_path)
    if args.command == "pilot":
        destination = freeze_pilot(
            study, study_path=study_path, output_root=args.output_root
        )
    elif args.command == "verify-pilot":
        destination = verify_frozen_pilot(
            study, study_path=study_path, output_root=args.output_root
        )
    else:
        destination = build_test_report(
            study, study_path=study_path, output_root=args.output_root
        )
    print(destination)


if __name__ == "__main__":
    main()
