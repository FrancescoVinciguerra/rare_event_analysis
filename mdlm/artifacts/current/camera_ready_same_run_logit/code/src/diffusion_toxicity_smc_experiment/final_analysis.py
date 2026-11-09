"""Build the final cost-matched rare-event analysis without model calls."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import yaml

from .configuration import decoder_id, evaluator_id, fingerprint, load_yaml
from .generate_direct import direct_run_directory
from .run_smc import smc_run_directory
from .serious_comparison import (
    _boxplot_svg,
    comparison_manifest,
    load_suite,
    resolved_comparison_entries,
)
from .storage import scan_direct_generations
from .submission_probability import (
    _ordered_direct_scores,
    direct_threshold_summaries,
    load_study,
    suite_component_estimates,
)


def _relative(parent: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (parent / candidate).resolve()


def load_final_study(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        study = yaml.safe_load(handle)
    if not isinstance(study, dict) or int(study.get("schema_version", 0)) != 1:
        raise ValueError("final analysis must be a schema-version 1 mapping")
    thresholds = study.get("rare_event_thresholds")
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or any(not isinstance(value, (int, float)) for value in thresholds)
    ):
        raise ValueError("rare_event_thresholds must be a non-empty numeric list")
    numeric = [float(value) for value in thresholds]
    if numeric != sorted(set(numeric)) or any(not 0.0 < value < 1.0 for value in numeric):
        raise ValueError("rare_event_thresholds must be unique, increasing and in (0,1)")
    if study.get("component_weight_policy") != "uniform":
        raise ValueError("the final analysis requires uniform component weights")
    bootstrap = study.get("bootstrap", {})
    if int(bootstrap.get("replicates", 0)) < 1000:
        raise ValueError("bootstrap.replicates must be at least 1000")
    return study


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_text_compatible(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json_compatible(path: Path, value: Any) -> None:
    _write_text_compatible(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    if not math.isfinite(maximum):
        raise FloatingPointError("logsumexp inputs have no finite maximum")
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalizer_consistency(summary: dict[str, Any], particles: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Check log(Z_hat) + log E_pi_hat[exp(-lambda phi)] ~= 0.

    The empirical expectation uses the terminal normalized SMC weights. This
    remains valid when final resampling is disabled and terminal weights are
    non-uniform.
    """

    lambda_value = float(summary["lambda"])
    final_log_weights = [float(row["final_log_weight"]) for row in particles]
    rewards = [float(row["reward"]) for row in particles]
    if any(not math.isfinite(value) for value in final_log_weights + rewards):
        raise FloatingPointError("normalizer diagnostic received NaN or Inf")
    log_weight_sum = _logsumexp(final_log_weights)
    normalized = [value - log_weight_sum for value in final_log_weights]
    log_reverse_moment = _logsumexp(
        [weight - lambda_value * reward for weight, reward in zip(normalized, rewards)]
    )
    log_z = float(summary["log_normalizer_estimate"])
    residual = log_z + log_reverse_moment
    return {
        "log_normalizer_estimate": log_z,
        "log_reverse_moment": log_reverse_moment,
        "log_identity_residual": residual,
        "multiplicative_identity": math.exp(residual),
    }


def _component_payload_rows(
    suite: dict[str, Any],
    suite_path: Path,
    *,
    output_root: Path,
    selected: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in resolved_comparison_entries(suite, suite_path):
        key = (str(entry["method_key"]), int(entry["particle_budget"]))
        if key not in selected:
            continue
        config = entry["config"]
        run_dir = smc_run_directory(config, output_root)
        lambdas = [float(value) for value in config["smc"]["tilted_lambdas"]]
        for run_index in range(int(config["smc"]["independent_runs"])):
            for component_index, lambda_value in enumerate(lambdas):
                path = run_dir / "components" / f"run_{run_index:03d}_tilted_{component_index:03d}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing component artifact: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = payload["summary"]
                diagnostic = normalizer_consistency(summary, payload["particles"])
                rows.append(
                    {
                        "method_key": key[0],
                        "particle_budget": key[1],
                        "run_index": run_index,
                        "lambda": lambda_value,
                        "seed": int(summary["seed"]),
                        **diagnostic,
                        "reverse_ess_over_n": float(summary["reverse_weight_ess_over_n"]),
                        "distinct_root_fraction": float(summary["distinct_root_ancestor_fraction"]),
                        "root_entropy_normalized": float(summary["normalized_root_ancestor_entropy"]),
                        "largest_root_family_fraction": float(summary["largest_root_family_fraction"]),
                        "resampling_count": int(summary["resampling_count"]),
                        "mcmc_acceptance_rate": summary.get("mcmc_acceptance_rate"),
                    }
                )
    return rows


def _with_run_offset(
    rows: Sequence[dict[str, Any]], *, offset: int, cohort: str
) -> list[dict[str, Any]]:
    """Give independent cohorts unique run identifiers without changing samples."""

    if offset < 0:
        raise ValueError("run offset must be non-negative")
    output = []
    for row in rows:
        copied = dict(row)
        copied["source_run_index"] = int(row["run_index"])
        copied["run_index"] = offset + int(row["run_index"])
        copied["cohort"] = cohort
        output.append(copied)
    return output


def _run_rows(component_rows: Sequence[dict[str, Any]], direct: dict[float, float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        grouped[(str(row["method_key"]), int(row["particle_budget"]), int(row["run_index"]), float(row["threshold"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, budget, run_index, threshold), rows in sorted(grouped.items()):
        cohorts = {str(row["cohort"]) for row in rows}
        source_indices = {int(row["source_run_index"]) for row in rows}
        if len(cohorts) != 1 or len(source_indices) != 1:
            raise ValueError("a pooled run combines multiple independent cohorts")
        lambdas = [float(row["lambda"]) for row in rows]
        if lambdas != sorted(set(lambdas)):
            raise ValueError("a run contains duplicate or unordered lambda components")
        estimates = [float(row["estimate"]) for row in rows]
        estimate = statistics.fmean(estimates)
        reference = direct[threshold]
        output.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "run_index": run_index,
                "source_run_index": next(iter(source_indices)),
                "cohort": next(iter(cohorts)),
                "threshold": threshold,
                "component_weights": json.dumps([1.0 / len(rows)] * len(rows)),
                "estimate": estimate,
                "direct_reference": reference,
                "error": estimate - reference,
                "absolute_error": abs(estimate - reference),
                "squared_error": (estimate - reference) ** 2,
                "zero_estimate": estimate == 0.0,
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
                "decoder_particle_steps": sum(int(row["decoder_particle_steps"]) for row in rows),
                "classifier_texts_scored": sum(int(row["classifier_texts_scored"]) for row in rows),
                "mean_reverse_ess_over_n": statistics.fmean(float(row["reverse_ess_over_n"]) for row in rows),
                "mean_distinct_root_fraction": statistics.fmean(float(row["distinct_root_ancestor_fraction"]) for row in rows),
                "peak_cuda_memory_allocated_bytes": max(int(row["peak_cuda_memory_allocated_bytes"]) for row in rows),
            }
        )
    return output


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_key"]), int(row["particle_budget"]), float(row["threshold"]))].append(row)
    output = []
    for (method, budget, threshold), selected in sorted(grouped.items()):
        errors = [float(row["error"]) for row in selected]
        estimates = [float(row["estimate"]) for row in selected]
        median_runtime = statistics.median(float(row["runtime_seconds"]) for row in selected)
        mse = statistics.fmean(value * value for value in errors)
        output.append(
            {
                "method_key": method,
                "particle_budget": budget,
                "threshold": threshold,
                "independent_runs": len(selected),
                "direct_reference": float(selected[0]["direct_reference"]),
                "mean_estimate": statistics.fmean(estimates),
                "estimate_sd": statistics.stdev(estimates),
                "bias": statistics.fmean(errors),
                "rmse": math.sqrt(mse),
                "zero_estimate_fraction": statistics.fmean(bool(row["zero_estimate"]) for row in selected),
                "median_runtime_seconds": median_runtime,
                "median_decoder_particle_steps": statistics.median(int(row["decoder_particle_steps"]) for row in selected),
                "median_classifier_texts_scored": statistics.median(int(row["classifier_texts_scored"]) for row in selected),
                "mse_times_runtime": mse * median_runtime,
                "median_reverse_ess_over_n": statistics.median(float(row["mean_reverse_ess_over_n"]) for row in selected),
                "median_distinct_root_fraction": statistics.median(float(row["mean_distinct_root_fraction"]) for row in selected),
                "peak_cuda_memory_allocated_bytes": max(int(row["peak_cuda_memory_allocated_bytes"]) for row in selected),
            }
        )
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_ratio(
    numerator_errors: Sequence[float],
    denominator_errors: Sequence[float],
    *,
    numerator_runtime: float,
    denominator_runtime: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Independent run-level bootstrap for RMSE and time-normalized MSE ratios."""

    if len(numerator_errors) < 2 or len(denominator_errors) < 2:
        raise ValueError("bootstrap requires at least two independent runs per method")
    rng = random.Random(seed)
    rmse_ratios = []
    cost_ratios = []
    for _ in range(replicates):
        numerator_mse = statistics.fmean(
            numerator_errors[rng.randrange(len(numerator_errors))] ** 2
            for _ in numerator_errors
        )
        denominator_mse = statistics.fmean(
            denominator_errors[rng.randrange(len(denominator_errors))] ** 2
            for _ in denominator_errors
        )
        if denominator_mse == 0.0:
            continue
        rmse_ratios.append(math.sqrt(numerator_mse / denominator_mse))
        cost_ratios.append(
            numerator_mse * numerator_runtime
            / (denominator_mse * denominator_runtime)
        )
    if not rmse_ratios:
        raise ValueError("bootstrap denominator MSE is always zero")
    return {
        "rmse_ratio": math.sqrt(
            statistics.fmean(value * value for value in numerator_errors)
            / statistics.fmean(value * value for value in denominator_errors)
        ),
        "rmse_ratio_ci_low": _percentile(rmse_ratios, 0.025),
        "rmse_ratio_ci_high": _percentile(rmse_ratios, 0.975),
        "probability_rmse_ratio_below_one": statistics.fmean(value < 1.0 for value in rmse_ratios),
        "mse_times_runtime_ratio": (
            statistics.fmean(value * value for value in numerator_errors) * numerator_runtime
            / (statistics.fmean(value * value for value in denominator_errors) * denominator_runtime)
        ),
        "mse_times_runtime_ratio_ci_low": _percentile(cost_ratios, 0.025),
        "mse_times_runtime_ratio_ci_high": _percentile(cost_ratios, 0.975),
        "probability_cost_ratio_below_one": statistics.fmean(value < 1.0 for value in cost_ratios),
    }


def _comparison_rows(
    rows: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row["method_key"]), int(row["particle_budget"]), float(row["threshold"]))].append(row)
    comparisons = [
        ("trajectory_n512_vs_fk_n512", ("trajectory_resample_move", 512), ("fk_bootstrap", 512)),
        ("trajectory_n192_vs_fk_n512_runtime_matched", ("trajectory_resample_move", 192), ("fk_bootstrap", 512)),
    ]
    output = []
    thresholds = sorted({float(row["threshold"]) for row in rows})
    for comparison_index, (name, numerator, denominator) in enumerate(comparisons):
        for threshold_index, threshold in enumerate(thresholds):
            num = by_group[(numerator[0], numerator[1], threshold)]
            den = by_group[(denominator[0], denominator[1], threshold)]
            num_runtime = statistics.median(float(row["runtime_seconds"]) for row in num)
            den_runtime = statistics.median(float(row["runtime_seconds"]) for row in den)
            result = bootstrap_ratio(
                [float(row["error"]) for row in num],
                [float(row["error"]) for row in den],
                numerator_runtime=num_runtime,
                denominator_runtime=den_runtime,
                replicates=replicates,
                seed=seed + 100 * comparison_index + threshold_index,
            )
            output.append(
                {
                    "comparison": name,
                    "threshold": threshold,
                    "numerator_runs": len(num),
                    "denominator_runs": len(den),
                    "numerator_median_runtime_seconds": num_runtime,
                    "denominator_median_runtime_seconds": den_runtime,
                    "runtime_ratio": num_runtime / den_runtime,
                    **result,
                }
            )
    return output


def _audit_rows(
    direct_scores: Sequence[dict[str, Any]],
    *,
    score_bands: Sequence[Sequence[float]],
    examples_per_band: int,
) -> list[dict[str, Any]]:
    by_score = sorted(direct_scores, key=lambda row: (float(row["event_toxic_probability"]), str(row["sample_id"])))
    output = []
    for band_index, (low, high) in enumerate(score_bands):
        eligible = [row for row in by_score if float(low) <= float(row["event_toxic_probability"]) < float(high)]
        if len(eligible) <= examples_per_band:
            chosen = eligible
        else:
            chosen = [
                eligible[round(index * (len(eligible) - 1) / (examples_per_band - 1))]
                for index in range(examples_per_band)
            ]
        for row in chosen:
            output.append(
                {
                    "audit_id": f"band{band_index:02d}_{len(output):04d}",
                    "score_band_low": float(low),
                    "score_band_high_exclusive": float(high),
                    "classifier_score": float(row["event_toxic_probability"]),
                    "classifier_event_at_0p5": float(row["event_toxic_probability"]) >= 0.5,
                    "text": str(row["event_classifier_text"]),
                    "manual_toxic_0_or_1": "",
                    "manual_confidence_1_to_5": "",
                    "manual_notes": "",
                }
            )
    return output


def build_final_report(study: dict[str, Any], *, study_path: Path, output_root: Path) -> Path:
    base_study_path = _relative(study_path.parent, str(study["base_submission_study"]))
    matched_path = _relative(study_path.parent, str(study["matched_runtime_suite"]))
    confirmatory_path = _relative(study_path.parent, str(study["confirmatory_suite"]))
    base_study = load_study(base_study_path)
    base_test_path = _relative(base_study_path.parent, str(base_study["suites"]["test"]))
    base_suite = load_suite(base_test_path)
    matched_suite = load_suite(matched_path)
    confirmatory_suite = load_suite(confirmatory_path)
    base_manifest = comparison_manifest(base_suite, base_test_path)
    matched_manifest = comparison_manifest(matched_suite, matched_path)
    confirmatory_manifest = comparison_manifest(confirmatory_suite, confirmatory_path)
    confirmatory_grid = {
        (str(entry["method_key"]), int(entry["particle_budget"]))
        for entry in resolved_comparison_entries(confirmatory_suite, confirmatory_path)
    }
    expected_confirmatory_grid = {
        ("fk_bootstrap", 192),
        ("fk_bootstrap", 512),
        ("trajectory_resample_move", 192),
        ("trajectory_resample_move", 512),
    }
    if confirmatory_grid != expected_confirmatory_grid:
        raise ValueError("confirmatory suite must contain the frozen 2x2 method/budget grid")
    base_runs = int(base_suite["smc"]["independent_runs"])
    matched_runs = int(matched_suite["smc"]["independent_runs"])
    confirmatory_runs = int(confirmatory_suite["smc"]["independent_runs"])
    if (base_runs, matched_runs, confirmatory_runs) != (32, 32, 32):
        raise ValueError("final analysis expects three frozen 32-run cohorts")
    for key in ("decoder_id", "evaluator_id"):
        if (
            base_manifest[key] != matched_manifest[key]
            or base_manifest[key] != confirmatory_manifest[key]
            or base_manifest[key] != study[f"expected_{key}"]
        ):
            raise ValueError(f"{key} mismatch across final-analysis suites")
    for label, manifest in (
        ("runtime-matched", matched_manifest),
        ("confirmatory", confirmatory_manifest),
    ):
        completion_path = (
            output_root
            / "mdlm_owt"
            / "comparisons"
            / fingerprint(manifest)
            / "completion.json"
        )
        if not completion_path.exists():
            raise FileNotFoundError(f"{label} suite is incomplete: {completion_path}")

    thresholds = [float(value) for value in study["rare_event_thresholds"]]
    direct_scores = _ordered_direct_scores(base_suite, base_test_path, output_root)
    direct_rows = direct_threshold_summaries(direct_scores, thresholds)
    direct = {float(row["threshold"]): float(row["empirical_probability"]) for row in direct_rows}
    base_components = _with_run_offset(suite_component_estimates(
        base_suite,
        base_test_path,
        output_root=output_root,
        methods={"fk_bootstrap", "trajectory_resample_move"},
        budgets={512},
        thresholds=thresholds,
    ), offset=0, cohort="submission")
    matched_components = _with_run_offset(suite_component_estimates(
        matched_suite,
        matched_path,
        output_root=output_root,
        methods={"trajectory_resample_move"},
        budgets={192},
        thresholds=thresholds,
    ), offset=0, cohort="runtime_matched")
    confirmatory_components = suite_component_estimates(
        confirmatory_suite,
        confirmatory_path,
        output_root=output_root,
        methods={"fk_bootstrap", "trajectory_resample_move"},
        budgets={192, 512},
        thresholds=thresholds,
    )
    confirmatory_by_group: list[dict[str, Any]] = []
    for method, budget, offset in (
        ("fk_bootstrap", 192, 0),
        ("fk_bootstrap", 512, base_runs),
        ("trajectory_resample_move", 192, matched_runs),
        ("trajectory_resample_move", 512, base_runs),
    ):
        selected_rows = [
            row
            for row in confirmatory_components
            if row["method_key"] == method and int(row["particle_budget"]) == budget
        ]
        confirmatory_by_group.extend(
            _with_run_offset(
                selected_rows,
                offset=offset,
                cohort="confirmatory",
            )
        )
    components = base_components + matched_components + confirmatory_by_group
    run_rows = _run_rows(components, direct)
    summary_rows = _summary_rows(run_rows)
    bootstrap_cfg = study["bootstrap"]
    comparison_rows = _comparison_rows(
        run_rows,
        replicates=int(bootstrap_cfg["replicates"]),
        seed=int(bootstrap_cfg["seed"]),
    )
    diagnostics = _with_run_offset(_component_payload_rows(
        base_suite,
        base_test_path,
        output_root=output_root,
        selected={("fk_bootstrap", 512), ("trajectory_resample_move", 512)},
    ), offset=0, cohort="submission") + _with_run_offset(_component_payload_rows(
        matched_suite,
        matched_path,
        output_root=output_root,
        selected={("trajectory_resample_move", 192)},
    ), offset=0, cohort="runtime_matched")
    confirmatory_diagnostics = _component_payload_rows(
        confirmatory_suite,
        confirmatory_path,
        output_root=output_root,
        selected={
            ("fk_bootstrap", 192),
            ("fk_bootstrap", 512),
            ("trajectory_resample_move", 192),
            ("trajectory_resample_move", 512),
        },
    )
    for method, budget, offset in (
        ("fk_bootstrap", 192, 0),
        ("fk_bootstrap", 512, base_runs),
        ("trajectory_resample_move", 192, matched_runs),
        ("trajectory_resample_move", 512, base_runs),
    ):
        diagnostics.extend(
            _with_run_offset(
                [
                    row
                    for row in confirmatory_diagnostics
                    if row["method_key"] == method
                    and int(row["particle_budget"]) == budget
                ],
                offset=offset,
                cohort="confirmatory",
            )
        )
    audit_cfg = study["qualitative_audit"]
    audit_rows = _audit_rows(
        direct_scores,
        score_bands=audit_cfg["score_bands"],
        examples_per_band=int(audit_cfg["examples_per_score_band"]),
    )

    manifest = {
        "kind": "mdlm_final_cost_matched_analysis_v2",
        "schema_version": 1,
        "study_name": study["study_name"],
        "decoder_id": base_manifest["decoder_id"],
        "evaluator_id": base_manifest["evaluator_id"],
        "base_suite_id": fingerprint(base_manifest),
        "matched_runtime_suite_id": fingerprint(matched_manifest),
        "confirmatory_suite_id": fingerprint(confirmatory_manifest),
        "component_weight_policy": "uniform",
        "rare_event_thresholds": thresholds,
        "primary_threshold": float(study["primary_threshold"]),
        "exploratory_threshold": float(study["exploratory_threshold"]),
        "bootstrap": bootstrap_cfg,
    }
    report_dir = output_root / "mdlm_owt" / "final_analysis" / fingerprint(manifest)
    report = {
        "kind": "mdlm_final_cost_matched_report_v2",
        "manifest": manifest,
        "direct_reference": direct_rows,
        "summary": summary_rows,
        "comparisons": comparison_rows,
        "normalizer_diagnostic": {
            "identity": "log(Z_hat) + log(sum_i W_i exp(-lambda phi_i)) = 0",
            "median_absolute_log_residual": statistics.median(abs(float(row["log_identity_residual"])) for row in diagnostics),
            "max_absolute_log_residual": max(abs(float(row["log_identity_residual"])) for row in diagnostics),
        },
        "interpretation_notes": [
            "Uniform component weights are primary; pilot-optimized weights are excluded.",
            "Threshold 0.99 is primary; 0.999 is exploratory because the 200000-sample direct reference has only 26 discoveries.",
            "Bootstrap units are independent SMC runs, never individual particles.",
            "Confirmatory seeds are disjoint; pooled groups contain 64 independent runs where an earlier 32-run cohort exists.",
            "The qualitative audit sheet is intentionally unlabeled and must be reviewed manually.",
        ],
    }
    _write_json_compatible(report_dir / "manifest.json", manifest)
    _write_json_compatible(report_dir / "report.json", report)
    _write_text_compatible(report_dir / "direct_reference.csv", _csv_text(direct_rows))
    _write_text_compatible(report_dir / "run_estimates.csv", _csv_text(run_rows))
    _write_text_compatible(report_dir / "summary.csv", _csv_text(summary_rows))
    _write_text_compatible(report_dir / "bootstrap_comparisons.csv", _csv_text(comparison_rows))
    _write_text_compatible(report_dir / "normalizer_and_genealogy.csv", _csv_text(diagnostics))
    _write_text_compatible(report_dir / "qualitative_audit.csv", _csv_text(audit_rows))

    labels = {
        ("fk_bootstrap", 192): "FK N=192",
        ("fk_bootstrap", 512): "FK N=512",
        ("trajectory_resample_move", 512): "Trajectory MH N=512",
        ("trajectory_resample_move", 192): "Trajectory MH N=192",
    }
    direct_intervals = {float(row["threshold"]): (float(row["wilson_95_low"]), float(row["wilson_95_high"])) for row in direct_rows}
    for threshold in thresholds:
        selected = [row for row in run_rows if float(row["threshold"]) == threshold]
        groups = []
        error_groups = []
        for key, label in labels.items():
            values = [row for row in selected if row["method_key"] == key[0] and int(row["particle_budget"]) == key[1]]
            groups.append({"label": label, "method_key": key[0], "values": [float(row["estimate"]) for row in values]})
            error_groups.append({"label": label, "method_key": key[0], "values": [float(row["absolute_error"]) for row in values]})
        tag = format(threshold, ".12g").replace(".", "p")
        _write_text_compatible(
            report_dir / f"estimate_threshold_{tag}.svg",
            _boxplot_svg(
                title=f"Rare-event probability: score >= {threshold:g}",
                y_label="estimated probability",
                groups=groups,
                reference=direct[threshold],
                band=direct_intervals[threshold],
            ),
        )
        _write_text_compatible(
            report_dir / f"absolute_error_threshold_{tag}.svg",
            _boxplot_svg(
                title=f"Absolute error: score >= {threshold:g}",
                y_label="absolute error vs direct reference",
                groups=error_groups,
            ),
        )
    z_groups = []
    genealogy_groups = []
    for key, label in labels.items():
        selected = [row for row in diagnostics if row["method_key"] == key[0] and int(row["particle_budget"]) == key[1]]
        z_groups.append({"label": label, "method_key": key[0], "values": [abs(float(row["log_identity_residual"])) for row in selected]})
        genealogy_groups.append({"label": label, "method_key": key[0], "values": [float(row["distinct_root_fraction"]) for row in selected]})
    _write_text_compatible(
        report_dir / "normalizer_identity_residual.svg",
        _boxplot_svg(title="Normalizer consistency", y_label="absolute log-identity residual", groups=z_groups),
    )
    _write_text_compatible(
        report_dir / "genealogical_diversity.svg",
        _boxplot_svg(title="Genealogical diversity", y_label="distinct root ancestor fraction", groups=genealogy_groups),
    )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    study_path = args.study.resolve()
    destination = build_final_report(
        load_final_study(study_path),
        study_path=study_path,
        output_root=args.output_root,
    )
    print(destination)


if __name__ == "__main__":
    main()
