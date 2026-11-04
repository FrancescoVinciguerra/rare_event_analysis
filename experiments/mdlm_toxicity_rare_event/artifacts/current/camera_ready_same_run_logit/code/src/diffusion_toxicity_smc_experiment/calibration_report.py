"""Report terminal-lambda calibration components without aggregating them."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .configuration import decoder_id, evaluator_id, load_yaml, validate_benchmark_config
from .generate_direct import direct_run_directory
from .run_smc import smc_run_directory
from .score_direct import latest_direct_summary_path
from .storage import atomic_write_json


def build_calibration_report(
    config: dict[str, Any],
    *,
    direct_run: Path | None = None,
    calibration_run: Path | None = None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    validate_benchmark_config(config)
    if config["smc"].get("experiment_phase") != "calibration":
        raise ValueError("calibration report requires experiment_phase: calibration")
    expected_decoder = decoder_id(config)
    expected_evaluator = evaluator_id(config)
    direct_run = direct_run or direct_run_directory(config, output_root)
    calibration_run = calibration_run or smc_run_directory(config, output_root)
    direct_summary_path = latest_direct_summary_path(
        direct_run / "evaluations" / expected_evaluator
    )
    calibration_summary_path = calibration_run / "summary.json"
    if direct_summary_path is None or not calibration_summary_path.exists():
        raise FileNotFoundError("direct scoring or lambda-calibration summary is missing")
    direct = json.loads(direct_summary_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    for name, payload in (("direct", direct), ("calibration", calibration)):
        if payload.get("decoder_id") != expected_decoder:
            raise ValueError(f"{name} decoder_id mismatch")
        if payload.get("evaluator_id") != expected_evaluator:
            raise ValueError(f"{name} evaluator_id mismatch")
    if calibration.get("experiment_phase") != "calibration" or calibration.get(
        "aggregation_performed"
    ) is not False:
        raise ValueError("SMC artifact is not a non-aggregating calibration run")

    components = calibration.get("diagnostic_components", []) + calibration.get(
        "calibration_components", []
    )
    rows: list[dict[str, Any]] = []
    for component in components:
        quantiles = component["reward_quantiles"]
        rows.append(
            {
                "component_role": component["component_role"],
                "lambda": component["lambda"],
                "num_particles": component["num_particles"],
                "tilted_event_fraction": component[
                    "tilted_event_fraction_not_p0_estimate"
                ],
                "event_discoveries": component["tilted_event_discoveries"],
                "reverse_snis_estimate": component["reverse_snis_estimate"],
                "reverse_ess": component["reverse_weight_ess"],
                "reverse_ess_over_n": component["reverse_weight_ess_over_n"],
                "resampling_count": component["resampling_count"],
                "distinct_root_ancestors": component["distinct_root_ancestors"],
                "distinct_root_ancestor_fraction": component[
                    "distinct_root_ancestor_fraction"
                ],
                "pre_final_max_weight": component[
                    "pre_final_resampling_max_normalized_weight"
                ],
                "reverse_max_weight": component["reverse_max_normalized_weight"],
                "reward_q00": quantiles["q00"],
                "reward_q25": quantiles["q25"],
                "reward_q50": quantiles["q50"],
                "reward_q75": quantiles["q75"],
                "reward_q100": quantiles["q100"],
                "runtime_seconds": component["runtime_seconds"],
                "mdlm_nfe": component["nfe"],
                "classifier_model_forward_calls": component[
                    "classifier_model_forward_calls"
                ],
                "classifier_texts_scored": component["classifier_texts_scored"],
                "peak_cuda_memory_allocated_bytes": component[
                    "peak_cuda_memory_allocated_bytes"
                ],
                "nonfinite_detected": component["nonfinite_detected"],
            }
        )
    rows.sort(key=lambda row: (row["component_role"] != "diagnostic", row["lambda"]))

    lambda_zero = [
        row
        for row in rows
        if row["component_role"] == "diagnostic" and float(row["lambda"]) == 0.0
    ]
    if len(lambda_zero) != 1:
        raise ValueError("calibration report requires exactly one lambda=0 diagnostic")
    diagnostic = lambda_zero[0]
    direct_probability = float(direct["empirical_probability"])
    diagnostic_probability = float(diagnostic["tilted_event_fraction"])
    direct_count = int(direct["evaluated_samples"])
    diagnostic_count = int(diagnostic["num_particles"])
    mc_standard_error = math.sqrt(
        direct_probability * (1.0 - direct_probability) / direct_count
        + diagnostic_probability
        * (1.0 - diagnostic_probability)
        / diagnostic_count
    )
    difference = diagnostic_probability - direct_probability
    z_score = difference / mc_standard_error if mc_standard_error > 0.0 else None
    compatible = (
        abs(z_score) <= 2.0 if z_score is not None else math.isclose(difference, 0.0)
    )

    payload = {
        "decoder_id": expected_decoder,
        "evaluator_id": expected_evaluator,
        "experiment_phase": "calibration",
        "aggregation_performed": False,
        "selection_performed": False,
        "direct_summary_path": str(direct_summary_path),
        "direct": direct,
        "lambda_zero_direct_compatibility": {
            "definition": (
                "diagnostic difference divided by the independent direct-plus-"
                "lambda-zero Monte Carlo standard error; not an SMC confidence interval"
            ),
            "difference": difference,
            "mc_standard_error": mc_standard_error,
            "z_score": z_score,
            "within_two_mc_standard_errors": compatible,
        },
        "components": rows,
        "selection_guidance": (
            "Prefer reverse ESS/N > 0.1 (ideally > 0.2), nonzero events, and no "
            "severe genealogical collapse; choose terminal lambdas manually."
        ),
    }
    destination = output_dir or calibration_run / "calibration_report"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"direct_{direct_count:08d}"
    json_path = destination / f"{stem}.json"
    csv_path = destination / f"{stem}.csv"
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing calibration report is incompatible: {json_path}")
    else:
        atomic_write_json(json_path, payload)
    if not csv_path.exists():
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return payload, destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path)
    parser.add_argument("--calibration-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    payload, destination = build_calibration_report(
        load_yaml(args.config),
        direct_run=args.direct_run,
        calibration_run=args.calibration_run,
        output_root=args.output_root,
        output_dir=args.output_dir,
    )
    print(destination)
    print(json.dumps(payload["lambda_zero_direct_compatibility"], sort_keys=True))


if __name__ == "__main__":
    main()
