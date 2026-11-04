"""Export threshold-specific component estimates without rebuilding the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_analysis import (
    _csv_text,
    _relative,
    _with_run_offset,
    _write_text_compatible,
    load_final_study,
)
from .serious_comparison import comparison_manifest, load_suite
from .submission_probability import load_study, suite_component_estimates


def export_component_estimates(
    study: dict,
    *,
    study_path: Path,
    output_root: Path,
    destination: Path,
) -> Path:
    base_study_path = _relative(study_path.parent, str(study["base_submission_study"]))
    matched_path = _relative(study_path.parent, str(study["matched_runtime_suite"]))
    confirmatory_path = _relative(study_path.parent, str(study["confirmatory_suite"]))
    base_study = load_study(base_study_path)
    base_test_path = _relative(base_study_path.parent, str(base_study["suites"]["test"]))
    base_suite = load_suite(base_test_path)
    matched_suite = load_suite(matched_path)
    confirmatory_suite = load_suite(confirmatory_path)

    manifests = [
        comparison_manifest(base_suite, base_test_path),
        comparison_manifest(matched_suite, matched_path),
        comparison_manifest(confirmatory_suite, confirmatory_path),
    ]
    for key in ("decoder_id", "evaluator_id"):
        values = {manifest[key] for manifest in manifests}
        if values != {study[f"expected_{key}"]}:
            raise ValueError(f"{key} mismatch across final-analysis suites")

    thresholds = [float(value) for value in study["rare_event_thresholds"]]
    base_runs = int(base_suite["smc"]["independent_runs"])
    matched_runs = int(matched_suite["smc"]["independent_runs"])
    if base_runs != 32 or matched_runs != 32:
        raise ValueError("component export expects frozen 32-run source cohorts")

    rows = _with_run_offset(
        suite_component_estimates(
            base_suite,
            base_test_path,
            output_root=output_root,
            methods={"fk_bootstrap", "trajectory_resample_move"},
            budgets={512},
            thresholds=thresholds,
        ),
        offset=0,
        cohort="submission",
    )
    rows.extend(
        _with_run_offset(
            suite_component_estimates(
                matched_suite,
                matched_path,
                output_root=output_root,
                methods={"trajectory_resample_move"},
                budgets={192},
                thresholds=thresholds,
            ),
            offset=0,
            cohort="runtime_matched",
        )
    )
    confirmatory = suite_component_estimates(
        confirmatory_suite,
        confirmatory_path,
        output_root=output_root,
        methods={"fk_bootstrap", "trajectory_resample_move"},
        budgets={192, 512},
        thresholds=thresholds,
    )
    for method, budget, offset in (
        ("fk_bootstrap", 192, 0),
        ("fk_bootstrap", 512, base_runs),
        ("trajectory_resample_move", 192, matched_runs),
        ("trajectory_resample_move", 512, base_runs),
    ):
        rows.extend(
            _with_run_offset(
                [
                    row
                    for row in confirmatory
                    if row["method_key"] == method
                    and int(row["particle_budget"]) == budget
                ],
                offset=offset,
                cohort="confirmatory",
            )
        )

    expected_rows = (32 * 2 + 32 + 32 * 4) * 4 * len(thresholds)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} component rows, found {len(rows)}")
    if any(float(row["reverse_ess_over_n"]) <= 0.0 for row in rows):
        raise ValueError("component export contains a non-positive reverse ESS")

    destination = destination.resolve()
    _write_text_compatible(destination, _csv_text(rows))
    manifest = {
        "kind": "mdlm_final_component_export",
        "schema_version": 1,
        "study_name": study["study_name"],
        "decoder_id": study["expected_decoder_id"],
        "evaluator_id": study["expected_evaluator_id"],
        "thresholds": thresholds,
        "rows": len(rows),
        "csv": destination.name,
    }
    _write_text_compatible(
        destination.with_suffix(".manifest.json"),
        json.dumps(manifest, indent=2) + "\n",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    study_path = args.study.resolve()
    result = export_component_estimates(
        load_final_study(study_path),
        study_path=study_path,
        output_root=args.output_root.resolve(),
        destination=args.destination,
    )
    print(result)


if __name__ == "__main__":
    main()
