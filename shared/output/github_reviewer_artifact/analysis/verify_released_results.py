from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final_analysis"
COMPONENTS = ROOT / "data" / "components"
THRESHOLDS = {0.5, 0.9, 0.99, 0.999}
METHODS = {"fk_bootstrap", "trajectory_resample_move"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_numeric_columns(records: list[dict[str, str]]) -> None:
    for row_index, row in enumerate(records, start=2):
        for key, value in row.items():
            if value in ("", None, "True", "False"):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise AssertionError(f"non-finite {key} at CSV row {row_index}")


def main() -> None:
    manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((FINAL / "report.json").read_text(encoding="utf-8"))
    run_rows = rows(FINAL / "run_estimates.csv")
    component_rows = rows(COMPONENTS / "component_estimates_all_thresholds.csv")
    metric_rows = rows(COMPONENTS / "component_metrics.csv")

    finite_numeric_columns(run_rows)
    finite_numeric_columns(component_rows)
    finite_numeric_columns(metric_rows)

    assert {float(row["threshold"]) for row in run_rows} == THRESHOLDS
    assert {row["method_key"] for row in run_rows} == METHODS
    assert {float(row["lambda"]) for row in component_rows} == {0.25, 0.5, 0.75, 1.0}
    assert all(int(row["particle_budget"]) in {192, 512} for row in run_rows)
    # Seven method/budget/cohort cells x 32 runs x four thresholds.
    # FK N=192 was measured only in the confirmatory cohort.
    assert len(run_rows) == 896
    assert len(component_rows) == 3584
    assert len(metric_rows) == 896

    direct = rows(FINAL / "direct_reference.csv")
    assert len(direct) == 4
    assert all(int(row["samples"]) == 200000 for row in direct)

    print("Released-result verification passed")
    assert report["manifest"]["decoder_id"] == manifest["decoder_id"]
    assert report["manifest"]["evaluator_id"] == manifest["evaluator_id"]
    print(f"decoder_id={manifest['decoder_id']}")
    print(f"evaluator_id={manifest['evaluator_id']}")
    print(f"analysis_manifest_kind={manifest['kind']}")
    print(f"run estimates={len(run_rows)}; component estimates={len(component_rows)}")


if __name__ == "__main__":
    main()
