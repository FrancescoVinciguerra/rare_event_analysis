"""Compare existing unconditional direct and SMC estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .configuration import decoder_id, evaluator_id, load_yaml
from .generate_direct import direct_run_directory
from .run_smc import smc_run_directory
from .score_direct import latest_direct_summary_path
from .storage import atomic_write_json


def compare_existing(
    config: dict[str, Any],
    *,
    direct_run: Path | None,
    smc_run: Path | None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    expected_decoder = decoder_id(config)
    expected_evaluator = evaluator_id(config)
    direct_run = direct_run or direct_run_directory(config, output_root)
    smc_run = smc_run or smc_run_directory(config, output_root)
    direct_eval_dir = direct_run / "evaluations" / expected_evaluator
    direct_path = latest_direct_summary_path(direct_eval_dir)
    smc_path = smc_run / "summary.json"
    if direct_path is None or not smc_path.exists():
        raise FileNotFoundError("direct or SMC summary is missing; comparison never regenerates")
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    smc = json.loads(smc_path.read_text(encoding="utf-8"))
    for name, payload in (("direct", direct), ("SMC", smc)):
        if payload.get("decoder_id") != expected_decoder:
            raise ValueError(f"{name} decoder_id does not match the unconditional P0")
        if payload.get("evaluator_id") != expected_evaluator:
            raise ValueError(f"{name} evaluator_id/event definition does not match")
        if payload.get("conditioning_mode") != "unconditional":
            raise ValueError(f"{name} artifact is not unconditional")

    if smc.get("experiment_phase") == "calibration" or not smc.get(
        "aggregation_performed", True
    ):
        raise ValueError(
            "lambda-calibration components are diagnostics and cannot be compared "
            "as an aggregated SMC estimator"
        )

    tilted = smc["tilted_components"]
    diagnostics = smc.get("diagnostic_components", [])
    aggregate = smc["aggregate"]
    row = {
        "direct_estimate": direct["empirical_probability"],
        "direct_binomial_standard_error": direct["binomial_standard_error"],
        "direct_wilson_low": direct["wilson_95_low"],
        "direct_wilson_high": direct["wilson_95_high"],
        "smc_estimate": aggregate["mean_reverse_snis_estimate"],
        "smc_empirical_standard_error": aggregate[
            "empirical_standard_error_across_runs"
        ],
        "difference_smc_minus_direct": (
            aggregate["mean_reverse_snis_estimate"] - direct["empirical_probability"]
        ),
        "direct_event_discoveries": direct["event_discoveries"],
        "tilted_event_discoveries": sum(
            int(component["tilted_event_discoveries"]) for component in tilted
        ),
        "diagnostic_event_discoveries": sum(
            int(component["tilted_event_discoveries"]) for component in diagnostics
        ),
        "direct_generation_runtime_seconds": direct.get("generation_runtime_seconds"),
        "smc_tilted_runtime_seconds": sum(
            float(component["runtime_seconds"]) for component in tilted
        ),
        "smc_diagnostic_runtime_seconds": sum(
            float(component["runtime_seconds"]) for component in diagnostics
        ),
        "direct_nfe": direct.get("generation_nfe"),
        "smc_tilted_nfe": sum(int(component["nfe"]) for component in tilted),
        "smc_diagnostic_nfe": sum(
            int(component["nfe"]) for component in diagnostics
        ),
    }
    destination = output_dir or smc_run / "comparison"
    json_path = destination / "comparison.json"
    csv_path = destination / "comparison.csv"
    payload = {
        "conditioning_mode": "unconditional",
        "decoder_id": expected_decoder,
        "evaluator_id": expected_evaluator,
        "direct_iid_estimator": direct,
        "smc_estimator": aggregate,
        "probability_estimation": row,
        "aggregation_definition": "p_hat = sum_j a_j * p_hat_j",
        "diagnostic_components": diagnostics,
        "fk_prompt_conditioned_rate_comparable": False,
        "rare_event_discovery_note": (
            "Tilted event fractions are discovery diagnostics only and are never P0 estimates."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing comparison is incompatible: {json_path}")
    else:
        atomic_write_json(json_path, payload)
    if not csv_path.exists():
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path)
    parser.add_argument("--smc-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(
        compare_existing(
            load_yaml(args.config),
            direct_run=args.direct_run,
            smc_run=args.smc_run,
            output_root=args.output_root,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
