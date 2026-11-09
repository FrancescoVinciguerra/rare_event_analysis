"""Run and report the resumable three-sampler MDLM comparison suite."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .configuration import canonical_json, decoder_id, evaluator_id, fingerprint, load_yaml
from .evaluation import direct_unconditional_summary
from .generate_direct import direct_run_directory
from .run_smc import smc_manifest, smc_run_directory
from .storage import (
    atomic_write_json,
    initialize_manifest,
    read_jsonl,
    scan_direct_generations,
)


SUITE_KIND = "mdlm_three_sampler_serious_comparison_v1"
SUBSET_SUITE_KIND = "mdlm_configured_sampler_serious_comparison_v1"
KNOWN_SAMPLERS = {
    "fk_bootstrap",
    "trajectory_resample_move",
    "trajectory_waste_free",
}
BUDGET_SEED_STRIDE = 10_000_000
METHOD_COLOURS = {
    "direct": "#444444",
    "fk_bootstrap": "#4472c4",
    "trajectory_resample_move": "#ed7d31",
    "trajectory_waste_free": "#70ad47",
}


def load_suite(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict):
        raise ValueError(f"comparison suite must be a mapping: {path}")
    return suite


def _config_path(suite_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else suite_path.parent / candidate


def resolved_comparison_entries(
    suite: dict[str, Any], suite_path: Path
) -> list[dict[str, Any]]:
    smc = suite.get("smc")
    if not isinstance(smc, dict):
        raise ValueError("suite.smc must be a mapping")
    budgets = smc.get("particle_budgets")
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(not isinstance(value, int) or value <= 0 for value in budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("particle_budgets must contain unique positive integers")
    runs = smc.get("independent_runs")
    if not isinstance(runs, int) or runs < 2:
        raise ValueError("serious comparison requires at least two independent runs")
    base_seed = smc.get("base_seed")
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("smc.base_seed must be a non-negative integer")
    budget_seed_stride = smc.get("budget_seed_stride", BUDGET_SEED_STRIDE)
    if not isinstance(budget_seed_stride, int) or budget_seed_stride <= 0:
        raise ValueError("smc.budget_seed_stride must be a positive integer")
    methods = smc.get("methods")
    sampler_set_policy = str(suite.get("sampler_set_policy", "strict_three"))
    if sampler_set_policy not in {"strict_three", "configured_subset"}:
        raise ValueError(
            "sampler_set_policy must be strict_three or configured_subset"
        )
    if not isinstance(methods, list) or not methods:
        raise ValueError("serious comparison requires at least one SMC method")
    if sampler_set_policy == "strict_three" and len(methods) != 3:
        raise ValueError("strict_three requires exactly three SMC methods")

    entries: list[dict[str, Any]] = []
    keys: set[str] = set()
    sampler_types: set[str] = set()
    seed_offsets: set[int] = set()
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("every SMC method must be a mapping")
        key = str(method.get("key", ""))
        label = str(method.get("label", ""))
        if not key or key in keys or not label:
            raise ValueError("method keys must be unique and labels non-empty")
        keys.add(key)
        seed_offset = method.get("seed_offset")
        if not isinstance(seed_offset, int) or seed_offset < 0:
            raise ValueError("method seed_offset must be a non-negative integer")
        if seed_offset in seed_offsets:
            raise ValueError("method seed offsets must be distinct")
        seed_offsets.add(seed_offset)
        config_path = _config_path(suite_path, str(method.get("config", "")))
        base_config = load_yaml(config_path)
        sampler = base_config["smc"].get("tilted_sampler", {"type": "fk_bootstrap"})
        sampler_type = str(sampler.get("type", "fk_bootstrap"))
        sampler_types.add(sampler_type)
        if key != sampler_type:
            raise ValueError(
                f"method key {key!r} does not match sampler type {sampler_type!r}"
            )
        for budget_index, budget in enumerate(budgets):
            config = json.loads(canonical_json(base_config))
            config["smc"]["num_particles"] = budget
            config["smc"]["independent_runs"] = runs
            config["smc"]["num_independent_runs"] = runs
            # Each budget is a separate estimator. Use disjoint random
            # populations instead of silently nesting the same seed stream.
            config["smc"]["base_seed"] = (
                base_seed + seed_offset + budget_seed_stride * budget_index
            )
            # smc_manifest performs the full benchmark validation, including
            # the N=M*P divisibility constraint for waste-free SMC.
            manifest = smc_manifest(config)
            entries.append(
                {
                    "method_key": key,
                    "method_label": label,
                    "particle_budget": budget,
                    "config_path": str(config_path),
                    "config": config,
                    "smc_manifest": manifest,
                    "smc_id": fingerprint(manifest),
                }
            )
    if not sampler_types.issubset(KNOWN_SAMPLERS):
        unknown = sorted(sampler_types.difference(KNOWN_SAMPLERS))
        raise ValueError(f"unsupported serious-comparison samplers: {unknown}")
    if sampler_set_policy == "strict_three" and sampler_types != KNOWN_SAMPLERS:
        raise ValueError(
            "suite must contain exactly FK bootstrap, trajectory resample-move, "
            "and trajectory waste-free"
        )
    component_seeds: set[int] = set()
    for entry in entries:
        entry_smc = entry["config"]["smc"]
        component_count = len(entry_smc["tilted_lambdas"])
        for run_index in range(runs):
            for component_index in range(component_count):
                seed = (
                    int(entry_smc["base_seed"])
                    + 1_000_000 * run_index
                    + component_index
                )
                if seed in component_seeds:
                    raise ValueError(
                        "serious comparison reuses an SMC component RNG seed"
                    )
                component_seeds.add(seed)
    return entries


def comparison_manifest(
    suite: dict[str, Any], suite_path: Path
) -> dict[str, Any]:
    direct_spec = suite.get("direct")
    if not isinstance(direct_spec, dict):
        raise ValueError("suite.direct must be a mapping")
    target = direct_spec.get("target_samples")
    block_size = direct_spec.get("reference_block_size")
    if not isinstance(target, int) or target <= 0:
        raise ValueError("direct.target_samples must be positive")
    if not isinstance(block_size, int) or block_size <= 1 or target % block_size:
        raise ValueError("reference_block_size must divide direct.target_samples")
    direct_generation_path = _config_path(
        suite_path, str(direct_spec.get("generation_config", ""))
    )
    direct_evaluation_path = _config_path(
        suite_path, str(direct_spec.get("evaluation_config", ""))
    )
    direct_generation_config = load_yaml(direct_generation_path)
    direct_evaluation_config = load_yaml(direct_evaluation_path)
    entries = resolved_comparison_entries(suite, suite_path)
    expected_decoder = decoder_id(direct_generation_config)
    if decoder_id(direct_evaluation_config) != expected_decoder:
        raise ValueError("direct generation and evaluation configs must share decoder_id")
    expected_evaluator = evaluator_id(direct_evaluation_config)
    for entry in entries:
        if entry["smc_manifest"]["decoder_id"] != expected_decoder:
            raise ValueError("all serious-comparison methods must share decoder_id")
        if entry["smc_manifest"]["evaluator_id"] != expected_evaluator:
            raise ValueError("all serious-comparison methods must share evaluator_id")
    sampler_set_policy = str(suite.get("sampler_set_policy", "strict_three"))
    smc_manifest_payload = {
        "particle_budgets": list(suite["smc"]["particle_budgets"]),
        "independent_runs": int(suite["smc"]["independent_runs"]),
        "base_seed": int(suite["smc"]["base_seed"]),
        "resolved_grid": [
            {
                "method_key": entry["method_key"],
                "method_label": entry["method_label"],
                "particle_budget": entry["particle_budget"],
                "smc_id": entry["smc_id"],
                "smc_manifest": entry["smc_manifest"],
            }
            for entry in entries
        ],
    }
    if "budget_seed_stride" in suite["smc"]:
        smc_manifest_payload["budget_seed_stride"] = int(
            suite["smc"]["budget_seed_stride"]
        )
    # Preserve the exact legacy manifest and fingerprint for the completed
    # three-sampler suite.  The policy marker is needed only by new subset
    # suites, which intentionally receive a new identity.
    if sampler_set_policy == "configured_subset":
        smc_manifest_payload["sampler_set_policy"] = sampler_set_policy
    return {
        "kind": (
            SUITE_KIND
            if sampler_set_policy == "strict_three"
            else SUBSET_SUITE_KIND
        ),
        "schema_version": int(suite.get("schema_version", 1)),
        "suite_name": str(suite.get("suite_name", "")),
        "decoder_id": expected_decoder,
        "evaluator_id": expected_evaluator,
        "direct": {
            "generation_config_path": str(direct_generation_path),
            "evaluation_config_path": str(direct_evaluation_path),
            "target_samples": target,
            "reference_block_size": block_size,
        },
        "smc": smc_manifest_payload,
        "report": suite.get("report", {}),
    }


def comparison_run_directory(
    suite: dict[str, Any], suite_path: Path, output_root: Path
) -> Path:
    manifest = comparison_manifest(suite, suite_path)
    return output_root / "mdlm_owt" / "comparisons" / fingerprint(manifest)


def _write_text_compatible(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json_compatible(path: Path, payload: Any) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _write_text_compatible(path, rendered)


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run_comparison_suite(
    suite: dict[str, Any], *, suite_path: Path, output_root: Path
) -> Path:
    manifest = comparison_manifest(suite, suite_path)
    suite_dir = comparison_run_directory(suite, suite_path, output_root)
    initialize_manifest(suite_dir, {"suite_id": fingerprint(manifest), **manifest})
    entries = resolved_comparison_entries(suite, suite_path)
    completed: list[dict[str, Any]] = []
    for position, entry in enumerate(entries, start=1):
        config = entry["config"]
        run_dir = smc_run_directory(config, output_root)
        summary_path = run_dir / "summary.json"
        resolved_path = suite_dir / "resolved_configs" / (
            f"{entry['method_key']}_n{entry['particle_budget']:04d}.json"
        )
        _write_json_compatible(resolved_path, config)
        reloaded_config = load_yaml(resolved_path)
        if canonical_json(reloaded_config) != canonical_json(config):
            raise RuntimeError(
                f"resolved SMC config changed during JSON round-trip: {resolved_path}"
            )
        reloaded_run_dir = smc_run_directory(reloaded_config, output_root)
        if reloaded_run_dir != run_dir:
            raise RuntimeError(
                "resolved SMC config changed its output identity: "
                f"expected {run_dir}, reloaded {reloaded_run_dir}"
            )
        reused = summary_path.exists()
        print(
            f"comparison_grid_item={position}/{len(entries)} "
            f"method={entry['method_key']} particles={entry['particle_budget']} "
            f"resume_complete={str(reused).lower()}",
            flush=True,
        )
        if not reused:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "diffusion_toxicity_smc_experiment.run_smc",
                    "--config",
                    str(resolved_path),
                    "--output-root",
                    str(output_root),
                ],
                check=True,
            )
        if not summary_path.exists():
            raise RuntimeError(f"SMC run did not produce its summary: {summary_path}")
        completed.append(
            {
                "method_key": entry["method_key"],
                "particle_budget": entry["particle_budget"],
                "smc_id": entry["smc_id"],
                "run_directory": str(run_dir),
            }
        )
    _write_json_compatible(
        suite_dir / "completion.json",
        {
            "kind": "serious_comparison_grid_completion_v1",
            "suite_id": fingerprint(manifest),
            "completed_grid_items": len(completed),
            "grid": completed,
        },
    )
    return suite_dir


def _direct_reference(
    generation_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    *,
    target: int,
    block_size: int,
    output_root: Path,
) -> tuple[dict[str, Any], list[float]]:
    run_dir = direct_run_directory(generation_config, output_root)
    inventory = scan_direct_generations(run_dir, decoder_id(generation_config))
    by_index = {
        int(row["sample_index"]): row
        for row in inventory.rows
        if int(row["sample_index"]) < target
    }
    if set(by_index) != set(range(target)):
        raise ValueError(f"direct reference does not contain indices 0..{target - 1}")
    eval_dir = run_dir / "evaluations" / evaluator_id(evaluation_config)
    scores: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_dir.glob("scores_*.jsonl")):
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in scores:
                raise ValueError(f"duplicate direct score for {sample_id}")
            scores[sample_id] = row
    ordered_scores = []
    for index in range(target):
        sample_id = str(by_index[index]["sample_id"])
        if sample_id not in scores:
            raise ValueError(f"direct sample {index} has not been scored")
        ordered_scores.append(scores[sample_id])
    summary = direct_unconditional_summary(ordered_scores)
    summary.update(
        {
            "decoder_id": decoder_id(generation_config),
            "evaluator_id": evaluator_id(evaluation_config),
            "target_samples": target,
            "reference_role": "high_precision_iid_monte_carlo_reference_not_exact_truth",
            # Resume may extend an older run using a different append-only
            # shard size. Count the persisted independent seeds rather than
            # inferring them from the current storage configuration.
            "independent_shard_seed_count": len(
                {int(row["shard_seed"]) for row in by_index.values()}
            ),
            "run_directory": str(run_dir),
        }
    )
    blocks = [
        sum(bool(row["is_event"]) for row in ordered_scores[start : start + block_size])
        / block_size
        for start in range(0, target, block_size)
    ]
    return summary, blocks


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _group_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method_key"], row["method_label"], row["particle_budget"])].append(row)
    output = []
    for (key, label, budget), values in grouped.items():
        estimates = [float(row["estimate"]) for row in values]
        errors = [float(row["error"]) for row in values]
        output.append(
            {
                "method_key": key,
                "method_label": label,
                "particle_budget": budget,
                "independent_runs": len(values),
                "mean_estimate": statistics.fmean(estimates),
                "estimate_standard_deviation": statistics.stdev(estimates),
                "bias_vs_direct_reference": statistics.fmean(errors),
                "mae_vs_direct_reference": statistics.fmean(abs(value) for value in errors),
                "rmse_vs_direct_reference": math.sqrt(
                    statistics.fmean(value * value for value in errors)
                ),
                "median_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in values
                ),
                "median_decoder_particle_steps": statistics.median(
                    float(row["decoder_particle_steps"]) for row in values
                ),
                "median_classifier_texts_scored": statistics.median(
                    float(row["classifier_texts_scored"]) for row in values
                ),
                "median_distinct_root_fraction": statistics.median(
                    float(row["mean_distinct_root_fraction"]) for row in values
                ),
                "median_reverse_ess_over_n": statistics.median(
                    float(row["mean_reverse_ess_over_n"]) for row in values
                ),
                "median_mcmc_acceptance_rate": (
                    statistics.median(
                        float(row["mcmc_acceptance_rate"])
                        for row in values
                        if row["mcmc_acceptance_rate"] is not None
                    )
                    if any(row["mcmc_acceptance_rate"] is not None for row in values)
                    else None
                ),
            }
        )
    return sorted(output, key=lambda row: (row["particle_budget"], row["method_key"]))


def _matched_budget_rows(
    grouped: Sequence[dict[str, Any]], metric: str
) -> tuple[float, list[dict[str, Any]]]:
    methods = sorted({str(row["method_key"]) for row in grouped})
    maxima = [max(float(row[metric]) for row in grouped if row["method_key"] == key) for key in methods]
    target = min(maxima)
    selected = []
    for key in methods:
        candidates = [row for row in grouped if row["method_key"] == key]
        best = min(candidates, key=lambda row: abs(float(row[metric]) - target))
        selected.append(
            {
                **best,
                "matching_metric": metric,
                "shared_target": target,
                "relative_target_mismatch": abs(float(best[metric]) - target) / target,
            }
        )
    return target, selected


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _box_statistics(values: Sequence[float]) -> dict[str, Any]:
    q1, median, q3 = (_quantile(values, value) for value in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    low_candidates = [value for value in values if value >= q1 - 1.5 * iqr]
    high_candidates = [value for value in values if value <= q3 + 1.5 * iqr]
    low = min(low_candidates) if low_candidates else min(values)
    high = max(high_candidates) if high_candidates else max(values)
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "low": low,
        "high": high,
        "outliers": [value for value in values if value < low or value > high],
    }


def _boxplot_svg(
    *,
    title: str,
    y_label: str,
    groups: Sequence[dict[str, Any]],
    reference: float | None = None,
    band: tuple[float, float] | None = None,
) -> str:
    width = max(980, 105 * len(groups) + 160)
    height = 600
    left, right, top, bottom = 90, 30, 55, 145
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = [float(value) for group in groups for value in group["values"]]
    if reference is not None:
        all_values.append(reference)
    if band is not None:
        all_values.extend(band)
    low, high = min(all_values), max(all_values)
    pad = (high - low) * 0.08 if high > low else max(abs(high) * 0.08, 1e-3)
    low, high = low - pad, high + pad

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="19">{html.escape(title)}</text>',
    ]
    if band is not None:
        y1, y2 = y(band[1]), y(band[0])
        pieces.append(
            f'<rect x="{left}" y="{y1:.2f}" width="{plot_width}" height="{y2-y1:.2f}" fill="#888888" opacity="0.16"/>'
        )
    if reference is not None:
        pieces.append(
            f'<line x1="{left}" y1="{y(reference):.2f}" x2="{width-right}" y2="{y(reference):.2f}" stroke="#333333" stroke-width="2" stroke-dasharray="7 5"/>'
        )
    for tick in range(6):
        value = low + tick * (high - low) / 5
        yy = y(value)
        pieces.extend(
            [
                f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#dddddd"/>',
                f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.4g}</text>',
            ]
        )
    spacing = plot_width / len(groups)
    for index, group in enumerate(groups):
        stats = _box_statistics([float(value) for value in group["values"]])
        x = left + spacing * (index + 0.5)
        box_width = min(44, spacing * 0.55)
        colour = METHOD_COLOURS.get(str(group["method_key"]), "#777777")
        pieces.extend(
            [
                f'<line x1="{x:.2f}" y1="{y(stats["low"]):.2f}" x2="{x:.2f}" y2="{y(stats["high"]):.2f}" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x-box_width/3:.2f}" y1="{y(stats["low"]):.2f}" x2="{x+box_width/3:.2f}" y2="{y(stats["low"]):.2f}" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x-box_width/3:.2f}" y1="{y(stats["high"]):.2f}" x2="{x+box_width/3:.2f}" y2="{y(stats["high"]):.2f}" stroke="{colour}" stroke-width="2"/>',
                f'<rect x="{x-box_width/2:.2f}" y="{y(stats["q3"]):.2f}" width="{box_width:.2f}" height="{y(stats["q1"])-y(stats["q3"]):.2f}" fill="{colour}" fill-opacity="0.28" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x-box_width/2:.2f}" y1="{y(stats["median"]):.2f}" x2="{x+box_width/2:.2f}" y2="{y(stats["median"]):.2f}" stroke="{colour}" stroke-width="3"/>',
            ]
        )
        for outlier in stats["outliers"]:
            pieces.append(
                f'<circle cx="{x:.2f}" cy="{y(outlier):.2f}" r="3" fill="none" stroke="{colour}"/>'
            )
        label = html.escape(str(group["label"]))
        pieces.append(
            f'<text transform="translate({x:.2f},{height-bottom+18}) rotate(38)" text-anchor="start" font-family="Arial" font-size="11">{label}</text>'
        )
    pieces.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333333"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333333"/>',
            f'<text transform="translate(22,{top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(y_label)}</text>',
            '</svg>\n',
        ]
    )
    return "".join(pieces)


def _frontier_svg(
    *, title: str, x_label: str, grouped: Sequence[dict[str, Any]], metric: str
) -> str:
    width, height = 920, 560
    left, right, top, bottom = 90, 35, 55, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    xs = [float(row[metric]) for row in grouped]
    ys = [float(row["rmse_vs_direct_reference"]) for row in grouped]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin) * 0.08 if xmax > xmin else max(xmax * 0.08, 1.0)
    ypad = (ymax - ymin) * 0.10 if ymax > ymin else max(ymax * 0.1, 1e-3)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, max(0.0, ymin - ypad), ymax + ypad

    def x(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_width

    def y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="19">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        xv = xmin + tick * (xmax - xmin) / 5
        yv = ymin + tick * (ymax - ymin) / 5
        xx, yy = x(xv), y(yv)
        pieces.extend(
            [
                f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#eeeeee"/>',
                f'<text x="{xx:.2f}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.4g}</text>',
                f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#eeeeee"/>',
                f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.4g}</text>',
            ]
        )
    for method in sorted({str(row["method_key"]) for row in grouped}):
        rows = sorted(
            (row for row in grouped if row["method_key"] == method),
            key=lambda row: float(row[metric]),
        )
        colour = METHOD_COLOURS[method]
        points = " ".join(
            f'{x(float(row[metric])):.2f},{y(float(row["rmse_vs_direct_reference"])):.2f}'
            for row in rows
        )
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>'
        )
        for row in rows:
            xx, yy = x(float(row[metric])), y(float(row["rmse_vs_direct_reference"]))
            pieces.extend(
                [
                    f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="{colour}"/>',
                    f'<text x="{xx+7:.2f}" y="{yy-7:.2f}" font-family="Arial" font-size="10">N={row["particle_budget"]}</text>',
                ]
            )
        last = rows[-1]
        pieces.append(
            f'<text x="{x(float(last[metric]))+8:.2f}" y="{y(float(last["rmse_vs_direct_reference"]))+16:.2f}" font-family="Arial" font-size="11" fill="{colour}">{html.escape(str(last["method_label"]))}</text>'
        )
    pieces.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333333"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333333"/>',
            f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)}</text>',
            f'<text transform="translate(22,{top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">RMSE vs direct reference</text>',
            '</svg>\n',
        ]
    )
    return "".join(pieces)


def build_comparison_report(
    suite: dict[str, Any], *, suite_path: Path, output_root: Path
) -> Path:
    manifest = comparison_manifest(suite, suite_path)
    suite_dir = comparison_run_directory(suite, suite_path, output_root)
    initialize_manifest(suite_dir, {"suite_id": fingerprint(manifest), **manifest})
    if not (suite_dir / "completion.json").exists():
        raise FileNotFoundError("SMC comparison grid is incomplete")
    direct_generation_config = load_yaml(
        Path(manifest["direct"]["generation_config_path"])
    )
    direct_evaluation_config = load_yaml(
        Path(manifest["direct"]["evaluation_config_path"])
    )
    direct, direct_blocks = _direct_reference(
        direct_generation_config,
        direct_evaluation_config,
        target=int(manifest["direct"]["target_samples"]),
        block_size=int(manifest["direct"]["reference_block_size"]),
        output_root=output_root,
    )
    run_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    entries = resolved_comparison_entries(suite, suite_path)
    for entry in entries:
        summary_path = smc_run_directory(entry["config"], output_root) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing SMC summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("decoder_id") != manifest["decoder_id"]:
            raise ValueError("SMC summary decoder_id mismatch")
        if summary.get("evaluator_id") != manifest["evaluator_id"]:
            raise ValueError("SMC summary evaluator_id mismatch")
        if summary.get("smc_id") != entry["smc_id"]:
            raise ValueError("SMC summary fingerprint mismatch")
        components_by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for component in summary["tilted_components"]:
            components_by_run[int(component["run_index"])].append(component)
            component_rows.append(
                {
                    "method_key": entry["method_key"],
                    "method_label": entry["method_label"],
                    "particle_budget": entry["particle_budget"],
                    "run_index": int(component["run_index"]),
                    "lambda": float(component["lambda"]),
                    "seed": int(component["seed"]),
                    "estimate": float(component["reverse_snis_estimate"]),
                    "runtime_seconds": float(component["runtime_seconds"]),
                    "decoder_particle_steps": int(component["total_decoder_particle_steps"]),
                    "classifier_texts_scored": int(component["classifier_texts_scored"]),
                    "reverse_ess_over_n": float(component["reverse_weight_ess_over_n"]),
                    "distinct_root_fraction": float(component["distinct_root_ancestor_fraction"]),
                    "root_entropy_normalized": float(component["normalized_root_ancestor_entropy"]),
                    "event_discoveries": int(component["tilted_event_discoveries"]),
                    "resampling_count": int(component["resampling_count"]),
                    "mcmc_proposals": int(component["mcmc_proposals"]),
                    "mcmc_accepted": int(component["mcmc_accepted"]),
                    "peak_cuda_memory_allocated_bytes": int(component["peak_cuda_memory_allocated_bytes"]),
                }
            )
        estimates = [float(value) for value in summary["aggregate"]["run_estimates"]]
        expected_runs = int(suite["smc"]["independent_runs"])
        if len(estimates) != expected_runs or set(components_by_run) != set(range(expected_runs)):
            raise ValueError("SMC summary does not contain every requested independent run")
        for run_index, estimate in enumerate(estimates):
            components = components_by_run[run_index]
            expected_components = len(entry["config"]["smc"]["tilted_lambdas"])
            if len(components) != expected_components:
                raise ValueError("independent SMC run is missing a tilted component")
            proposals = sum(int(row["mcmc_proposals"]) for row in components)
            accepted = sum(int(row["mcmc_accepted"]) for row in components)
            error = estimate - float(direct["empirical_probability"])
            run_rows.append(
                {
                    "method_key": entry["method_key"],
                    "method_label": entry["method_label"],
                    "particle_budget": entry["particle_budget"],
                    "run_index": run_index,
                    "estimate": estimate,
                    "direct_reference": float(direct["empirical_probability"]),
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error * error,
                    "runtime_seconds": sum(float(row["runtime_seconds"]) for row in components),
                    "decoder_particle_steps": sum(int(row["total_decoder_particle_steps"]) for row in components),
                    "nfe": sum(int(row["nfe"]) for row in components),
                    "classifier_texts_scored": sum(int(row["classifier_texts_scored"]) for row in components),
                    "event_discoveries": sum(int(row["tilted_event_discoveries"]) for row in components),
                    "resampling_count": sum(int(row["resampling_count"]) for row in components),
                    "mean_distinct_root_fraction": _mean(float(row["distinct_root_ancestor_fraction"]) for row in components),
                    "mean_root_entropy_normalized": _mean(float(row["normalized_root_ancestor_entropy"]) for row in components),
                    "mean_reverse_ess_over_n": _mean(float(row["reverse_weight_ess_over_n"]) for row in components),
                    "mcmc_proposals": proposals,
                    "mcmc_acceptance_rate": accepted / proposals if proposals else None,
                    "peak_cuda_memory_allocated_bytes": max(int(row["peak_cuda_memory_allocated_bytes"]) for row in components),
                }
            )
    grouped = _group_rows(run_rows)
    time_target, equal_time = _matched_budget_rows(grouped, "median_runtime_seconds")
    compute_target, equal_compute = _matched_budget_rows(
        grouped, "median_decoder_particle_steps"
    )
    report_dir = suite_dir / "report"
    report = {
        "kind": "mdlm_three_sampler_serious_report_v1",
        "suite_id": fingerprint(manifest),
        "decoder_id": manifest["decoder_id"],
        "evaluator_id": manifest["evaluator_id"],
        "direct_reference": direct,
        "direct_reference_blocks": direct_blocks,
        "group_summary": grouped,
        "equal_runtime_comparison": {
            "target_seconds": time_target,
            "selected_rows": equal_time,
            "selection_rule": "nearest median runtime to the largest target attainable by every method",
        },
        "equal_decoder_compute_comparison": {
            "target_particle_steps": compute_target,
            "selected_rows": equal_compute,
            "selection_rule": "nearest median decoder particle-step count to the largest target attainable by every method",
        },
        "compute_accounting_note": (
            "Wall time includes decoder, classifier and sampler overhead inside each component. "
            "Decoder particle-steps and classifier texts are also reported separately; they are not collapsed into an arbitrary scalar."
        ),
    }
    _write_json_compatible(report_dir / "serious_report.json", report)
    _write_text_compatible(report_dir / "independent_runs.csv", _csv_text(run_rows))
    _write_text_compatible(report_dir / "components.csv", _csv_text(component_rows))
    _write_text_compatible(report_dir / "group_summary.csv", _csv_text(grouped))
    _write_text_compatible(report_dir / "equal_runtime.csv", _csv_text(equal_time))
    _write_text_compatible(report_dir / "equal_decoder_compute.csv", _csv_text(equal_compute))

    ordered_groups = []
    for budget in suite["smc"]["particle_budgets"]:
        for method in suite["smc"]["methods"]:
            selected = [
                row for row in run_rows
                if row["particle_budget"] == budget and row["method_key"] == method["key"]
            ]
            ordered_groups.append(
                {
                    "method_key": method["key"],
                    "label": f"{method['label']} N={budget}",
                    "rows": selected,
                }
            )
    estimate_groups = [
        {"method_key": "direct", "label": "Direct blocks", "values": direct_blocks}
    ] + [
        {**group, "values": [float(row["estimate"]) for row in group["rows"]]}
        for group in ordered_groups
    ]
    other_plots = {
        "absolute_error_boxplot.svg": (
            "Absolute error across independent runs",
            "|estimate - direct reference|",
            "absolute_error",
        ),
        "runtime_boxplot.svg": (
            "Runtime across independent runs",
            "seconds for four tilted components",
            "runtime_seconds",
        ),
        "genealogy_boxplot.svg": (
            "Genealogical diversity",
            "mean distinct root fraction",
            "mean_distinct_root_fraction",
        ),
        "reverse_ess_boxplot.svg": (
            "Reverse-weight efficiency",
            "mean reverse ESS / N",
            "mean_reverse_ess_over_n",
        ),
    }
    _write_text_compatible(
        report_dir / "estimate_boxplot.svg",
        _boxplot_svg(
            title="Rare-event probability estimates",
            y_label="estimated probability",
            groups=estimate_groups,
            reference=float(direct["empirical_probability"]),
            band=(float(direct["wilson_95_low"]), float(direct["wilson_95_high"])),
        ),
    )
    for filename, (title, y_label, field) in other_plots.items():
        groups = [
            {**group, "values": [float(row[field]) for row in group["rows"]]}
            for group in ordered_groups
        ]
        _write_text_compatible(
            report_dir / filename,
            _boxplot_svg(title=title, y_label=y_label, groups=groups),
        )
    acceptance_groups = []
    for group in ordered_groups:
        values = [
            float(row["mcmc_acceptance_rate"])
            for row in group["rows"]
            if row["mcmc_acceptance_rate"] is not None
        ]
        if values:
            acceptance_groups.append({**group, "values": values})
    if acceptance_groups:
        _write_text_compatible(
            report_dir / "mh_acceptance_boxplot.svg",
            _boxplot_svg(
                title="Metropolis-Hastings acceptance",
                y_label="accepted proposals / proposals",
                groups=acceptance_groups,
            ),
        )
    _write_text_compatible(
        report_dir / "error_vs_runtime.svg",
        _frontier_svg(
            title="Accuracy-cost frontier by wall time",
            x_label="median runtime (seconds)",
            grouped=grouped,
            metric="median_runtime_seconds",
        ),
    )
    _write_text_compatible(
        report_dir / "error_vs_decoder_compute.svg",
        _frontier_svg(
            title="Accuracy-cost frontier by decoder work",
            x_label="median decoder particle-steps",
            grouped=grouped,
            metric="median_decoder_particle_steps",
        ),
    )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if args.command == "run":
        destination = run_comparison_suite(
            suite, suite_path=args.suite.resolve(), output_root=args.output_root
        )
    else:
        destination = build_comparison_report(
            suite, suite_path=args.suite.resolve(), output_root=args.output_root
        )
    print(destination)


if __name__ == "__main__":
    main()
