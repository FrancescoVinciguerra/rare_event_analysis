"""Evaluate score >= 0.995 on the saved log(q) and logit ladders."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from analyze_relative_variance import analyze
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from plot_same_run_ladder import render

EXPERIMENT = Path(__file__).resolve().parents[2]
REPO = EXPERIMENT.parent
sys.path[:0] = [str(EXPERIMENT / "src"), str(REPO / "shared")]

from diffusion_toxicity_smc_experiment.low_budget import estimates
from diffusion_toxicity_smc_experiment.same_run_reuse import (
    mc_pool, native_clock_count, write_csv,
)
from rea_llms.statistics import wilson_interval


THRESHOLD = 0.995
SUFFIX = "_3_thresholds"
CHECKPOINT_FILE = f"checkpoint_runs{SUFFIX}.csv"


def load_json(path):
    return json.loads(path.read_bytes())


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def checkpoint_key(row):
    return tuple(int(row[key]) for key in ("seed", "level", "sweep"))


def held_out_reference(pool, studies):
    """Discard the union of the MC prefixes used by both ladders."""
    starts = sorted({block["pool_start"] for _, blocks, _ in studies for block in blocks.values()})
    block_size = len(pool) // 12
    if starts != [rank * block_size for rank in range(12)]:
        raise ValueError("Expected twelve disjoint MC seed blocks")
    ends = {start: start for start in starts}
    for rows, blocks, _ in studies:
        sizes = {int(row["seed"]): 0 for row in rows}
        for row in rows:
            seed = int(row["seed"])
            count = int(row["crude_n"])
            if count != native_clock_count(float(row["smc_runtime_seconds_cumulative"]),
                                           float(row["calibration_seconds_per_path"])):
                raise ValueError("MC count does not match calibrated wall-clock budget")
            sizes[seed] = max(sizes[seed], count)
        if len(sizes) != len(starts):
            raise ValueError("Unexpected number of MC seed blocks")
        for expected_start, (seed, count) in zip(starts, sorted(sizes.items())):
            block = blocks[str(seed)]
            start = block["pool_start"]
            size = block["fixed_block_size"]
            if start != expected_start or size != block_size:
                raise ValueError("Unexpected MC block assignment")
            if not 0 < count <= size or len(block["scores"]) != count:
                raise ValueError("Saved MC prefix has unexpected length")
            if not np.array_equal(pool[start:start + count], block["scores"]):
                raise ValueError("MC prefix differs from source pool")
            ends[start] = max(ends[start], start + count)

    slices = list(zip([0, *[ends[start] for start in starts]], [*starts, len(pool)]))
    if any(start > end for start, end in slices):
        raise ValueError("MC prefixes overlap")
    samples = np.concatenate([pool[start:end] for start, end in slices])
    hits = int((samples >= THRESHOLD).sum())
    lower, upper = wilson_interval(hits, len(samples))
    return dict(threshold=THRESHOLD, hits=hits, samples=len(samples),
                p_reference=hits / len(samples), wilson_95_low=lower, wilson_95_high=upper,
                retained_index_ranges=slices, excluded_matched_MC_samples=len(pool) - len(samples))


def extend_rows(root, rows, blocks, policy, reference):
    lookup = {(checkpoint_key(row), float(row["threshold"])): row for row in rows}
    if len(lookup) != 2 * 12 * 65:
        raise ValueError("Expected two original thresholds on twelve complete ladders")
    added = []
    for source in policy["SMC_sources"]:
        raw = (root / source["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Population differs from the original report source")
        population = json.loads(raw)
        seed = int(population["summary"]["seed"])
        for cp in population["population_checkpoints"]:
            key = (seed, cp["level"], cp["sweep"])
            original = {a: lookup[key, a] for a in (0.99, 0.999)}
            row = original[0.99]
            prefix = np.asarray(blocks[str(seed)]["scores"][:int(row["crude_n"])])
            recomputed = {}
            for threshold in (0.99, THRESHOLD, 0.999):
                result = estimates(cp["normalized_log_weights"], cp["rewards"], cp["log_z_hat"],
                                   cp["scores"], threshold, cp["lambda"])
                hits = int((prefix >= threshold).sum())
                recomputed[threshold] = dict(result, crude_event_particles=hits)
                if threshold in original:
                    check = original[threshold]
                    for field, value in result.items():
                        if not math.isclose(value, float(check[field]), rel_tol=1e-12):
                            raise ValueError(f"Failed to reproduce {key}, {threshold}, {field}")
                    if hits != int(check["crude_event_particles"]) or hits / len(prefix) != float(check["crude"]):
                        raise ValueError("Failed to reproduce original crude estimates")
                else:
                    event_count = result["event_particles"]
                    added.append(dict(row, threshold=threshold, p_reference=reference["p_reference"],
                                      eligible_population=event_count > 0, event_fraction=event_count / int(row["N"]),
                                      **result, crude=hits / len(prefix), crude_event_particles=hits, crude_zero=hits == 0))
            for method in ("smc", "snis", "event_particles", "crude_event_particles"):
                if not recomputed[0.999][method] <= recomputed[THRESHOLD][method] <= recomputed[0.99][method]:
                    raise ValueError(f"Nested event ordering failed for {key}, {method}")
    if len(added) != 12 * 65 or len({checkpoint_key(r) for r in added}) != len(added):
        raise ValueError("Missing or duplicated new estimates")
    return sorted([*rows, *added], key=lambda r: (checkpoint_key(r), float(r["threshold"])))


def main():
    artifacts = EXPERIMENT / "artifacts" / "current"
    old = artifacts / "camera_ready_same_run_ladder"
    new = artifacts / "camera_ready_same_run_logit"
    studies = [
        (load_rows(root / "checkpoint_runs.csv"), load_json(root / "saved_MC_blocks.json"),
         load_json(root / "comparison_policy.json"))
        for root in (old, new)
    ]
    pool_root = artifacts / "camera_ready_gradual_budget"
    study = load_json(pool_root / "manifest.json")["study"]
    pool, sources = mc_pool(pool_root, study)
    for _, _, policy in studies:
        frozen = [dict(source, file=Path(source["file"]).as_posix()) for source in policy["MC_sources"]]
        actual = [dict(source, file=Path(source["file"]).as_posix()) for source in sources]
        if actual != frozen or len(pool) != policy["MC_pool_samples"]:
            raise ValueError("MC pool does not match the original report")
    if not np.isfinite(pool).all() or np.any((pool < 0) | (pool > 1)):
        raise ValueError("Invalid toxicity scores in MC pool")
    reference = held_out_reference(pool, studies)
    extended = [extend_rows(root, *study, reference) for root, study in zip((old, new), studies)]
    for root, rows in zip((old, new), extended):
        write_csv(root / CHECKPOINT_FILE, rows)
    hits, samples = (format(reference[key], ",").replace(",", ".") for key in ("hits", "samples"))
    note = (f'Riferimenti stimati: score 0,99 = 444/200.000; 0,995 = {hits}/{samples} '
            '(campioni fuori dalle baseline); 0,999 = 26/200.000.')
    (new / f"reference{SUFFIX}.json").write_text(
        json.dumps(dict(reference, source_pool=str(pool_root.relative_to(REPO)),
                        original_reference="shared/output/github_reviewer_artifact/data/final_analysis/direct_reference.csv",
                        original_checkpoint_rows_verified=2 * 2 * 12 * 65,
                        new_simulations=False), indent=2) + "\n", encoding="utf-8")
    render(new, checkpoint_file=CHECKPOINT_FILE, suffix=SUFFIX, reference_note=note)
    analyze(old, new, new, statistic="std", checkpoint_file=CHECKPOINT_FILE, suffix=SUFFIX, reference_note=note)
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
