"""Score persistent IID unconditional generations without regenerating them."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .configuration import (
    canonical_json,
    decoder_id,
    evaluator_id,
    evaluator_manifest,
    load_yaml,
)
from .evaluation import (
    ToxicityEvaluator,
    direct_unconditional_summary,
    identical_classifier_inputs,
    unconditional_mdlm_prefix_classifier_text,
    unconditional_trimmed_classifier_text,
)
from .generate_direct import direct_run_directory
from .runtime import load_decoder_tokenizer, load_evaluator_runtime
from .storage import (
    atomic_write_json,
    atomic_write_jsonl,
    initialize_manifest,
    read_jsonl,
    scan_direct_generations,
)


def latest_direct_summary_path(eval_dir: Path) -> Path | None:
    candidates = list((eval_dir / "summaries").glob("summary_*.json"))
    legacy = eval_dir / "summary.json"
    if legacy.exists():
        candidates.append(legacy)
    if not candidates:
        return None

    def evaluated_count(path: Path) -> int:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload.get("evaluated_samples", -1))

    return max(candidates, key=lambda path: (evaluated_count(path), str(path)))


def _write_summary_snapshot(eval_dir: Path, summary: dict[str, Any]) -> Path:
    count = int(summary["evaluated_samples"])
    snapshot = eval_dir / "summaries" / f"summary_{count:08d}.json"
    if snapshot.exists():
        existing = json.loads(snapshot.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(summary):
            raise ValueError(f"existing direct summary snapshot is incompatible: {snapshot}")
    else:
        atomic_write_json(snapshot, summary)
    legacy = eval_dir / "summary.json"
    if not legacy.exists():
        atomic_write_json(legacy, summary)
    return snapshot


def score_direct_run(
    config: dict[str, Any],
    *,
    direct_run: Path | None,
    allow_network: bool,
) -> Path:
    identifier = decoder_id(config)
    run_dir = direct_run or direct_run_directory(config)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing direct manifest: {manifest_path}")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("decoder_id") != identifier:
        raise ValueError("direct run decoder_id does not match evaluator configuration")
    if stored.get("conditioning_mode") != "unconditional":
        raise ValueError("direct run is not unconditional")
    inventory = scan_direct_generations(run_dir, identifier)

    eval_id = evaluator_id(config)
    eval_dir = run_dir / "evaluations" / eval_id
    initialize_manifest(
        eval_dir,
        {
            "evaluator_id": eval_id,
            "decoder_id": identifier,
            "conditioning_mode": "unconditional",
            "evaluator_manifest": evaluator_manifest(config),
            "storage_schema": "unconditional_direct_evaluation_shards_v1",
        },
    )
    summary_path = latest_direct_summary_path(eval_dir)
    if summary_path is not None:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("decoder_id") == identifier
            and existing.get("evaluator_id") == eval_id
            and int(existing.get("evaluated_samples", -1)) == inventory.count
        ):
            return eval_dir

    evaluator_runtime = load_evaluator_runtime(config, local_files_only=not allow_network)
    decoder_tokenizer = load_decoder_tokenizer(config, local_files_only=not allow_network)
    numerical_batching = config["evaluation"]["numerical_batching"]
    evaluator = ToxicityEvaluator(
        evaluator_runtime.classifier,
        evaluator_runtime.classifier_tokenizer,
        evaluator_runtime.device,
        positive_label_index=int(config["evaluation"]["classifier"]["positive_label_index"]),
        batch_size=int(config["evaluation"].get("batch_size", 16)),
        padding=str(numerical_batching["padding"]),
        pad_final_batch_to_batch_size=bool(
            numerical_batching["pad_final_batch_to_batch_size"]
        ),
    )
    reward_cfg = config["evaluation"]["reward"]
    event_cfg = config["evaluation"]["event"]
    scoring_started = time.perf_counter()
    generation_runtime = 0.0
    generation_nfe = 0

    for generation_shard in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        score_path = eval_dir / generation_shard.name.replace("shard_", "scores_", 1)
        generation_rows = read_jsonl(generation_shard)
        if generation_rows:
            generation_runtime += float(
                generation_rows[0].get("generation_runtime_seconds_for_shard", 0.0)
            )
            generation_nfe += int(generation_rows[0].get("nfe_for_shard", 0))
        if score_path.exists():
            existing = read_jsonl(score_path)
            if [row.get("sample_id") for row in existing] != [
                row.get("sample_id") for row in generation_rows
            ]:
                raise ValueError(f"existing score shard is incompatible: {score_path}")
            continue

        reward_texts = []
        event_texts = []
        event_token_ids = []
        for row in generation_rows:
            reward_ids = row["token_ids"][: int(reward_cfg["completion_tokens"])]
            reward_texts.append(
                decoder_tokenizer.decode(reward_ids, skip_special_tokens=True)
            )
            if event_cfg["text_scope"] == "first_generated_mdlm_tokens_only":
                event_text, trimmed_ids = unconditional_mdlm_prefix_classifier_text(
                    token_ids=row["token_ids"],
                    decoder_tokenizer=decoder_tokenizer,
                    completion_tokens=int(event_cfg["completion_tokens"]),
                )
            else:
                event_text, trimmed_ids = unconditional_trimmed_classifier_text(
                    decoded_text=str(row["decoded_text"]),
                    trimming_tokenizer=evaluator_runtime.completion_trimming_tokenizer,
                    completion_tokens=int(event_cfg["completion_tokens"]),
                )
            event_texts.append(event_text)
            event_token_ids.append(trimmed_ids)
        reward_scores = evaluator.score_texts(
            reward_texts, max_length=int(reward_cfg["classifier_max_length"])
        )
        if identical_classifier_inputs(
            reward_texts,
            first_max_length=int(reward_cfg["classifier_max_length"]),
            second_texts=event_texts,
            second_max_length=int(event_cfg["classifier_max_length"]),
        ):
            event_scores = reward_scores
        else:
            event_scores = evaluator.score_texts(
                event_texts, max_length=int(event_cfg["classifier_max_length"])
            )
        output_rows = []
        for generation, reward_row, event_row, event_text, trimmed_ids in zip(
            generation_rows,
            reward_scores,
            event_scores,
            event_texts,
            event_token_ids,
        ):
            output_rows.append(
                {
                    "sample_id": generation["sample_id"],
                    "conditioning_mode": "unconditional",
                    "decoder_id": identifier,
                    "evaluator_id": eval_id,
                    "reward": reward_row["reward"],
                    "reward_logits": reward_row["logits"],
                    "reward_toxic_probability": reward_row["toxic_probability"],
                    "event_logits": event_row["logits"],
                    "event_toxic_probability": event_row["toxic_probability"],
                    "is_event": event_row["is_event"],
                    "event_classifier_text": event_text,
                    "event_completion_token_ids": trimmed_ids,
                    "event_token_id_space": (
                        "mdlm_decoder"
                        if event_cfg["text_scope"]
                        == "first_generated_mdlm_tokens_only"
                        else "roberta_large"
                    ),
                }
            )
        atomic_write_jsonl(score_path, output_rows)

    all_scores = []
    seen: set[str] = set()
    for score_path in sorted(eval_dir.glob("scores_*.jsonl")):
        for row in read_jsonl(score_path):
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate evaluated sample_id {row['sample_id']}")
            seen.add(row["sample_id"])
            all_scores.append(row)
    summary = direct_unconditional_summary(all_scores)
    summary.update(
        {
            "decoder_id": identifier,
            "evaluator_id": eval_id,
            "evaluated_samples": len(all_scores),
            "generation_runtime_seconds": generation_runtime,
            "generation_nfe": generation_nfe,
            "scoring_runtime_seconds": time.perf_counter() - scoring_started,
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(evaluator_runtime.device))
                if evaluator_runtime.device.type == "cuda"
                else 0
            ),
        }
    )
    _write_summary_snapshot(eval_dir, summary)
    return eval_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    direct_run = args.direct_run
    if direct_run is None and args.output_root is not None:
        direct_run = direct_run_directory(config, args.output_root)
    print(
        score_direct_run(
            config,
            direct_run=direct_run,
            allow_network=args.allow_network,
        )
    )
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_allocated_bytes={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
