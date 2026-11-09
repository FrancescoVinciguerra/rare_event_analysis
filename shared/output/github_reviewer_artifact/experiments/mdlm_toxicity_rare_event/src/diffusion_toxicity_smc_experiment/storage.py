"""Persistent, append-only direct-generation and evaluation shards."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .configuration import canonical_json, fingerprint


GENERATION_REQUIRED_FIELDS = {
    "sample_id",
    "sample_index",
    "conditioning_mode",
    "shard_seed",
    "position_in_shard",
    "token_ids",
    "completion_token_ids",
    "decoded_text",
    "checkpoint_revision",
    "tokenizer_revision",
    "decoder_id",
    "generated_at_utc",
}


@dataclass(frozen=True)
class DirectInventory:
    rows: tuple[dict[str, Any], ...]
    sample_indices: set[int]
    sample_ids: set[str]

    @property
    def count(self) -> int:
        return len(self.rows)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def initialize_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(manifest):
            raise ValueError(
                f"manifest mismatch at {path}; use a different decoder/evaluator directory"
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, manifest)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {path}:{line_number}")
            rows.append(row)
    return rows


def scan_direct_generations(run_dir: Path, expected_decoder_id: str) -> DirectInventory:
    all_rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    indices: set[int] = set()
    generation_dir = run_dir / "generations"
    for shard in sorted(generation_dir.glob("shard_*.jsonl")):
        for row in read_jsonl(shard):
            missing = GENERATION_REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(f"{shard} is missing fields: {sorted(missing)}")
            if row["decoder_id"] != expected_decoder_id:
                raise ValueError(f"decoder_id mismatch in {shard}")
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r}")
            sample_ids.add(sample_id)
            if row["conditioning_mode"] != "unconditional":
                raise ValueError(f"non-unconditional sample in {shard}")
            sample_index = int(row["sample_index"])
            if sample_index in indices:
                raise ValueError(f"duplicate sample_index {sample_index}")
            indices.add(sample_index)
            all_rows.append(row)
    return DirectInventory(tuple(all_rows), indices, sample_ids)


def stable_sample_id(*, decoder_id: str, sample_index: int, shard_seed: int) -> str:
    return fingerprint(
        {
            "decoder_id": decoder_id,
            "sample_index": sample_index,
            "shard_seed": shard_seed,
        }
    )


def missing_indices(
    inventory: DirectInventory,
    *,
    target: int,
    task_index: int = 0,
    num_tasks: int = 1,
) -> list[int]:
    if target < 0 or num_tasks <= 0 or not 0 <= task_index < num_tasks:
        raise ValueError("invalid target or task partition")
    return [
        index
        for index in range(target)
        if index % num_tasks == task_index and index not in inventory.sample_indices
    ]
