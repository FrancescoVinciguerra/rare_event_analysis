from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


EXPORT_FIELDS = (
    "condition",
    "cache_file",
    "cache_run_index",
    "sample_id",
    "seed",
    "phi",
    "predicted_class",
    "safe_probability",
    "controversial_probability",
    "unsafe_probability",
    "exact_leak",
    "completion",
)


def _condition_from_cache_path(path: Path) -> str:
    name = path.name
    if not name.endswith(".json.gz"):
        raise ValueError(f"Expected a .json.gz cache, got {path}")
    cache_stem = name[: -len(".json.gz")]
    condition, separator, fingerprint = cache_stem.rpartition("_")
    if not separator or len(fingerprint) != 16:
        return cache_stem
    return condition


def read_cached_completions(
    cache_dir: str | Path,
    *,
    secret: str,
) -> list[dict[str, Any]]:
    """Read all cached direct rows without loading either neural model."""

    cache_root = Path(cache_dir)
    archives = sorted(cache_root.glob("*.json.gz"))
    if not archives:
        raise FileNotFoundError(f"No direct baseline archives found in {cache_root}")

    rows: list[dict[str, Any]] = []
    for archive in archives:
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        condition = _condition_from_cache_path(archive)
        for cache_run_index, run in enumerate(payload.get("runs", [])):
            run_seed = run.get("seed")
            for source_index, stored in enumerate(run.get("rows", [])):
                completion = str(stored.get("completion", ""))
                rows.append(
                    {
                        "condition": condition,
                        "cache_file": archive.name,
                        "cache_run_index": cache_run_index,
                        "sample_id": stored.get("sample_id", source_index),
                        "seed": stored.get("seed", run_seed),
                        "phi": stored.get("phi"),
                        "predicted_class": stored.get("predicted_class"),
                        "safe_probability": stored.get("safe_probability"),
                        "controversial_probability": stored.get(
                            "controversial_probability"
                        ),
                        "unsafe_probability": stored.get("unsafe_probability"),
                        "exact_leak": bool(secret and secret in completion),
                        "completion": completion,
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["condition"]),
            -float(row["phi"] if row["phi"] is not None else float("-inf")),
            int(row["sample_id"]),
        )
    )
    return rows


def write_exports(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write human-readable and structured exports plus one download ZIP."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    text_path = destination / "direct_completions_full.txt"
    jsonl_path = destination / "direct_completions_full.jsonl"
    csv_path = destination / "direct_completions_full.csv"

    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            handle.write("=" * 100 + "\n")
            handle.write(
                f"ROW {index} | condition={row['condition']} | "
                f"sample_id={row['sample_id']} | seed={row['seed']}\n"
            )
            handle.write(
                f"phi={row['phi']} | class={row['predicted_class']} | "
                f"exact_leak={row['exact_leak']}\n"
            )
            handle.write("-" * 100 + "\n")
            handle.write(str(row["completion"]))
            handle.write("\n\n")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    archive_base = destination / "direct_completions_export"
    archive_base.with_suffix(".zip").unlink(missing_ok=True)
    zip_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=destination,
            base_dir=".",
        )
    )
    return {
        "text": text_path,
        "jsonl": jsonl_path,
        "csv": csv_path,
        "zip": zip_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export full direct-baseline completions without model inference."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secret", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_cached_completions(args.cache_dir, secret=args.secret)
    paths = write_exports(rows, args.output_dir)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "exact_leaks": sum(bool(row["exact_leak"]) for row in rows),
                "conditions": sorted({str(row["condition"]) for row in rows}),
                "files": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
