from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Sequence


def read_cached_completions(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Read completion text and phi from every direct cache without inference."""

    cache_root = Path(cache_dir)
    archives = sorted(cache_root.glob("*.json.gz"))
    if not archives:
        raise FileNotFoundError(f"No direct baseline archives found in {cache_root}")

    rows: list[dict[str, Any]] = []
    for archive in archives:
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        for run in payload.get("runs", []):
            for stored in run.get("rows", []):
                rows.append(
                    {
                        "phi": float(stored["phi"]),
                        "completion": str(stored.get("completion", "")),
                    }
                )
    rows.sort(key=lambda row: -float(row["phi"]))
    return rows


def write_export(
    rows: Sequence[dict[str, Any]],
    output_file: str | Path,
) -> Path:
    """Write exactly one human-readable file containing completion and phi."""

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"PHI: {row['phi']}\n")
            handle.write("COMPLETION:\n")
            handle.write(str(row["completion"]))
            handle.write("\n" + "=" * 100 + "\n\n")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export cached direct completions and phi to one text file."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_cached_completions(args.cache_dir)
    output_path = write_export(rows, args.output_file)
    print(json.dumps({"rows": len(rows), "output_file": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
