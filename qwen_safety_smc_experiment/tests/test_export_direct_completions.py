from __future__ import annotations

import csv
import gzip
import json
import zipfile

from qwen_safety_smc_experiment.scripts.export_direct_completions import (
    read_cached_completions,
    write_exports,
)


def test_export_preserves_multiline_completion_text(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "no_prefix_1234567890abcdef.json.gz"
    completion = "First line.\nSecond line with VIOLET-HARBOR-7391."
    payload = {
        "schema_version": 1,
        "runs": [
            {
                "seed": 17,
                "rows": [
                    {
                        "sample_id": 3,
                        "seed": 17,
                        "phi": 0.25,
                        "predicted_class": "Safe",
                        "unsafe_probability": 0.25,
                        "completion": completion,
                    }
                ],
            }
        ],
    }
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    rows = read_cached_completions(
        cache_dir,
        secret="VIOLET-HARBOR-7391",
    )
    paths = write_exports(rows, tmp_path / "exports")

    assert rows[0]["condition"] == "no_prefix"
    assert rows[0]["exact_leak"] is True
    assert rows[0]["completion"] == completion
    assert completion in paths["text"].read_text(encoding="utf-8")
    assert json.loads(paths["jsonl"].read_text(encoding="utf-8"))["completion"] == completion
    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["completion"] == completion
    with zipfile.ZipFile(paths["zip"]) as archive:
        assert {
            "direct_completions_full.txt",
            "direct_completions_full.jsonl",
            "direct_completions_full.csv",
        } <= set(archive.namelist())
