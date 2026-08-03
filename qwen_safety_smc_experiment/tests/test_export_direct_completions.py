from __future__ import annotations

import gzip
import json

from qwen_safety_smc_experiment.scripts.export_direct_completions import (
    read_cached_completions,
    write_export,
)


def test_export_writes_only_completion_and_phi_to_one_file(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "no_prefix_1234567890abcdef.json.gz"
    completion = "First line.\nSecond line with the complete response."
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
                        "completion": completion,
                    }
                ],
            }
        ],
    }
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    rows = read_cached_completions(cache_dir)
    output_dir = tmp_path / "exports"
    output_path = write_export(
        rows,
        output_dir / "direct_completions_with_phi.txt",
    )
    exported = output_path.read_text(encoding="utf-8")

    assert rows == [{"phi": 0.25, "completion": completion}]
    assert exported.startswith("PHI: 0.25\nCOMPLETION:\n")
    assert completion in exported
    assert "predicted_class" not in exported
    assert list(output_dir.iterdir()) == [output_path]
