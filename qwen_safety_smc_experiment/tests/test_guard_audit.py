from __future__ import annotations

import numpy as np
import pandas as pd

from qwen_safety_smc_experiment.guard_audit import prepare_guard_audit_data


def test_stratified_guard_audit_is_deterministic_and_retains_high_scores() -> None:
    direct = pd.DataFrame(
        {
            "sample_id": np.arange(40),
            "completion": [f"Yes direct {i}" for i in range(40)],
            "completion_token_ids": [[i, i + 1] for i in range(40)],
            "phi": np.linspace(0.0, 1.0, 40),
        }
    )
    smc = pd.DataFrame(
        {
            "full_completion": [f"Yes smc {i}" for i in range(40)],
            "completion_token_ids": [[i, i + 1] for i in range(40)],
            "phi": np.linspace(0.02, 0.98, 40),
            "normalized_weight": np.full(40, 1 / 40),
        }
    )
    kwargs = dict(
        assistant_prefix="Yes ",
        top_per_source=6,
        num_phi_bins=4,
        samples_per_bin_per_source=2,
        selection_seed=7,
    )
    full, first = prepare_guard_audit_data(direct, smc, **kwargs)
    _, second = prepare_guard_audit_data(direct, smc, **kwargs)

    assert len(full) == 80
    assert len(first) == 28
    assert first[["source", "sample_id"]].equals(second[["source", "sample_id"]])
    for source, frame in first.groupby("source"):
        assert (frame["selection_reason"] == "top_phi").sum() == 6
        expected = full[full.source == source].nlargest(6, "stream_0_6b_phi")
        assert set(expected.sample_id) <= set(frame.sample_id)
    assert first.completion_without_forced_prefix.str.startswith("Yes ").sum() == 0
