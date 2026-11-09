"""Plot the saved paired-checkpoint pilot with the same uncertainty layout as the budget study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from combine_particle_budgets import uncertainty_figure, write_csv
from plot_particle_scaling import group_results, read_csv


TARGETS = ((.99, (.5, 1.)), (.995, (1., 1.5, 2.)))
SIZES = (24, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    root = parser.parse_args().results
    status = json.loads((root / "report_status.json").read_bytes())
    if not status["complete"] or status["completed_runs"] != 48:
        raise ValueError("Expected the completed 48-ladder pilot")
    comparisons = tuple((event, tilt) for event, tilts in TARGETS for tilt in tilts)
    groups, summary = group_results(read_csv(root / "runs.csv"), read_csv(root / "summary.csv"),
                                    comparisons, sizes=SIZES)
    lookup = {}
    for key, saved in summary.items():
        row = {field: float(saved[field]) for field in (
            "p_reference", "mean_crude_n", "mean_ratio", "median_ratio", "relative_standard_deviation",
        )}
        row["zero_count"] = int(saved["zero_count"])
        row["fraction_one_expected_crude_hit"] = row["p_reference"] * row["mean_crude_n"]
        lookup[key] = row
    for n in SIZES:
        cohorts = [{(r["seed"], r["replicate"], r["crude_pool_start"]) for r in groups[event, tilt, n]}
                   for event, tilt in comparisons]
        if any(cohort != cohorts[0] for cohort in cohorts[1:]):
            raise ValueError("Checkpoint cohorts differ within one particle count")
    panels = tuple(
        (f"q ≥ {event:g} · N={n}".replace(".", ","), tuple((event, tilt, (n,)) for tilt in tilts))
        for event, tilts in TARGETS for n in SIZES
    )
    numeric_groups = {
        key: [{field: float(r[field]) for field in ("smc_ratio", "crude_ratio", "event_particles", "crude_hits")}
              for r in rows]
        for key, rows in groups.items()
    }
    intervals = uncertainty_figure(root, numeric_groups, lookup, panels, positive_only=False, label_tilt=True)
    intervals.extend(uncertainty_figure(root, numeric_groups, lookup, panels, positive_only=True, label_tilt=True))
    write_csv(root / "rare_event_bootstrap_uncertainty.csv", intervals)
    for (event, tilt, n), rows in sorted(numeric_groups.items(), key=lambda entry: (entry[0][0], entry[0][2], entry[0][1])):
        smc = summary[event, tilt, n, "smc"]
        crude = summary[event, tilt, n, "crude"]
        count = len(rows)
        print(
            f"q={event:g} N={n} lambda={tilt:g} {float(smc['mean_smc_seconds']):.1f} s "
            f"MC n={float(smc['mean_crude_n']):.0f} "
            f"positive SMC/MC={count-int(smc['zero_count'])}/{count-int(crude['zero_count'])} "
            f"factor10 SMC/MC={smc['within_factor_ten']}/{crude['within_factor_ten']} "
            f"median event particles={np.median([r['event_particles'] for r in rows]):g}"
        )


if __name__ == "__main__":
    main()
