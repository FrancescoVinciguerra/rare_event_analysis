"""Independent, prespecified-size follow-up of the null-tilt control.

Original 2000-run results are retained. This additional check uses stream 3.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
from model import Config, P_EVENT, run_smc


def main():
    root_seed, n, reps = 20260929, 2048, 4000
    out = Path(__file__).resolve().parent / "results"
    rows = []
    start = time.perf_counter()
    for rep in range(reps):
        seed = [root_seed, 3, 0, rep]
        result = run_smc(np.random.default_rng(np.random.SeedSequence(seed)), n, Config(tilt=0))
        estimates = result.estimates()
        assert estimates["ordinary"] == estimates["snis"]
        rows.append({"replicate": rep, "seed": json.dumps(seed), "estimate": estimates["ordinary"]})
    values = np.array([r["estimate"] for r in rows])
    p, q = P_EVENT, 1-P_EVENT
    theoretical_var = p*q/n
    fourth_central_moment = (3*(n*p*q)**2 + n*p*q*(1-6*p*q))/n**4
    exact_se_sample_variance = np.sqrt((fourth_central_moment - (reps-3)/(reps-1)*theoretical_var**2)/reps)
    summary = {"seed": root_seed, "stream": 3, "n": n, "replicates": reps,
               "mean": float(values.mean()), "variance": float(values.var(ddof=1)),
               "theoretical_variance": theoretical_var, "exact_se_sample_variance": float(exact_se_sample_variance),
               "variance_z_score": float((values.var(ddof=1)-theoretical_var)/exact_se_sample_variance),
               "zero_frequency": float(np.mean(values == 0)), "theoretical_zero_frequency": q**n,
               "seconds": time.perf_counter()-start}
    with (out / "zero_tilt_followup.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "zero_tilt_followup.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
