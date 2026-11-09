"""Vary disjoint seed pairings while holding the saved runs and budget rule fixed."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compare_averaged_runs import CASES, EXPERIMENT, METHODS, crude_pool, matched_controls, source_rows, write_csv


PAIRING_SEED = 20261109
METRICS = ("positive", "within_factor_ten", "median_ratio", "relative_sd", "mean_log_relative_error")


def cohorts(root, trials):
    groups = source_rows(root)
    scores = crude_pool(root)
    rng = np.random.default_rng(PAIRING_SEED)
    for case, (threshold, tilt, n) in enumerate(CASES):
        small, large = (groups[threshold, tilt, size] for size in (n, 2 * n))
        ref = small[0]["ref"]
        small_log = np.array([row["log_p"] for row in small]) - np.log(ref)
        large_log = np.array([row["log_p"] for row in large]) - np.log(ref)
        small_time = np.array([row["seconds"] for row in small])
        large_time = np.array([row["seconds"] for row in large])
        small_mc = small_time / np.array([row["rate"] for row in small])
        large_mc = np.floor(large_time / np.array([row["rate"] for row in large])).astype(int)
        num_pairs = len(small) // 2
        block_size = len(scores) // (len(CASES) * num_pairs)
        starts = (case * num_pairs + np.arange(num_pairs)) * block_size
        event_prefix = np.r_[0, np.cumsum(scores >= threshold)]
        samples, max_gap, unique = [], 0., set()
        for _ in range(trials):
            pairs = np.sort(rng.permutation(len(small)).reshape(-1, 2), axis=1)
            pairs = pairs[np.argsort(pairs[:, 0])]
            matched = matched_controls(small_time[pairs].sum(axis=1), large_time)
            if matched is None:
                continue
            controls, gaps = matched
            max_gap = max(max_gap, float(gaps.max()))
            unique.add(pairs.tobytes())
            mc_pair = np.floor(small_mc[pairs].sum(axis=1)).astype(int)
            mc_large = large_mc[controls]
            if max(mc_pair.max(), mc_large.max()) > block_size:
                raise ValueError("Insufficient distinct saved MC scores for a matched budget")
            crude_pair = (event_prefix[starts + mc_pair] - event_prefix[starts]) / (mc_pair * ref)
            crude_large = (event_prefix[starts + mc_large] - event_prefix[starts]) / (mc_large * ref)
            pair_ratio = np.exp(np.logaddexp.reduce(small_log[pairs], axis=1) - np.log(2))
            samples.append((pair_ratio, np.exp(large_log[controls]), crude_pair, crude_large))
        if not samples:
            raise ValueError(f"No complete budget matches for {(threshold, tilt, n)}")
        yield dict(threshold=threshold, tilt=tilt, small_n=n, attempted=trials, accepted=len(samples),
                   unique_pairings=len(unique), max_budget_gap=max_gap, estimates_per_trial=num_pairs,
                   source_small_runs=len(small), source_large_runs=len(large)), np.asarray(samples)


def summarize(meta, ratios):
    values = dict(
        positive=np.count_nonzero(ratios > 0, axis=2),
        within_factor_ten=np.count_nonzero((ratios >= .1) & (ratios <= 10), axis=2),
        median_ratio=np.median(ratios, axis=2),
        relative_sd=np.std(ratios, axis=2, ddof=1),
        mean_log_relative_error=np.mean(np.log1p(np.abs(ratios - 1)) / np.log(10), axis=2),
    )
    for metric in METRICS:
        data = values[metric]
        if metric == "median_ratio":
            # Closeness to one on the multiplicative scale. Zeros are infinitely far away.
            with np.errstate(divide="ignore"):
                rank = -np.abs(np.log(data))
        else:
            rank = -data if metric in ("relative_sd", "mean_log_relative_error") else data
        ties = (rank[:, 0] == rank[:, 1]) | np.isclose(rank[:, 0], rank[:, 1], rtol=1e-12, atol=1e-15)
        pair_wins = (rank[:, 0] > rank[:, 1]) & ~ties
        large_wins = (rank[:, 1] > rank[:, 0]) & ~ties
        for method, column in zip(METHODS, data.T, strict=True):
            q05, median, q95 = np.quantile(column, [.05, .5, .95])
            yield dict(**meta, metric=metric, method=method, q05=q05, median=median, q95=q95,
                       smc_pair_wins_pct=100 * pair_wins.mean(), smc_tie_pct=100 * ties.mean(),
                       smc_large_wins_pct=100 * large_wins.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT / "artifacts/current")
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    rows = [row for meta, ratios in cohorts(args.root, args.trials) for row in summarize(meta, ratios)]
    output = args.root / "camera_ready_low_particle_budget/paired_runs_sensitivity.csv"
    write_csv(output, rows)
    print(output)


if __name__ == "__main__":
    main()
