"""Reproducible experiment; run from any directory. See README.md."""
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm

from model import Config, P_EVENT, THRESHOLD, base_paths, run_smc, ideal_variances
from report import create_report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pilot(config, blocks, size, root_seed, config_id):
    us, cs, blocks_out = [], [], []
    start = time.perf_counter()
    for b in range(blocks):
        seed = [root_seed, 2, config_id, b]
        result = run_smc(np.random.default_rng(np.random.SeedSequence(seed)), size, config)
        u, c = result.contributions()
        us.append(u)
        cs.append(c)
        blocks_out.append({"config_id": config_id, "block": b, "seed": json.dumps(seed),
                           "mean_u": u.mean(), "mean_c": c.mean(),
                           "second_u": np.mean(u*u), "second_centered": np.mean((u-P_EVENT*c)**2),
                           "mean_w": np.exp(result.log_weight).mean()})
    u, c = np.concatenate(us), np.concatenate(cs)
    diagnostics = {"pilot_n": len(u), "pilot_seconds": time.perf_counter()-start,
                   "mean_u": float(u.mean()), "se_mean_u": float(u.std(ddof=1)/np.sqrt(len(u))),
                   "mean_c": float(c.mean()), "se_mean_c": float(c.std(ddof=1)/np.sqrt(len(c))),
                   "mean_w": float(np.mean([b["mean_w"] for b in blocks_out])),
                   "true_z": float(np.exp(config.tilt**2/2))}
    for name, squares, subtract in (("ordinary", u*u, P_EVENT**2), ("snis", (u-P_EVENT*c)**2, 0)):
        value = float(squares.mean() - subtract)
        se = float(squares.std(ddof=1)/np.sqrt(len(squares)))
        halfway = len(squares)//2
        half_gap = float(abs(squares[:halfway].mean()-squares[halfway:].mean())/value)
        max_share = float(squares.max()/squares.sum())
        top_count = max(1, len(squares)//1000)
        top_share = float(np.partition(squares, -top_count)[-top_count:].sum()/squares.sum())
        reliable = se/value < 0.15 and max_share < 0.02 and half_gap < 0.4
        diagnostics[name] = {"sigma2": value, "se": se, "relative_se": se/value,
                             "half_relative_gap": half_gap, "largest_share": max_share,
                             "top_0p1_percent_share": top_share, "stable_pilot": bool(reliable)}
    return diagnostics, blocks_out


def summary_row(config_id, config, method, values, times, costs, n, pilot_data, accepted):
    values = np.asarray(values)
    empirical = float(values.var(ddof=1))
    squared_errors = (values-P_EVENT)**2
    empirical_se = float(((values-values.mean())**2).std(ddof=1)/np.sqrt(len(values)))
    exact_mc_case = method == "crude" or config.tilt == 0
    pred = P_EVENT*(1-P_EVENT) if exact_mc_case else pilot_data[method]["sigma2"]
    pred_se = 0 if exact_mc_case else pilot_data[method]["se"]
    return {"config_id": config_id, "lambda": config.tilt if config else 0,
            "max_depth": config.max_depth if config else 0, "method": method,
            "n": n, "replicates": len(values), "mean": float(values.mean()),
            "bias": float(values.mean()-P_EVENT), "bias_mc_se": float(np.sqrt(empirical/len(values))),
            "variance": empirical, "variance_mc_se": empirical_se,
            "rmse": float(np.sqrt(squared_errors.mean())), "zero_frequency": float(np.mean(values == 0)),
            "median_seconds": float(np.median(times)), "mean_transitions": float(np.mean(costs)),
            "acceptance": accepted, "predicted_sigma2": pred, "predicted_variance": pred/n,
            "prediction_se": pred_se/n, "empirical_over_predicted": empirical/(pred/n),
            "stable_pilot": True if method == "crude" else pilot_data[method]["stable_pilot"],
            "variance_cost_vs_crude": empirical*np.mean(costs)/(P_EVENT*(1-P_EVENT)*16)}


def plot_results(out, summary, diagnostics, n):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = [r for r in summary if r["config_id"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, depth in zip(axes, (1, 4, 16)):
        for method, color, marker in (("ordinary", "#146B8C", "o"), ("snis", "#BA4A30", "s")):
            group = [r for r in rows if r["max_depth"] == depth and r["method"] == method]
            x = [r["lambda"] for r in group]
            ax.errorbar(x, [n*r["variance"] for r in group], yerr=[1.96*n*r["variance_mc_se"] for r in group],
                        color=color, marker=marker, label=f"{method}: replications", capsize=3)
            ax.plot(x, [r["predicted_sigma2"] for r in group], "--", color=color, label=f"{method}: path pilot")
            for r in group:
                if not r["stable_pilot"]:
                    ax.scatter(r["lambda"], r["predicted_sigma2"], marker="x", s=110, color="black", zorder=5)
        ax.axhline(P_EVENT*(1-P_EVENT), color="gray", linestyle=":", label="Crude MC: exact")
        ax.set(title=f"Uniform depth 1,...,{depth}", xlabel="Terminal tilt lambda", yscale="log", xticks=[1, 2, 3])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("CLT coefficient N × variance")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Actual RollBack histories: empirical variance and independent pilot\nBlack ×: pilot moment estimate is unstable")
    fig.tight_layout()
    fig.savefig(out / "variance_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for depth, style in ((1, ":"), (4, "--"), (16, "-")):
        for method, color, marker in (("ordinary", "#146B8C", "o"), ("snis", "#BA4A30", "s")):
            group = [r for r in rows if r["max_depth"] == depth and r["method"] == method]
            x = [r["lambda"] for r in group]
            opts = dict(color=color, linestyle=style, marker=marker, label=f"{method}, D={depth}")
            axes[0, 0].errorbar(x, [r["bias"]/P_EVENT for r in group], yerr=[1.96*r["bias_mc_se"]/P_EVENT for r in group], capsize=2, **opts)
            axes[0, 1].plot(x, [r["rmse"]/P_EVENT for r in group], **opts)
            axes[1, 0].plot(x, [r["zero_frequency"] for r in group], **opts)
            axes[1, 1].plot(x, [r["variance_cost_vs_crude"] for r in group], **opts)
    axes[0, 0].axhline(0, color="gray", linestyle=":")
    axes[0, 1].axhline(np.sqrt(P_EVENT*(1-P_EVENT)/n)/P_EVENT, color="gray", linestyle=":")
    axes[1, 0].axhline((1-P_EVENT)**n, color="gray", linestyle=":")
    axes[1, 1].axhline(1, color="gray", linestyle=":")
    for ax, title in zip(axes.flat, ("Relative bias (95% Monte Carlo error bars)", "Relative RMSE", "Fraction of zero estimates", "Variance × transition cost / crude MC")):
        ax.set(title=title, xlabel="Terminal tilt lambda", xticks=[1, 2, 3])
        ax.grid(alpha=.2)
    axes[1, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_and_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(0, 3, 151)
    for method, color in (("ordinary", "#146B8C"), ("snis", "#BA4A30")):
        ax.plot(grid, [ideal_variances(v)[method] for v in grid], color=color, label=f"Ideal independent tilted samples: {method}")
        group = [r for r in rows if r["max_depth"] == 16 and r["method"] == method]
        ax.plot([r["lambda"] for r in group], [n*r["variance"] for r in group], "o--", color=color, label=f"Actual RollBack D=16: {method}")
    ax.axhline(P_EVENT*(1-P_EVENT), color="gray", linestyle=":", label="Crude MC")
    ax.set(xlabel="Terminal tilt lambda", ylabel="CLT coefficient N × variance", yscale="log",
           title="Ideal tilted sampling is a different experiment")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "ideal_vs_actual.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--pilot-blocks", type=int, default=128)
    parser.add_argument("--pilot-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    configs = [Config(tilt=0, max_depth=16)] + [Config(tilt=tilt, max_depth=d) for d in (1, 4, 16) for tilt in (1, 2, 3)]
    metadata = {**vars(args), "output": str(out), "threshold": THRESHOLD, "true_probability": P_EVENT,
                "schedule": "linear", "depth_law": "uniform integers 1,...,max_depth", "resampling": False,
                "configs": [asdict(c) for c in configs], "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                "platform": platform.platform(), "seed_convention": "SeedSequence([root, stream, config_id, replicate]); stream=0 crude,1 SMC,2 pilot"}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    raw, summary, pilot_blocks, all_diagnostics = [], [], [], {}
    times, estimates = [], []
    for rep in range(args.replicates):
        seed = [args.seed, 0, 0, rep]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        now = time.perf_counter()
        paths = base_paths(rng, args.particles, 16)
        estimate = float(np.mean(paths[:, -1] >= THRESHOLD))
        seconds = time.perf_counter()-now
        estimates.append(estimate)
        times.append(seconds)
        raw.append({"config_id": -1, "replicate": rep, "seed": json.dumps(seed), "ordinary": estimate,
                    "snis": estimate, "seconds": seconds, "transitions": args.particles*16, "acceptance": 0})
    summary.append(summary_row(-1, None, "crude", estimates, times, [args.particles*16], args.particles, None, 0))
    print(f"Crude MC complete: {args.replicates} independent replications", flush=True)
    for cid, config in enumerate(configs):
        times, costs, accepts, values = [], [], [], {"ordinary": [], "snis": []}
        for rep in range(args.replicates):
            seed = [args.seed, 1, cid, rep]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            now = time.perf_counter()
            result = run_smc(rng, args.particles, config)
            estimates = result.estimates()
            seconds = time.perf_counter()-now
            acceptance = result.accepted/result.proposals
            times.append(seconds)
            costs.append(result.transitions)
            accepts.append(acceptance)
            for method, estimate in estimates.items():
                values[method].append(estimate)
            raw.append({"config_id": cid, "replicate": rep, "seed": json.dumps(seed), **estimates,
                        "seconds": seconds, "transitions": result.transitions, "acceptance": acceptance})
        diag, block_rows = pilot(config, args.pilot_blocks, args.pilot_size, args.seed, cid)
        all_diagnostics[str(cid)] = diag
        pilot_blocks.extend(block_rows)
        for method in values:
            summary.append(summary_row(cid, config, method, values[method], times, costs, args.particles, diag, float(np.mean(accepts))))
        # Checkpoint completed configurations, allowing inspection during a run.
        write_csv(out / "replicates.csv", raw)
        write_csv(out / "summary.csv", summary)
        write_csv(out / "pilot_blocks.csv", pilot_blocks)
        (out / "pilot_diagnostics.json").write_text(json.dumps(all_diagnostics, indent=2), encoding="utf-8")
        print(f"Config {cid}: lambda={config.tilt}, D={config.max_depth}, complete; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    plot_results(out, summary, all_diagnostics, args.particles)
    metadata["total_seconds"] = time.perf_counter()-start
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    create_report(out)
    print(f"Finished in {metadata['total_seconds']:.1f}s. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
