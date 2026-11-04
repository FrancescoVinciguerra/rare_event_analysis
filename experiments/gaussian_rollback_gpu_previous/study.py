"""Reproducible GPU benchmark. Embedded without local imports in Colab notebook."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from gpu_model import (a, P_A, T, D, DTYPE, DEVICE, GPUConfig, require_gpu,
                       generator, gaussian_paths, rollback, systematic,
                       run_smc_population, crude_paths, crude_direct, measured,
                       make_seed, warmup)

OUT = Path("rollback_gpu_results")
ROOT_SEED = 20260929
N = 128
PILOT_N = 64
R = 4
TILTS = [0.5, 1.0, 1.5, 2.0]
PILOT_REPS = 200
REPS = 1200
DIRECT_REPS = 600


def check_algorithm():
    """Check distribution, invariant kernel, exact branches, normalizer and estimates."""
    device = require_gpu()
    assert T == D == 16 and abs(a - norm.isf(P_A)) < 1e-14
    paths = gaussian_paths(131072, generator(make_seed(1, 0, 0)))
    mean, second = paths[:, -1].mean().item(), paths[:, -1].var(unbiased=True).item()
    assert abs(mean) < .012 and abs(second-1) < .018, (mean, second)
    assert torch.all(paths[:, 0] == 0)
    assert abs((paths[:, 1:]-paths[:, :-1]).var().item()-1/T) < .001

    n = 131072
    origin = gaussian_paths(n, generator(make_seed(1, 1, 0)))
    beta = 1.5
    # Exact tilted reference is used ONLY in this diagnostic, never as sampler.
    tilted = origin + torch.arange(T+1, device=DEVICE, dtype=DTYPE)[None, :] * (beta/T)
    moved, depths, accepted = rollback(tilted, beta, generator(make_seed(1, 2, 0)))
    counts = torch.bincount(depths, minlength=D+1)[1:].cpu().numpy()
    expected = n/D
    assert counts.shape == (16,) and np.max(abs(counts-expected))/expected < .055
    cut = T-depths
    cols = torch.arange(T+1, device=DEVICE)[None, :]
    assert torch.all(torch.where(cols <= cut[:, None], moved == tilted, True))
    tilt_mean = moved[:, -1].mean().item()
    tilt_var = moved[:, -1].var(unbiased=True).item()
    assert abs(tilt_mean-beta) < .025 and abs(tilt_var-1) < .025

    # Same deterministic systematic selection as NumPy/CPU reference.
    weights = np.array([.05, .35, .10, .50], dtype=np.float32)
    log_weights = torch.tensor(np.log(weights), device=DEVICE)
    offset = torch.tensor(.37, device=DEVICE)
    gpu_parent = systematic(log_weights, generator(23), offset).cpu().numpy()
    cpu_parent = np.searchsorted(np.cumsum(weights), (np.arange(4)+.37)/4, side="left")
    np.testing.assert_array_equal(gpu_parent, cpu_parent)

    branch = {}
    for label, config in (("none", GPUConfig(128, 4, 0.0)),
                          ("selected", GPUConfig(128, 2, 4.0)),
                          ("final", GPUConfig(128, 1, 4.0))):
        result = run_smc_population(config, make_seed(2, len(branch), 0), debug=True)
        trace = result["history"]
        assert len(trace) == config.levels
        for h in trace:
            log_c = torch.logsumexp(h["old_log_w"] + config.tilt/config.levels*h["old_x"], 0)
            assert abs((h["log_c"]-log_c).item()) < 2e-6
            assert abs((h["log_z"]-h["old_log_z"]-log_c).item()) < 2e-6
            assert h["resampled"] == (h["level"] < config.levels and h["ess"].item() < .5*config.n)
            pre_log_w = h["old_log_w"] + config.tilt/config.levels*h["old_x"]-log_c
            assert abs(result["ess_by_level"][h["level"]-1] - torch.exp(-torch.logsumexp(2*pre_log_w, 0)).item()/config.n) < 2e-6
            assert abs(result["max_weight_by_level"][h["level"]-1] - pre_log_w.max().exp().item()) < 2e-6
        assert not trace[-1]["resampled"]
        if label == "none":
            assert not result["resampled_levels"]
            assert abs(result["z_hat"]-1) < 1e-5
            assert result["distinct_root_fraction"] == 1.0
        if label == "selected":
            assert result["resampled_levels"]
            assert result["distinct_root_fraction"] < 1.0
        if label == "final":
            assert trace[-1]["ess"].item() < .5*config.n
        terminal = result["paths"][:, -1]
        weights_terminal = result["log_weights"].exp()
        assert abs(weights_terminal.sum().item()-1) < 2e-6
        corrected = weights_terminal*torch.exp(-config.tilt*terminal)
        numerator = (corrected*(terminal >= a)).sum().item()
        denominator = corrected.sum().item()
        assert abs(result["smc"]-result["z_hat"]*numerator) < 1e-5
        assert abs(result["snis"]-numerator/denominator) < 1e-6
        reverse_weights = corrected/denominator
        assert abs(result["terminal_ess_fraction"] - 1/(config.n*weights_terminal.square().sum().item())) < 2e-6
        assert abs(result["terminal_max_weight"] - weights_terminal.max().item()) < 2e-6
        assert abs(result["reverse_ess_fraction"] - 1/(config.n*reverse_weights.square().sum().item())) < 2e-6
        assert abs(result["reverse_max_weight"] - reverse_weights.max().item()) < 2e-6
        assert result["distinct_roots"] == torch.unique(result["roots"]).numel()
        branch[label] = {"ess": [float(h["ess"].item()) for h in trace],
                         "resampled_levels": result["resampled_levels"],
                         "z_hat": result["z_hat"],
                         "used_transitions": result["used_transitions"]}
    device.update({"x0_mean_test": mean, "x0_var_test": second,
                   "tilted_kernel_mean_test": tilt_mean, "tilted_kernel_var_test": tilt_var,
                   "depth_min_count": int(counts.min()), "depth_max_count": int(counts.max()),
                   "branch_tests": branch})
    return device


def records_for(method, count, reps, stream, config_id=0, tilt=0.0):
    records = []
    for rep in range(reps):
        seed = make_seed(stream, config_id, rep, ROOT_SEED)
        if method == "smc":
            value, seconds = measured(run_smc_population, GPUConfig(count, R, tilt), seed)
            common = {"lambda": tilt, "n": count, "R": R, "run": rep, "seed": seed,
                      "seconds": seconds, "z_hat": value["z_hat"],
                      "reverse_mass": value["reverse_mass"],
                      "resampling_count": value["resampling_count"],
                      "resampled_levels": ",".join(map(str, value["resampled_levels"])),
                      "ess_by_level": ",".join(f"{e:.6g}" for e in value["ess_by_level"]),
                      "max_weight_by_level": ",".join(f"{e:.6g}" for e in value["max_weight_by_level"]),
                      "terminal_ess_fraction": value["terminal_ess_fraction"],
                      "terminal_max_weight": value["terminal_max_weight"],
                      "reverse_ess_fraction": value["reverse_ess_fraction"],
                      "reverse_max_weight": value["reverse_max_weight"],
                      "distinct_root_fraction": value["distinct_root_fraction"],
                      "used_transitions": value["used_transitions"],
                      "gaussian_variates_drawn": value["gaussian_variates_drawn"]}
            for estimator in ("smc", "snis"):
                records.append({"method": estimator, "estimate": value[estimator], **common})
        else:
            fn = crude_paths if method == "crude_paths" else crude_direct
            value, seconds = measured(fn, count, seed)
            records.append({"method": method, "estimate": value["estimate"],
                            "lambda": tilt, "n": count, "R": 0, "run": rep, "seed": seed,
                            "seconds": seconds, "z_hat": np.nan, "reverse_mass": np.nan,
                            "resampling_count": 0, "resampled_levels": "",
                            "ess_by_level": "",
                            "max_weight_by_level": "",
                            "terminal_ess_fraction": np.nan,
                            "terminal_max_weight": np.nan,
                            "reverse_ess_fraction": np.nan,
                            "reverse_max_weight": np.nan,
                            "distinct_root_fraction": np.nan,
                            "used_transitions": value["used_transitions"],
                            "gaussian_variates_drawn": value["gaussian_variates_drawn"]})
    return records


def median_latency(fn, count, stream, config_id, reps=64):
    times = []
    for rep in range(reps):
        _, dt = measured(fn, count, make_seed(stream, config_id, rep, ROOT_SEED))
        times.append(dt)
    return float(np.median(times))


def match_count(target_seconds, fn, stream, config_id, initial, max_count):
    count = initial
    for iteration in range(7):
        duration = median_latency(fn, count, stream+iteration, config_id)
        ratio = np.clip(target_seconds/max(duration, 1e-9), .3, 3.0)
        updated = int(np.clip(round(count*ratio/256)*256, 256, max_count))
        if abs(updated-count) <= 256:
            break
        count = updated
    return count, median_latency(fn, count, stream+20, config_id, reps=96)


def summarize(frame):
    rows = []
    keys = ["method", "lambda", "n", "R"]
    for key, group in frame.groupby(keys, sort=True):
        vals = group.estimate.to_numpy(dtype=float)
        errs = vals-P_A
        sq = errs*errs
        b = len(vals)
        rmse = float(np.sqrt(sq.mean()))
        bias = float(errs.mean())
        variance = float(vals.var(ddof=1))
        # Independent whole-run uncertainty, not a no-resampling CLT.
        se_rmse = float(sq.std(ddof=1)/(2*np.sqrt(b)*rmse)) if rmse > 0 else np.nan
        rows.append(dict(zip(keys, key)) | {"reps": b, "bias": bias,
                    "bias_se": float(vals.std(ddof=1)/np.sqrt(b)),
                    "variance": variance, "rmse": rmse, "rmse_se": se_rmse,
                    "zero_fraction": float(np.mean(vals == 0)),
                    "median_ms": float(group.seconds.median()*1000),
                    "mean_ms": float(group.seconds.mean()*1000),
                    "median_transitions": float(group.used_transitions.median()),
                    "median_gaussian_draws": float(group.gaussian_variates_drawn.median()),
                    "resampling_run_fraction": float(np.mean(group.resampling_count > 0))})
    return pd.DataFrame(rows)


def plot_results(summary, out):
    smc = summary[summary.method.isin(["smc", "snis"])].sort_values("lambda")
    fixed = summary[(summary.method == "crude_paths") & (summary.n == N)]
    matched = summary[(summary.method == "crude_paths") & (summary.n != N)]
    direct = summary[summary.method == "crude_direct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for method, label in (("smc", r"$\widehat p^{SMC}_{A,R}$"),
                          ("snis", r"$\widehat p^{SNIS}_{A,R}$")):
        f = smc[smc.method == method]
        axes[0].errorbar(f["lambda"], f.rmse, yerr=1.96*f.rmse_se, marker="o", capsize=2, label=label)
        axes[1].errorbar(f["lambda"], f.rmse, yerr=1.96*f.rmse_se, marker="o", capsize=2, label=label)
    if not fixed.empty:
        axes[0].axhline(fixed.iloc[0].rmse, color="black", linestyle="--", label=f"crude paths, N={N}")
    axes[1].errorbar(matched["lambda"], matched.rmse, yerr=1.96*matched.rmse_se,
                     marker="s", capsize=2, label="crude paths, matched GPU time")
    axes[1].errorbar(direct["lambda"], direct.rmse, yerr=1.96*direct.rmse_se,
                     marker="^", capsize=2, label="direct Gaussian, matched GPU time")
    axes[0].set_title("Same number of trajectories")
    axes[1].set_title("Same GPU wall-clock budget")
    for ax in axes:
        ax.set_xlabel(r"terminal tilt $\lambda=\beta_R$")
        ax.set_ylabel("RMSE vs exact $p_A=0.001$")
        ax.set_yscale("log")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "rmse_comparison.png", dpi=180)
    plt.close(fig)


def diagnostic_tables(frame):
    """One record per level/run and empirical distributions across complete runs."""
    smc = frame[frame.method == "snis"].copy()
    records = []
    for row in smc.to_dict("records"):
        pre_ess = [float(x) for x in row["ess_by_level"].split(",")]
        pre_max = [float(x) for x in row["max_weight_by_level"].split(",")]
        resampled = {int(x) for x in str(row["resampled_levels"]).split(",") if x and x != "nan"}
        for k in range(1, R+1):
            records.append({"lambda": row["lambda"], "run": row["run"], "level": k,
                            "pre_ess_fraction": pre_ess[k-1],
                            "pre_max_weight": pre_max[k-1],
                            "resampled": int(k in resampled)})
    levels = pd.DataFrame(records)
    long = []
    for row in levels.to_dict("records"):
        for metric in ("pre_ess_fraction", "pre_max_weight", "resampled"):
            long.append({"lambda": row["lambda"], "run": row["run"],
                         "stage": f"before_resampling_k{row['level']}", "metric": metric,
                         "value": row[metric]})
    for row in smc.to_dict("records"):
        for metric, stage in (("terminal_ess_fraction", "terminal"),
                              ("terminal_max_weight", "terminal"),
                              ("reverse_ess_fraction", "reverse"),
                              ("reverse_max_weight", "reverse"),
                              ("distinct_root_fraction", "genealogy")):
            long.append({"lambda": row["lambda"], "run": row["run"],
                         "stage": stage, "metric": metric, "value": row[metric]})
    distribution = pd.DataFrame(long)
    quantiles = distribution.groupby(["lambda", "stage", "metric"]).value.agg(
        n="count", mean="mean", sd="std", minimum="min",
        p05=lambda x: x.quantile(.05), p25=lambda x: x.quantile(.25),
        median="median", p75=lambda x: x.quantile(.75),
        p95=lambda x: x.quantile(.95), maximum="max").reset_index()
    return levels, distribution, quantiles


def diagnostic_association(frame):
    """Descriptive conditional errors; not a causal or ESS guarantee analysis."""
    smc = frame[frame.method == "snis"].set_index(["lambda", "run"])
    rows = []
    for tilt in TILTS:
        group = smc.loc[tilt]
        for metric in ("terminal_ess_fraction", "reverse_ess_fraction",
                       "distinct_root_fraction"):
            values = group[metric]
            if values.nunique() <= 1:
                bins = pd.Series("all", index=values.index)
            else:
                bins = pd.qcut(values, 4, labels=False, duplicates="drop").astype(str)
            for bin_label in sorted(bins.unique()):
                selected = bins.index[bins == bin_label]
                for method in ("smc", "snis"):
                    method_runs = frame[(frame.method == method) & (frame["lambda"] == tilt)].set_index("run")
                    estimates = method_runs.loc[selected, "estimate"].to_numpy()
                    errors = estimates-P_A
                    rows.append({"lambda": tilt, "diagnostic": metric, "bin": bin_label,
                                 "method": method, "runs": len(estimates),
                                 "diagnostic_min": float(values.loc[selected].min()),
                                 "diagnostic_median": float(values.loc[selected].median()),
                                 "diagnostic_max": float(values.loc[selected].max()),
                                 "bias": float(errors.mean()),
                                 "rmse": float(np.sqrt(np.mean(errors*errors))),
                                 "zero_fraction": float(np.mean(estimates == 0))})
    return pd.DataFrame(rows)


def plot_diagnostics(levels, frame, out):
    smc = frame[frame.method == "snis"]
    colors = plt.cm.viridis(np.linspace(.12, .85, len(TILTS)))
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for ax, metric, title in ((axes[0, 0], "pre_ess_fraction", "ESS/N before resampling"),
                              (axes[0, 1], "pre_max_weight", "Max normalized weight before resampling")):
        for tilt, color in zip(TILTS, colors):
            sub = levels[levels["lambda"] == tilt]
            q = sub.groupby("level")[metric].quantile([.05, .25, .5, .75, .95]).unstack()
            x = q.index.to_numpy(dtype=float)
            ax.fill_between(x, q[.05], q[.95], color=color, alpha=.12)
            ax.fill_between(x, q[.25], q[.75], color=color, alpha=.25)
            ax.plot(x, q[.5], marker="o", color=color, label=f"λ={tilt:g}")
        ax.set_title(title + "\n5–95%, 25–75%, median")
        ax.set_xlabel("SMC level k")
        ax.set_xticks(range(1, R+1))
        ax.grid(alpha=.2)
    axes[0, 0].axhline(.5, linestyle="--", color="black", linewidth=.8)
    axes[0, 0].legend(ncol=2, fontsize=8)
    for ax, pair, title in ((axes[1, 0], ("terminal_ess_fraction", "reverse_ess_fraction"), "Terminal vs corrected ESS/N"),
                            (axes[1, 1], ("terminal_max_weight", "reverse_max_weight"), "Terminal vs corrected max weight")):
        for index, (metric, color, offset, label) in enumerate(zip(pair, ("#3769a8", "#df7835"), (-.17, .17), ("SMC weights", "inverse-corrected"))):
            arrays = [smc[smc["lambda"] == tilt][metric].to_numpy() for tilt in TILTS]
            positions = np.arange(len(TILTS)) + offset
            bp = ax.boxplot(arrays, positions=positions, widths=.28, whis=[5,95], showfliers=False, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(.6)
            ax.plot([], [], color=color, linewidth=7, alpha=.6, label=label)
        ax.set_xticks(range(len(TILTS)), [f"{t:g}" for t in TILTS])
        ax.set_xlabel("terminal tilt λ")
        ax.set_title(title + "\nbox: 25–75%; whiskers: 5–95%")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[2, 0].boxplot([smc[smc["lambda"] == tilt].distinct_root_fraction.to_numpy() for tilt in TILTS],
                       positions=np.arange(len(TILTS)), whis=[5,95], showfliers=False)
    axes[2, 0].set_xticks(range(len(TILTS)), [f"{t:g}" for t in TILTS])
    axes[2, 0].set_xlabel("terminal tilt λ")
    axes[2, 0].set_title("Distinct initial ancestor fraction (5–95% whiskers)")
    axes[2, 0].grid(alpha=.2)
    selection = levels.groupby(["lambda", "level"]).resampled.mean().unstack()
    for tilt, color in zip(TILTS, colors):
        axes[2, 1].plot(selection.columns, selection.loc[tilt], marker="o", color=color, label=f"λ={tilt:g}")
    axes[2, 1].set_xticks(range(1, R+1))
    axes[2, 1].set_xlabel("SMC level k")
    axes[2, 1].set_ylabel("fraction of runs with resampling")
    axes[2, 1].set_title("Systematic resampling activation")
    axes[2, 1].grid(alpha=.2)
    fig.savefig(out / "weight_degeneracy.png", dpi=170)
    plt.close(fig)


def plot_association(association, out):
    data = association[association.diagnostic == "reverse_ess_fraction"]
    colors = plt.cm.viridis(np.linspace(.12, .85, len(TILTS)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for tilt, color in zip(TILTS, colors):
        for method, style in (("smc", "--"), ("snis", "-")):
            sub = data[(data["lambda"] == tilt) & (data.method == method)].sort_values("diagnostic_median")
            axes[0].plot(sub.diagnostic_median, sub.bias, marker="o", linestyle=style,
                         color=color, label=f"λ={tilt:g}, {method}")
            axes[1].plot(sub.diagnostic_median, sub.rmse, marker="o", linestyle=style, color=color)
            if method == "snis":
                axes[2].plot(sub.diagnostic_median, sub.zero_fraction, marker="o", color=color, label=f"λ={tilt:g}")
    for ax, title, ylabel in zip(axes, ("Conditional bias", "Conditional RMSE", "Zero-estimate fraction"),
                                  ("bias vs exact $p_A$", "RMSE vs exact $p_A$", "fraction")):
        ax.set_title(title)
        ax.set_xlabel("median corrected ESS/N within bin")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].legend(fontsize=7, ncol=2)
    axes[2].legend(fontsize=8)
    fig.suptitle("Descriptive bins of inverse-corrected ESS/N; high ESS does not guarantee event hits")
    fig.savefig(out / "degeneracy_error_association.png", dpi=170)
    plt.close(fig)


def compare_methods(frame, summary):
    rows = []
    fixed = summary[(summary.method == "crude_paths") & (summary.n == N)].iloc[0]
    for tilt in TILTS:
        group = summary[summary["lambda"] == tilt]
        smc = group[group.method == "smc"].iloc[0]
        snis = group[group.method == "snis"].iloc[0]
        matched = group[(group.method == "crude_paths") & (group.n != N)].iloc[0]
        direct = group[group.method == "crude_direct"].iloc[0]
        per_run = frame[(frame["lambda"] == tilt) & frame.method.isin(["smc", "snis"])].pivot(index="run", columns="method", values="estimate")
        paired = (per_run["snis"].to_numpy()-P_A)**2-(per_run["smc"].to_numpy()-P_A)**2
        rows.append({"lambda": tilt,
                     "smc_rmse": smc.rmse, "snis_rmse": snis.rmse,
                     "crude_N128_rmse": fixed.rmse,
                     "crude_matched_rmse": matched.rmse,
                     "crude_direct_matched_rmse": direct.rmse,
                     "snis_minus_smc_MSE": float(paired.mean()),
                     "snis_minus_smc_MSE_se_paired": float(paired.std(ddof=1)/np.sqrt(len(paired))),
                     "snis_minus_crude_matched_RMSE": snis.rmse-matched.rmse,
                     "snis_minus_crude_matched_RMSE_se": float(np.hypot(snis.rmse_se, matched.rmse_se)),
                     "smc_median_ms": snis.median_ms,
                     "crude_matched_median_ms": matched.median_ms,
                     "time_ratio_crude_over_smc": matched.median_ms/snis.median_ms,
                     "crude_direct_median_ms": direct.median_ms,
                     "time_ratio_direct_over_smc": direct.median_ms/snis.median_ms,
                     "crude_matched_n": int(matched.n),
                     "crude_direct_matched_n": int(direct.n)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    checks = check_algorithm()
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("GPU and algorithm checks passed:", json.dumps(checks, indent=2), flush=True)
    warm = warmup(50)
    print(f"Warm-up: {warm:.3f} seconds", flush=True)
    pilot_rows = []
    for j, tilt in enumerate(TILTS):
        pilot_rows += records_for("smc", PILOT_N, PILOT_REPS, 10, j, tilt)
    pilot = pd.DataFrame(pilot_rows)
    pilot.to_csv(OUT / "pilot_runs_N64.csv", index=False)
    pilot_summary = summarize(pilot)
    pilot_summary.to_csv(OUT / "pilot_summary_N64.csv", index=False)
    print("N=64 pilot summary:\n", pilot_summary.to_string(index=False), flush=True)
    all_rows = records_for("crude_paths", N, REPS, 20)
    allocations = []
    for j, tilt in enumerate(TILTS):
        print(f"Main terminal tilt {tilt:.2f} ({j+1}/{len(TILTS)})", flush=True)
        smc_rows = records_for("smc", N, REPS, 50, j, tilt)
        all_rows += smc_rows
        target = float(np.median([row["seconds"] for row in smc_rows if row["method"] == "smc"]))
        path_count, path_pilot_time = match_count(target, crude_paths, 30, j, 8192, 524288)
        direct_count, direct_pilot_time = match_count(target, crude_direct, 40, j, 524288, 67108864)
        allocations.append({"lambda": tilt, "measured_smc_median_ms": target*1000,
                            "crude_paths_n": path_count, "crude_paths_pilot_ms": path_pilot_time*1000,
                            "direct_gaussian_n": direct_count, "direct_pilot_ms": direct_pilot_time*1000})
        all_rows += records_for("crude_paths", path_count, REPS, 60, j, tilt)
        all_rows += records_for("crude_direct", direct_count, DIRECT_REPS, 70, j, tilt)
        pd.DataFrame(all_rows).to_csv(OUT / "runs.csv", index=False)
        pd.DataFrame(allocations).to_csv(OUT / "allocations.csv", index=False)
    frame = pd.DataFrame(all_rows)
    summary = summarize(frame)
    summary.to_csv(OUT / "summary.csv", index=False)
    comparison = compare_methods(frame, summary)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    plot_results(summary, OUT)
    levels, distribution, quantiles = diagnostic_tables(frame)
    levels.to_csv(OUT / "level_diagnostics.csv", index=False)
    quantiles.to_csv(OUT / "diagnostic_quantiles.csv", index=False)
    association = diagnostic_association(frame)
    association.to_csv(OUT / "diagnostic_error_association.csv", index=False)
    plot_diagnostics(levels, frame, OUT)
    plot_association(association, OUT)
    report = {"ground_truth": P_A, "threshold_theoretical": a,
              "N_smc": N, "pilot_N": PILOT_N, "R": R,
              "beta_schedule": "beta_k = k lambda / 4, k=0,...,4",
              "terminal_tilts": TILTS, "T": T, "depths": "uniform integers 1,...,16",
              "ESS_trigger": "ESS/N < 0.5, only when k < R",
              "reps_smc_and_path_MC": REPS, "reps_direct_control": DIRECT_REPS,
              "pilot_reps_per_tilt": PILOT_REPS, "warmup_repetitions": 50,
              "root_seed": ROOT_SEED, "gpu": checks["gpu"], "dtype": str(DTYPE),
              "timing": "perf_counter around full estimate, cuda.synchronize before and after"}
    (OUT / "parameters.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)
    print("\nCOMPARISON\n", comparison.to_string(index=False), flush=True)
    print("\nDIAGNOSTIC QUANTILES\n", quantiles.to_string(index=False), flush=True)
    print("\nDIAGNOSTIC ASSOCIATIONS\n", association.to_string(index=False), flush=True)
    print("\nALLOCATIONS\n", pd.DataFrame(allocations).to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
