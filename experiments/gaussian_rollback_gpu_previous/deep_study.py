"""Tests, independent replications, crude-MC budgets and plots for deep tilt."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from deep_model import (T, N, R, DELTA_BETA, LAMBDA, P_EVENTS, THRESHOLDS,
                        DEVICE, DTYPE, require_gpu, make_seed, generator,
                        gaussian_paths, rollback_M_k, systematic, timed,
                        initialize, weight_before_M_k, evaluate_level,
                        terminal_copy_evaluation, main_resample_mutate_evaluate,
                        run_deep, crude_paths_two_events, crude_prefix_estimates,
                        warmup)

OUT = Path("deep_tilt_results")
ROOT_SEED = 20261001
REPS = 320
CRUDE_REPS = 320
SINGLE_SEED = make_seed(1, 0, root=ROOT_SEED)


def check_algorithm():
    device = require_gpu()
    assert (T, N, R, DELTA_BETA, LAMBDA) == (16, 128, 20, .25, 5.0)
    for p, threshold in zip(P_EVENTS, THRESHOLDS):
        assert abs(threshold-norm.isf(p)) < 1e-14
    paths = gaussian_paths(131072, generator(make_seed(90, 0)))
    assert abs(paths[:, -1].mean().item()) < .012
    assert abs(paths[:, -1].var(unbiased=True).item()-1) < .018
    beta = 1.5
    tilted = paths + torch.arange(T+1, device=DEVICE, dtype=DTYPE)[None, :]*(beta/T)
    moved, depths, _ = rollback_M_k(tilted, beta, generator(make_seed(90, 1)))
    counts = torch.bincount(depths, minlength=T+1)[1:].cpu().numpy()
    assert counts.shape == (T,) and np.max(abs(counts-len(paths)/T))/(len(paths)/T) < .055
    cut = T-depths
    assert torch.all(torch.where(torch.arange(T+1, device=DEVICE)[None, :] <= cut[:, None],
                                 moved == tilted, True))
    assert abs(moved[:, -1].mean().item()-beta) < .025
    assert abs(moved[:, -1].var(unbiased=True).item()-1) < .025
    weights = np.array([.05, .35, .10, .50], dtype=np.float32)
    parents = systematic(torch.tensor(np.log(weights), device=DEVICE), generator(1),
                         torch.tensor(.37, device=DEVICE)).cpu().numpy()
    np.testing.assert_array_equal(parents, np.searchsorted(np.cumsum(weights), (np.arange(4)+.37)/4))

    rows, traces = run_deep(make_seed(91, 0), debug=True)
    assert len(rows) == R*len(P_EVENTS)*2 and len(traces) == R
    assert not any(row["resampled"] for row in rows if row["level"] == R)
    selected = [t["level"] for t in traces if t["main"]["resampled"]]
    assert selected, "The resampling branch must actually execute in tests."
    unselected = [t["level"] for t in traces if not t["main"]["resampled"]]
    assert unselected
    for trace in traces:
        level, beta = trace["level"], trace["beta"]
        before = trace["before"]
        unnorm = before["log_weights"]+DELTA_BETA*before["x"]
        increment = torch.logsumexp(unnorm, 0)
        pre_weights = unnorm-increment
        assert abs(increment.item()-trace["pre"]["log_increment"]) < 2e-6
        assert abs((before["log_z"]+increment-trace["terminal"]["log_z"]).item()) < 3e-6
        expected_ess = torch.exp(-torch.logsumexp(2*pre_weights, 0)).item()/N
        assert abs(expected_ess-trace["pre"]["pre_ess_fraction"]) < 2e-6
        assert trace["main"]["resampled"] == (level < R and expected_ess < .5)
        torch.testing.assert_close(trace["terminal"]["log_weights"], pre_weights)
        torch.testing.assert_close(trace["terminal"]["roots"], before["roots"])
        for result in (trace["main"], trace["terminal"]):
            x = result["paths"][:, -1].cpu().numpy().astype(float)
            w = result["log_weights"].exp().cpu().numpy().astype(float)
            z = math.exp(result["log_z"].item())
            reverse = w*np.exp(-beta*x)
            assert abs(result["reverse_ess_fraction"]-1/(N*np.sum((reverse/reverse.sum())**2))) < 2e-5
            for j, threshold in enumerate(THRESHOLDS):
                numerator = np.sum(reverse*(x >= threshold))
                assert abs(result["smc"][j]-z*numerator) < max(2e-8, 2e-4*P_EVENTS[j])
                assert abs(result["snis"][j]-numerator/reverse.sum()) < max(2e-8, 2e-4*P_EVENTS[j])
            assert result["event_particles"][1] <= result["event_particles"][0]
    return device | {"uniform_depth_min": int(counts.min()),
                     "uniform_depth_max": int(counts.max()),
                     "resampled_test_levels": selected,
                     "no_resampling_test_levels": unselected,
                     "no_final_resampling": True}


def summarize_estimates(frame):
    records = []
    keys = ["population", "p_A", "level", "beta"]
    for key, group in frame.groupby(keys, sort=True):
        p_true = key[1]
        for estimator in ("smc", "snis"):
            vals = group[estimator].to_numpy(dtype=float)
            errors = vals-p_true
            squared = errors*errors
            rmse = float(np.sqrt(squared.mean()))
            records.append(dict(zip(keys, key)) | {"estimator": estimator,
                           "runs": len(group), "bias": float(errors.mean()),
                           "variance": float(vals.var(ddof=1)), "rmse": rmse,
                           "rmse_se": float(squared.std(ddof=1)/(2*np.sqrt(len(group))*rmse)) if rmse > 0 else np.nan,
                           "zero_fraction": float(np.mean(vals == 0)),
                           "median_event_particles": float(group.event_particles.median()),
                           "mean_event_particles": float(group.event_particles.mean()),
                           "median_smc_ess_fraction": float(group.smc_ess_fraction.median()),
                           "median_reverse_ess_fraction": float(group.reverse_ess_fraction.median()),
                           "median_root_fraction": float(group.distinct_root_fraction.median()),
                           "median_main_cumulative_ms": float(group.main_cumulative_seconds.median()*1000),
                           "median_terminal_copy_cumulative_ms": float(group.terminal_copy_cumulative_seconds.median()*1000),
                           "median_main_transitions": float(group.main_cumulative_transitions.median()),
                           "median_terminal_copy_transitions": float(group.terminal_copy_cumulative_transitions.median())})
    return pd.DataFrame(records)


def diagnostic_quantiles(frame):
    records = []
    unique = frame[frame.p_A == P_EVENTS[0]]
    for (population, level, beta), group in unique.groupby(["population", "level", "beta"]):
        for metric in ("pre_ess_fraction", "pre_max_weight", "smc_ess_fraction",
                       "smc_max_weight", "reverse_ess_fraction", "reverse_max_weight",
                       "distinct_root_fraction", "event_particles",
                       "main_cumulative_seconds", "main_cumulative_transitions",
                       "extra_cumulative_seconds", "extra_cumulative_transitions"):
            value = group[metric]
            records.append({"population": population, "level": level, "beta": beta,
                            "metric": metric, "minimum": float(value.min()),
                            "p05": float(value.quantile(.05)), "p25": float(value.quantile(.25)),
                            "median": float(value.median()), "p75": float(value.quantile(.75)),
                            "p95": float(value.quantile(.95)), "maximum": float(value.max())})
    return pd.DataFrame(records)


def median_crude_latency(count, stream, level, reps=8):
    times = []
    # Discard short-lived clock/allocation transients before measuring this size.
    for rep in range(4):
        timed(crude_paths_two_events, int(count), make_seed(stream+1000, rep, level))
    for rep in range(reps):
        _, elapsed = timed(crude_paths_two_events, int(count), make_seed(stream, rep, level))
        times.append(elapsed)
    return float(np.median(times))


def calibrate_count(target, initial, level, stream):
    count = int(np.clip(round(initial/128)*128, 128, 12000000))
    for iteration in range(6):
        elapsed = median_crude_latency(count, stream+iteration, level, reps=24)
        ratio = np.clip(target/max(elapsed, 1e-9), .4, 2.5)
        updated = int(np.clip(round(count*ratio/128)*128, 128, 12000000))
        if abs(updated-count) <= 128:
            break
        count = updated
    measured = median_crude_latency(count, stream+20, level, reps=48)
    for correction in range(2):
        if abs(measured/target-1) <= .08:
            break
        ratio = np.clip(target/max(measured, 1e-9), .5, 2.0)
        count = int(np.clip(round(count*ratio/128)*128, 128, 12000000))
        measured = median_crude_latency(count, stream+30+correction, level, reps=48)
    return count, measured


def prepare_crude_budgets(frame):
    subset = frame[frame.p_A == P_EVENTS[0]]
    rows = []
    for population in ("main", "terminal_copy"):
        previous_count = 4096
        previous_target = None
        for level in range(1, R+1):
            group = subset[(subset.population == population) & (subset.level == level)]
            if population == "main":
                target_seconds = float(group.main_cumulative_seconds.median())
                target_transitions = float(group.main_cumulative_transitions.median())
            else:
                target_seconds = float(group.terminal_copy_cumulative_seconds.median())
                target_transitions = float(group.terminal_copy_cumulative_transitions.median())
            transitions_n = max(1, int(round(target_transitions/T)))
            if previous_target is not None:
                previous_count = max(128, int(previous_count*target_seconds/previous_target))
            time_n, measured_seconds = calibrate_count(target_seconds, previous_count, level,
                                                       300 if population == "main" else 400)
            previous_count, previous_target = time_n, target_seconds
            rows.append({"population": population, "level": level,
                         "beta": DELTA_BETA*level, "cost_match": "transitions",
                         "crude_n": transitions_n,
                         "target_smc_transitions": target_transitions,
                         "target_smc_seconds": target_seconds,
                         "calibrated_crude_seconds": median_crude_latency(transitions_n, 500, level, reps=24)})
            rows.append({"population": population, "level": level,
                         "beta": DELTA_BETA*level, "cost_match": "gpu_time",
                         "crude_n": time_n,
                         "target_smc_transitions": target_transitions,
                         "target_smc_seconds": target_seconds,
                         "calibrated_crude_seconds": measured_seconds})
    return pd.DataFrame(rows)


def run_crude_budgets(budgets):
    requested = budgets.crude_n.to_numpy(dtype=int)
    rows = []
    for rep in range(CRUDE_REPS):
        seed = make_seed(50, rep)
        estimates = crude_prefix_estimates(requested, seed)
        for budget, values in zip(budgets.to_dict("records"), estimates):
            for event_index, p_true in enumerate(P_EVENTS):
                rows.append({"run": rep, "seed": seed, "population": budget["population"],
                             "level": budget["level"], "beta": budget["beta"],
                             "cost_match": budget["cost_match"], "p_A": p_true,
                             "crude_n": budget["crude_n"], "estimate": values[event_index]})
        if (rep+1) % 80 == 0:
            print(f"Crude MC independent replicates: {rep+1}/{CRUDE_REPS}", flush=True)
    return pd.DataFrame(rows)


def summarize_crude(frame, budgets):
    records = []
    keys = ["population", "cost_match", "p_A", "level", "beta", "crude_n"]
    for key, group in frame.groupby(keys, sort=True):
        p_true, count = key[2], key[5]
        errors = group.estimate.to_numpy()-p_true
        squared = errors*errors
        rmse = float(np.sqrt(squared.mean()))
        budget = budgets[(budgets.population == key[0]) & (budgets.cost_match == key[1]) &
                         (budgets.level == key[3])].iloc[0]
        records.append(dict(zip(keys, key)) | {"runs": len(group),
                       "bias": float(errors.mean()), "variance": float(group.estimate.var(ddof=1)),
                       "rmse": rmse,
                       "rmse_se": float(squared.std(ddof=1)/(2*np.sqrt(len(group))*rmse)) if rmse > 0 else np.nan,
                       "zero_fraction": float(np.mean(group.estimate == 0)),
                       "exact_binomial_rmse": float(np.sqrt(p_true*(1-p_true)/count)),
                       "crude_transitions": count*T,
                       "calibrated_crude_ms": budget.calibrated_crude_seconds*1000,
                       "target_smc_ms": budget.target_smc_seconds*1000,
                       "time_ratio_crude_to_smc": budget.calibrated_crude_seconds/budget.target_smc_seconds})
    return pd.DataFrame(records)


def plot_single_run(frame, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_index, p_true in enumerate(P_EVENTS):
        for population, color in (("main", "#1e6da8"), ("terminal_copy", "#dd733d")):
            group = frame[(frame.p_A == p_true) & (frame.population == population)]
            for estimator, style in (("smc", "-"), ("snis", "--")):
                axes[row_index, 0].plot(group.beta, group[estimator], color=color, linestyle=style,
                                        marker=".", label=f"{population}, {estimator}")
            axes[row_index, 1].plot(group.beta, group.event_particles, marker="o", color=color,
                                    label=population)
        axes[row_index, 0].axhline(p_true, color="black", linestyle=":", label="exact $p_A$")
        axes[row_index, 0].set_yscale("symlog", linthresh=p_true/10)
        axes[row_index, 0].set_ylabel(f"estimate of $p_A={p_true:g}$")
        axes[row_index, 1].set_ylabel("particles terminating in A")
        for ax in axes[row_index]:
            ax.set_xlabel(r"current $\beta_k$")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Per-level estimates; no averaging across levels")
    axes[0, 1].set_title("Rare-event particles")
    fig.savefig(out / "single_run.png", dpi=170)
    plt.close(fig)


def plot_replications(summary, crude, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, p_true in enumerate(P_EVENTS):
        for j, population in enumerate(("main", "terminal_copy")):
            ax = axes[i, j]
            for estimator, style in (("smc", "-"), ("snis", "--")):
                sub = summary[(summary.p_A == p_true) & (summary.population == population) &
                              (summary.estimator == estimator)].sort_values("beta")
                ax.plot(sub.beta, sub.rmse, linestyle=style, marker="o", label=estimator)
                ax.fill_between(sub.beta, np.maximum(1e-12, sub.rmse-1.96*sub.rmse_se),
                                sub.rmse+1.96*sub.rmse_se, alpha=.12)
            for cost, color in (("transitions", "#555555"), ("gpu_time", "#cf7432")):
                sub = crude[(crude.p_A == p_true) & (crude.population == population) &
                            (crude.cost_match == cost)].sort_values("beta")
                ax.plot(sub.beta, sub.exact_binomial_rmse, marker="s", color=color,
                        label=f"crude MC, matched {cost} (exact binomial RMSE)")
            ax.set_yscale("log")
            ax.set_title(f"{population}, $p_A={p_true:g}$")
            ax.set_xlabel(r"current $\beta_k$")
            ax.set_ylabel("RMSE")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.savefig(out / "rmse_by_level.png", dpi=170)
    plt.close(fig)


def plot_diagnostics(frame, out):
    data = frame[(frame.p_A == P_EVENTS[0]) & (frame.population == "main")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric, title in ((axes[0, 0], "smc_ess_fraction", "SMC ESS/N"),
                              (axes[0, 1], "reverse_ess_fraction", "Inverse-corrected ESS/N"),
                              (axes[1, 0], "distinct_root_fraction", "Distinct initial ancestor fraction"),
                              (axes[1, 1], "reverse_max_weight", "Largest corrected weight")):
        q = data.groupby("beta")[metric].quantile([.05, .25, .5, .75, .95]).unstack()
        x = q.index.to_numpy()
        ax.fill_between(x, q[.05], q[.95], alpha=.15)
        ax.fill_between(x, q[.25], q[.75], alpha=.3)
        ax.plot(x, q[.5], marker=".")
        ax.set_title(title + " (5–95%, 25–75%, median)")
        ax.set_xlabel(r"current $\beta_k$")
        ax.grid(alpha=.2)
    fig.savefig(out / "diagnostics_by_level.png", dpi=170)
    plt.close(fig)


def single_experiment():
    OUT.mkdir(exist_ok=True)
    single_rows, _ = run_deep(SINGLE_SEED)
    single = pd.DataFrame(single_rows)
    single.to_csv(OUT / "single_run.csv", index=False)
    plot_single_run(single, OUT)
    print("Single run first: seed", SINGLE_SEED, flush=True)
    print(single[(single.population == "main") & (single.p_A == P_EVENTS[1])][
        ["level", "beta", "smc", "snis", "event_particles", "smc_ess_fraction",
         "reverse_ess_fraction", "distinct_root_fraction", "main_cumulative_seconds",
         "main_cumulative_transitions"]].to_string(index=False), flush=True)
    return single


def replicated_experiment(checks):
    OUT.mkdir(exist_ok=True)
    all_rows = []
    for rep in range(REPS):
        rows, _ = run_deep(make_seed(10, rep))
        for row in rows:
            row["run"] = rep
        all_rows += rows
        if (rep+1) % 80 == 0:
            print(f"Independent full SMC runs: {rep+1}/{REPS}", flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT / "runs.csv", index=False)
    summary = summarize_estimates(frame)
    summary.to_csv(OUT / "summary.csv", index=False)
    diagnostics = diagnostic_quantiles(frame)
    diagnostics.to_csv(OUT / "diagnostic_quantiles.csv", index=False)
    plot_diagnostics(frame, OUT)
    budgets = prepare_crude_budgets(frame)
    budgets.to_csv(OUT / "crude_budgets.csv", index=False)
    crude_runs = run_crude_budgets(budgets)
    crude_runs.to_csv(OUT / "crude_runs.csv", index=False)
    crude_summary = summarize_crude(crude_runs, budgets)
    crude_summary.to_csv(OUT / "crude_summary.csv", index=False)
    plot_replications(summary, crude_summary, OUT)
    parameters = checks | {"root_seed": ROOT_SEED, "single_seed": SINGLE_SEED,
                           "independent_smc_runs": REPS, "independent_crude_runs": CRUDE_REPS,
                           "schedule": "beta_k=0.25*k, k=0,...,20",
                           "resampling": "systematic when ESS/N<0.5 and k<R",
                           "mutation": "one RollBack-MH per level, depth uniform 1,...,16",
                           "timing": "CUDA synchronize before and after each timed segment; main cumulative excludes terminal-copy evaluation",
                           "crude_time_matching": "standalone full-trajectory P sampler calibrated against median cumulative GPU time",
                           "crude_transition_matching": "median used Gaussian transitions divided by T=16",
                           "crude_estimates": "independent P trajectory pools by replicate; prefixes reused across level budgets"}
    (OUT / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print("\nMAIN SUMMARY (selected levels)\n", summary[(summary.population == "main") &
        (summary.level.isin([1, 5, 10, 15, 20]))].to_string(index=False), flush=True)
    print("\nCRUDE BUDGETS (selected levels)\n", budgets[budgets.level.isin([1, 5, 10, 15, 20])].to_string(index=False), flush=True)
    return summary, crude_summary, diagnostics, budgets


def main():
    OUT.mkdir(exist_ok=True)
    checks = check_algorithm()
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    warmup(20)
    single = single_experiment()
    summary, crude_summary, diagnostics, budgets = replicated_experiment(checks)
    return single, summary, crude_summary, diagnostics, budgets


if __name__ == "__main__":
    main()
