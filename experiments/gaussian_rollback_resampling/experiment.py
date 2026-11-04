"""Whole-algorithm independent runs; also embedded in the Colab notebook."""
import json
import time
import platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def population_row(result, config, policy, cid, rep, seed, seconds):
    estimates = result.estimates(config.tilt)
    return {"config_id": cid, "lambda": config.tilt, "D": config.max_depth,
            "policy": policy, "replicate": rep, "seed": json.dumps(seed), **estimates,
            "log_z": result.log_z, "seconds": seconds, "transitions": result.transitions,
            "acceptance": result.accepted/result.proposals if result.proposals else 0,
            "distinct_roots": int(np.unique(result.roots).size),
            "tilted_probability_estimate": float(np.sum(np.exp(result.log_weights)*(result.paths[:, -1] >= THRESHOLD))),
            "resampling_count": sum(s["resampled"] for s in result.levels),
            "resampled_levels": json.dumps([s["level"] for s in result.levels if s["resampled"]]),
            "ess_before": json.dumps([s["ess_before"] for s in result.levels]),
            "ess_after": json.dumps([s["ess_after"] for s in result.levels]),
            "log_c_path": json.dumps([s["log_c"] for s in result.levels]),
            "log_z_path": json.dumps([s["log_z"] for s in result.levels]),
            "stage_transitions": json.dumps([s["transitions_stage"] for s in result.levels])}


def summarize_runs(frame, n, steps):
    summaries = []
    for (cid, policy), group in frame.groupby(["config_id", "policy"], sort=True):
        if policy == "crude":
            estimators = ["ordinary"]
        else:
            estimators = ["ordinary", "snis"]
        for estimator in estimators:
            values = group[estimator].to_numpy()
            b = len(values)
            variance = values.var(ddof=1)
            bias = values.mean()-P_EVENT
            summaries.append({"config_id": cid, "lambda": group["lambda"].iloc[0],
                              "D": group["D"].iloc[0], "policy": policy, "estimator": estimator,
                              "N": n, "runs": b, "mean": values.mean(), "bias": bias,
                              "bias_mc_se": np.sqrt(variance/b), "variance": variance,
                              "variance_mc_se": ((values-values.mean())**2).std(ddof=1)/np.sqrt(b),
                              "rmse": np.sqrt(np.mean((values-P_EVENT)**2)),
                              "zero_frequency": np.mean(values == 0), "median_seconds": group.seconds.median(),
                              "mean_transitions": group.transitions.mean(),
                              "variance_times_cost_relative_mc": variance*group.transitions.mean()/(P_EVENT*(1-P_EVENT)*steps),
                              "mse_times_cost_relative_mc": np.mean((values-P_EVENT)**2)*group.transitions.mean()/(P_EVENT*(1-P_EVENT)*steps),
                              "resampling_run_fraction": np.mean(group.resampling_count > 0),
                              "mean_resampling_count": group.resampling_count.mean(),
                              "mean_z_hat": group.z_hat.mean(), "z_hat_mc_se": group.z_hat.std(ddof=1)/np.sqrt(b),
                              "true_z": np.exp(float(group["lambda"].iloc[0])**2/2),
                              "mean_reverse_mass": group.reverse_mass.mean(),
                              "reverse_mass_mc_se": group.reverse_mass.std(ddof=1)/np.sqrt(b),
                              # A diagnostic using complete independent runs, not an IID-particle CLT.
                              "whole_run_linearization_variance": np.var(group.ordinary-P_EVENT*group.reverse_mass, ddof=1)})
    return pd.DataFrame(summaries)


def run_comparison(n=2048, runs=2000, root_seed=20260929, output_dir="rollback_smc_results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = [Config(tilt=0, max_depth=16)] + [Config(tilt=lam, max_depth=d) for d in (1, 4, 16) for lam in (1, 2, 3)]
    manifest = {"N": n, "runs": runs, "root_seed": root_seed, "p": P_EVENT, "a": THRESHOLD,
                "configs": [asdict(c) for c in configs], "resampler": "systematic, one common uniform offset, input order retained",
                "trigger": "ESS/N < 0.5 and k < R", "order": "weight -> resample if triggered -> MH at new beta",
                "final_resampling": False, "seed_scheme": "SeedSequence([root_seed, stream, config_id, replicate])",
                "streams": {"crude": 0, "independent_baseline": 1, "ess_systematic": 4},
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "pandas": pd.__version__, "variance_method": "sample variance across independent complete algorithm runs; no IID-particle CLT for resampling"}
    started = time.perf_counter()
    rows = []
    for rep in range(runs):
        seed = [root_seed, 0, 0, rep]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        now = time.perf_counter()
        paths = base_paths(rng, n, 16)
        value = float(np.mean(paths[:, -1] >= THRESHOLD))
        seconds = time.perf_counter()-now
        rows.append({"config_id": -1, "lambda": 0, "D": 0, "policy": "crude", "replicate": rep,
                     "seed": json.dumps(seed), "ordinary": value, "snis": value, "reverse_mass": 1.,
                     "z_hat": 1., "log_z": 0., "seconds": seconds, "transitions": n*16,
                     "acceptance": 0., "distinct_roots": n, "tilted_probability_estimate": value,
                     "resampling_count": 0, "resampled_levels": "[]", "ess_before": "[]", "ess_after": "[]",
                     "log_c_path": "[]", "log_z_path": "[]", "stage_transitions": "[]"})
    print(f"Crude MC: {runs} run complete", flush=True)
    for cid, cfg in enumerate(configs):
        for policy, stream in (("independent", 1), ("ess_systematic", 4)):
            for rep in range(runs):
                seed = [root_seed, stream, cid, rep]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                now = time.perf_counter()
                if policy == "independent":
                    result = run_independent_baseline(rng, n, cfg)
                else:
                    result = run_smc(rng, n, cfg)
                estimates = result.estimates(cfg.tilt)
                seconds = time.perf_counter()-now
                assert np.isfinite(estimates["ordinary"]) and np.isfinite(estimates["snis"])
                assert not result.levels[-1]["resampled"]
                rows.append(population_row(result, cfg, policy, cid, rep, seed, seconds))
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
            print(f"lambda={cfg.tilt:g}, D={cfg.max_depth}, {policy}: {runs} run; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize_runs(frame, n, 16)
    summary.to_csv(out / "summary.csv", index=False)
    # Not sufficient to run a dormant ESS policy: require actual activation.
    active = frame.query("policy == 'ess_systematic' and resampling_count > 0")
    assert len(active) > 0, "Experiment invalid: resampling never activated"
    assert not any(8 in json.loads(v) for v in frame.resampled_levels)
    for _, row in frame.query("policy != 'crude'").iterrows():
        np.testing.assert_allclose(sum(json.loads(row.log_c_path)), row.log_z, atol=1e-11)
        assert row.transitions == n*16 + sum(json.loads(row.stage_transitions))
    manifest["active_resampling_runs"] = len(active)
    manifest["simulation_and_export_seconds"] = time.perf_counter()-started
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Resampling realmente attivo in {len(active)} run; mai al livello finale.")
    return frame, summary, manifest


def create_figures(frame, summary, manifest, output_dir="rollback_smc_results"):
    out = Path(output_dir)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    n = manifest["N"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    colors = {"ordinary": "#176b8c", "snis": "#b94b32"}
    for ax, depth in zip(axes, (1, 4, 16)):
        for policy, style in (("independent", "--"), ("ess_systematic", "-")):
            for estimator in ("ordinary", "snis"):
                g = summary.query("D == @depth and policy == @policy and estimator == @estimator and `lambda` > 0").sort_values("lambda")
                ax.errorbar(g["lambda"], g.variance, yerr=1.96*g.variance_mc_se, linestyle=style,
                            marker="o" if estimator == "ordinary" else "s", color=colors[estimator],
                            capsize=2, label=f"{policy}: {estimator}")
        ax.axhline(P_EVENT*(1-P_EVENT)/n, color="gray", linestyle=":", label="Crude MC: exact")
        ax.set(title=f"Uniform rollback depth 1,...,{depth}", xlabel="lambda", yscale="log", xticks=[1, 2, 3])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Variance across independent full runs")
    axes[-1].legend(fontsize=7)
    fig.suptitle("ESS systematic vs independent baseline — empirical variances, not an IID-particle CLT")
    fig.tight_layout()
    fig.savefig(out / "variance_comparison.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    levels = np.arange(1, 9)
    for col, depth in enumerate((1, 4, 16)):
        for tilt, color in zip((1, 2, 3), ("#176b8c", "#b94b32", "#5b4594")):
            g = frame.query("D == @depth and `lambda` == @tilt and policy == 'ess_systematic'")
            ess = np.array([json.loads(v) for v in g.ess_before])/n
            activation = np.array([[k in json.loads(v) for k in levels] for v in g.resampled_levels])
            axes[0, col].plot(levels, ess.mean(axis=0), "o-", color=color, label=f"lambda={tilt}")
            axes[0, col].fill_between(levels, np.quantile(ess, .1, axis=0), np.quantile(ess, .9, axis=0), color=color, alpha=.12)
            axes[1, col].plot(levels, activation.mean(axis=0), "o-", color=color)
        axes[0, col].axhline(.5, color="black", linestyle=":")
        axes[0, col].set(title=f"D={depth}", ylim=(0, 1.05))
        axes[1, col].set(xlabel="SMC level k", ylim=(-.03, 1.03), xticks=levels)
        axes[0, col].grid(alpha=.2)
        axes[1, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("ESS/N before resampling (mean, 10–90%)")
    axes[1, 0].set_ylabel("Fraction of runs resampling at level k")
    axes[0, -1].legend()
    fig.suptitle("The ESS branch activates; the final level never resamples")
    fig.tight_layout()
    fig.savefig(out / "resampling_diagnostics.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for depth, style in ((1, ":"), (4, "--"), (16, "-")):
        for estimator in ("ordinary", "snis"):
            g = summary.query("policy == 'ess_systematic' and D == @depth and estimator == @estimator and `lambda` > 0").sort_values("lambda")
            options = dict(color=colors[estimator], linestyle=style, marker="o" if estimator == "ordinary" else "s", label=f"{estimator}, D={depth}")
            axes[0, 0].errorbar(g["lambda"], g.bias/P_EVENT, yerr=1.96*g.bias_mc_se/P_EVENT, capsize=2, **options)
            axes[0, 1].plot(g["lambda"], g.rmse/P_EVENT, **options)
            axes[1, 0].plot(g["lambda"], g.zero_frequency, **options)
            axes[1, 1].plot(g["lambda"], g.mse_times_cost_relative_mc, **options)
    axes[0, 0].axhline(0, color="gray", linestyle=":")
    axes[0, 1].axhline(np.sqrt(P_EVENT*(1-P_EVENT)/n)/P_EVENT, color="gray", linestyle=":")
    axes[1, 0].axhline((1-P_EVENT)**n, color="gray", linestyle=":")
    axes[1, 1].axhline(1, color="gray", linestyle=":")
    for ax, title in zip(axes.flat, ("Relative bias (Monte Carlo uncertainty)", "Relative RMSE", "Zero-estimate frequency", "MSE × transitions / crude MC")):
        ax.set(title=title, xlabel="lambda", xticks=[1, 2, 3])
        ax.grid(alpha=.2)
    axes[1, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_cost.png", dpi=160)
    plt.show()
    plt.close(fig)


def table_markdown(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(lines)


def write_report(frame, summary, manifest, output_dir="rollback_smc_results"):
    out = Path(output_dir)
    active = summary.query("policy == 'ess_systematic' and `lambda` > 0")
    best = active.loc[active.rmse.idxmin()]
    selected = summary[(summary.D == best.D)&(summary["lambda"] == best["lambda"])]
    crude = summary.query("policy == 'crude'")
    comparison = pd.concat([crude, selected])
    metrics = ["lambda", "D", "policy", "estimator", "bias", "bias_mc_se", "variance", "rmse", "zero_frequency"]
    cost_cols = ["lambda", "D", "policy", "median_seconds", "mean_transitions", "resampling_run_fraction", "mean_resampling_count"]
    norm_cols = ["lambda", "D", "mean_z_hat", "z_hat_mc_se", "true_z", "mean_reverse_mass", "reverse_mass_mc_se"]
    report = ["# Risultati: RollBack SMC con resampling ESS/systematic", "",
              f"N={manifest['N']}; {manifest['runs']} run indipendenti dell'intero algoritmo per configurazione; seed principale {manifest['root_seed']}. T=16, R=8, una mutazione per livello; lambda=1,2,3; profondità uniforme 1,...,D, D=1,4,16. Controllo lambda=0. Baseline senza resampling ricalcolata con gli stessi seed originali, senza modificare i file precedenti.", "",
              "Politica: systematic con un offset uniforme, ordine input conservato, ESS/N < 0.5, pesatura -> eventuale resampling -> mutazione a beta_k. Nessun resampling finale. Z_hat accumula tutti gli incrementi, anche senza resampling.", "",
              f"Resampling realmente attivo in {manifest['active_resampling_runs']} run. Durata simulazioni e salvataggio: {manifest['simulation_and_export_seconds']:.2f} s. I tempi includono anche la diagnostica. Le due stime di una popolazione condividono il costo; i loro tempi non vanno sommati.", "",
              "## Teoria e limiti", "",
              "L'ordinario è non distorto a N finito per la politica implementata, per unbiasedness condizionata dei conteggi systematic. E[Z_hat]=Z ed E[reverse_mass]=1. SNIS è un rapporto e può essere distorto. La derivazione include una decomposizione martingala esatta della varianza con covarianze systematic; non attribuisce a questa politica la CLT IID della baseline né una CLT multinomiale. La CLT tra run complete, a N fissato, riguarda invece la media delle run con centro nella sua aspettativa.", "",
              "## Confronto nella configurazione con RMSE minimo osservato", "",
              f"Minimo descrittivo della griglia con resampling: lambda={best['lambda']:g}, D={best.D:g}, {best.estimator}. Non è una prova di ottimalità né una selezione indipendente dei parametri.", "",
              table_markdown(comparison[metrics]), "",
              f"Per questo stimatore, MSE × costo in transizioni / crude MC teorico = {best.mse_times_cost_relative_mc:.3f}. Un valore >1 significa che il guadagno per particella non compensa il costo. Non è una simulazione a budget abbinato.", "",
              "## Tutte le configurazioni: accuratezza", "", table_markdown(summary[metrics]), "",
              "Bias e varianze sono stimati da run intere indipendenti. SE(bias) è l'incertezza della media delle run, non della singola popolazione. I dati originali conservano seed, ESS per livello, resampling, log-incrementi, Z_hat e transizioni. Zero stime nulle osservate non significa probabilità di zero esattamente nulla.", "",
              "## Costo e attivazione", "", table_markdown(summary.query("estimator == 'ordinary'")[cost_cols]), "",
              "## Normalizzatori e identità della popolazione", "", table_markdown(summary.query("policy == 'ess_systematic' and estimator == 'ordinary'")[norm_cols]), "",
              "Le medie vanno confrontate con i loro errori Monte Carlo; non si pretende uguaglianza numerica esatta a campione finito. Nel CSV `whole_run_linearization_variance` è Var(U_N-p V_N) tra run: diagnostica di primo ordine del rapporto, non varianza CLT teorica dell'SMC e non predizione indipendente.", "",
              "## Grafici", "", "![Varianze](variance_comparison.png)", "",
              "![Attivazione ESS](resampling_diagnostics.png)", "", "![Accuratezza e costo](accuracy_cost.png)", ""]
    (out / "RISULTATI.md").write_text("\n".join(report), encoding="utf-8")
    return comparison[metrics]


if __name__ == "__main__":
    from smc_resampling import *
    frame, summary, manifest = run_comparison()
    create_figures(frame, summary, manifest)
    print(write_report(frame, summary, manifest).to_string(index=False))
