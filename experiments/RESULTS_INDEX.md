# Saved experiment results

This index accompanies the code, notebooks, individual-run data, figures, and
historical results published together in this repository. The files retain
their existing experiment directories. See `results_manifest.json` for file
sizes and SHA256 hashes of the published snapshot.

## Gaussian full-trajectory toy model

The native model has T=16 Gaussian transitions and terminal marginal N(0,1).
Event thresholds use theoretical normal quantiles; the event probabilities in
these experiments are exact. Each study's parameters, seeds, and comparison
budget are recorded in its notebook and manifests. Historical studies used
different grids: do not pool their results or apply no-resampling variance
formulas to runs with resampling.

| Study | Code / notebook | Saved runs and results |
|---|---|---|
| Independent paths, no resampling | [Code and derivation](gaussian_rollback_smc/README.md) | [Results](gaussian_rollback_smc/RISULTATI.md), [individual replicates](gaussian_rollback_smc/results/replicates.csv) |
| ESS-triggered systematic resampling | [Self-contained Colab notebook](gaussian_rollback_resampling/RollBack_SMC_Colab.ipynb), [theory](gaussian_rollback_resampling/TEORIA.md) | [Individual runs](gaussian_rollback_resampling/results/runs.csv), [summary](gaussian_rollback_resampling/results/summary.csv), [results folder](gaussian_rollback_resampling/results/) |
| Historical GPU comparisons | [Colab notebook](gaussian_rollback_gpu_previous/RollBack_SMC_GPU_Colab.ipynb), [sampler](gaussian_rollback_gpu_previous/gpu_model.py) | [Results](gaussian_rollback_gpu_previous/results/), [individual runs](gaussian_rollback_gpu_previous/results/runs.csv) |
| Historical deeper tilt ladders | [Colab notebook](gaussian_rollback_gpu_previous/RollBack_SMC_deep_tilts_Colab.ipynb), [sampler](gaussian_rollback_gpu_previous/deep_model.py) | [Results](gaussian_rollback_gpu_previous/deep_results/), [individual runs](gaussian_rollback_gpu_previous/deep_results/runs.csv) |

Other checkpoint notebooks, small verification runs, saved plots, and original
result archives are retained in the corresponding directories. Execution
verification records describe which notebook versions were actually run.

## MDLM toxicity experiments

The toxicity threshold q is a classifier score. The estimated event
probabilities are 444/200,000 for q ≥ 0.99, 334/298,800 for q ≥ 0.995,
and 26/200,000 for q ≥ 0.999.

| Study | Results |
|---|---|
| Three-threshold reanalysis including q ≥ 0.995 | [Ratio boxplots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/plots/same_run_ladder_ratio_3_thresholds.png), [log-error boxplots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/plots/same_run_ladder_log_error_3_thresholds.png), [relative standard deviation](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/plots/relative_std_SMC_3_thresholds.png), [dispersion table](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/relative_std_comparison_3_thresholds.csv) |
| Completed same-run ladder, Phi=log(q), N=128, 12 seeds, lambda=0.5,...,8, 4 MH/level | [Report](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/REPORT.html), [all checkpoints](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/checkpoint_runs.csv), [log-error plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/plots/same_run_ladder_log_error.png), [ratio plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/plots/same_run_ladder_ratio.png) |
| Completed logit-Phi follow-up, same parameters and 12 seeds | [Report](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/REPORT.html), [all checkpoints](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/checkpoint_runs.csv), [log-error plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/plots/same_run_ladder_log_error.png), [ratio plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/plots/same_run_ladder_ratio.png), [validation](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/delivery_validation.json) |
| Gradual low-budget / strong-tilt screening | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_gradual_budget/) |
| Accuracy-budget pilot | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_accuracy_budget_pilot/) |
| Earlier schedule / mutation exploration | [Transition comparison](mdlm_toxicity_rare_event/artifacts/current/camera_ready_exploration_v2/), [wall-clock follow-up](mdlm_toxicity_rare_event/artifacts/current/camera_ready_exploration_v2_wall_clock/) |
| Earlier low-budget pilot | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_low_budget_v1/) |
| Historical Avon outputs | [Archive](mdlm_toxicity_rare_event/artifacts/archive/), [completeness notes](mdlm_toxicity_rare_event/artifacts/README.md) |

**All 12 logit ladders are complete and validated.** The [completion record](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/COMPLETION_STATUS.json)
distinguishes successful scientific runs from the scheduler's final status:
all three GPU workers exited successfully; the batch job reported FAILED only
because Matplotlib was absent during plotting. Both figures were subsequently
rendered locally from the saved data. No simulation was repeated.

All 1560 checkpoint/event estimator pairs were recomputed from saved weights,
Phi, and normalizers, and all MC prefix estimates and source hashes were
verified. Cross-platform float64 differences were below 4e-15 relative.
The earlier [3-run partial snapshot](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/PARTIAL_STATUS.json)
is retained as historical provenance; it is superseded by the completion record.
See also the [configuration](mdlm_toxicity_rare_event/configs/mdlm_owt_same_run_logit.yaml)
and [methods](mdlm_toxicity_rare_event/docs/SAME_RUN_LOGIT.md).

For the completed same-run ladder, MC reuses a frozen pool of independent
native trajectories in fixed disjoint blocks across seeds, with nested prefixes
within a seed. Counts use measured cumulative SMC time and independently
calibrated MC throughput on the same physical GPU and CPU allocation. MC times
are calibrated equivalents, not newly measured timings at every checkpoint.
Each estimate uses 1 GPU and 4 CPUs; three GPUs parallelize independent seeds.
Both SMC estimators share a population and its generation cost.

Figures preserve zero estimates, identify positive-only conditional panels,
show full tails as well as zooms, and distinguish the median from the mean.
Checkpoints of the same run are correlated and are not independent replicates.

### Reanalysis at q ≥ 0.995

The [reference calculation](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/reference_3_thresholds.json)
uses the 298,800 saved crude trajectories outside the union of the prefixes
used by the two same-run ladders. It gives 0.0011178 with a 95% Wilson interval
of [0.0010042, 0.0012442]. The earlier thresholds use their original
200,000-trajectory reference.

The extended checkpoint tables contain 780 additional estimates for each
potential. [Logit checkpoints](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/checkpoint_runs_3_thresholds.csv) ·
[Log(q) checkpoints](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/checkpoint_runs_3_thresholds.csv)

Relative standard deviation is the sample standard deviation across the
12 independent runs at each fixed tilt, divided by the event's reference
probability. All zero estimates contribute to this calculation. Run
`python experiments/mdlm_toxicity_rare_event/analysis/add_threshold_0995.py`
from the repository root to regenerate the tables and figures from saved
populations and crude scores.

## Paper analysis bundle

The [reviewer artifact](../output/github_reviewer_artifact/README.md) contains
released aggregate and per-run/component tables, frozen source, and scripts to
reproduce the paper figures. Its [data card](../output/github_reviewer_artifact/data/DATA_CARD.md)
describes scope and reference-data availability.

## Publication scope

This snapshot publishes the scientific result files found in the experiment
directories, including prior results and available Gaussian toy versions.
Model weights, virtual environments, caches, credentials, private SSH/Avon
connection files, machine logs, and temporary working directories are excluded.
Generated text and terminal token arrays, where saved as part of an experiment,
are research samples from the stated models rather than independently
calibrated labels of harmful content.
