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

The toxicity threshold q is a classifier score, not the original event
probability. The independent finite reference is 444/200000 for q>=0.99 and
26/200000 for q>=0.999; it is not exact ground truth. The Gaussian probabilities
above and these estimated probabilities must not be conflated.

| Study | Results |
|---|---|
| Completed same-run ladder, Phi=log(q), N=128, 12 seeds, lambda=0.5,...,8, 4 MH/level | [Report](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/REPORT.html), [all checkpoints](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/checkpoint_runs.csv), [log-error plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/same_run_ladder_log_error.png), [ratio plots](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_ladder/same_run_ladder_ratio.png) |
| Logit-Phi follow-up, same parameters | [Configuration](mdlm_toxicity_rare_event/configs/mdlm_owt_same_run_logit.yaml), [methods](mdlm_toxicity_rare_event/docs/SAME_RUN_LOGIT.md), [status](mdlm_toxicity_rare_event/artifacts/same_run_logit_job_status.json), [available artifacts](mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit/) |
| Gradual low-budget / strong-tilt screening | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_gradual_budget/) |
| Accuracy-budget pilot | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_accuracy_budget_pilot/) |
| Earlier schedule / mutation exploration | [Transition comparison](mdlm_toxicity_rare_event/artifacts/current/camera_ready_exploration_v2/), [wall-clock follow-up](mdlm_toxicity_rare_event/artifacts/current/camera_ready_exploration_v2_wall_clock/) |
| Earlier low-budget pilot | [Report and run data](mdlm_toxicity_rare_event/artifacts/current/camera_ready_low_budget_v1/) |
| Historical Avon outputs | [Archive](mdlm_toxicity_rare_event/artifacts/archive/), [completeness notes](mdlm_toxicity_rare_event/artifacts/README.md) |

**Logit job 2081350 was still running when this publication was prepared.**
Launch metadata and code do not constitute completed experimental results.
Available run files and the status JSON distinguish partial from completed
studies; no unavailable results are inferred or regenerated for publication.

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
