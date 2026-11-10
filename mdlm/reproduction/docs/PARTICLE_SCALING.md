# Population size at fixed tilt and under a lower budget

Both campaigns use the logit RollBack–MH pipeline, with the earlier 12-seed [logit ladder](SAME_RUN_LOGIT.md) as a pilot. The new campaigns use distinct seeds. The YAMLs specify the schedules, rollback choices, mutation counts and systematic resampling policy. Compare the non-self-normalized SMC estimator with crude MC.

| Design | Config and replication | Which populations supply the estimates |
|---|---|---|
| Fixed terminal tilt | [Particle-scaling YAML](../../configs/mdlm_owt_particle_scaling.yaml). $N=128,160,192,256$, two independent tilt settings per size, 24 runs per setting, **192 runs** total. | Terminate at 0.5 for $q\ge0.99$ and $q\ge0.995$, or at 2.5 for $q\ge0.999$. Both terminal populations also evaluate the other thresholds for secondary analysis. |
| Lower particle budget | [Low-particle YAML](../../configs/mdlm_owt_low_particle_budget.yaml). $N=64,96$, 24 ladders per size, **48 ladders** total. | Continue one ladder to 3 per seed. Evaluate $q\ge0.99$ at 0.5, $q\ge0.995$ at 1, and $q\ge0.999$ at 0.5 and 3, using the population after the fourth MH move. |

A separate run ending at a given tilt skips terminal resampling. A continuing ladder can resample at that same tilt before moving on. Consequently the earlier pilot checkpoints and the low-budget checkpoints are distinct from independently terminated populations. Preserve this distinction when comparing the campaigns. Use cumulative synchronized times and native transition counts at the actual checkpoint.

## Matched crude controls and independent references

Both campaigns reuse **121,355 native crude trajectories** from the completed [accuracy-budget pilot](ACCURACY_VS_BUDGET.md). The fixed-tilt study uses disjoint blocks of **5,056** paths across replicate ranks. Within a rank, take nested prefixes across settings. The lower-budget comparison likewise uses disjoint rank blocks and nested prefixes across checkpoints. Set the prefix length to the floor of measured cumulative SMC seconds divided by calibrated crude seconds per path on the same physical GPU with four CPU cores. Report a budget that exceeds its block. The low-budget records also retain exact native-transition budgets.

The reference counts at $q\ge0.99,0.995,0.999$ are **444/200,000**, **334/298,800**, and **26/200,000**, respectively. The accuracy-budget crude streams differ from the gradual-budget streams used for the 0.995 reference. Treat the references as finite IID estimates.

## Estimation and reporting

Reconstruct $\ell=\log\widehat Z+\operatorname{logsumexp}_{i\in A}(\log\overline W_i-\lambda\Phi_i)$ from the saved population. For $\delta=\ell-\log p_{\rm ref}$, compute the log-relative error $\log_{10}(1+|\exp(\delta)-1|)$ as $\delta/\log(10)$ for $\delta\ge0$, and $\operatorname{log1p}(-\operatorname{expm1}(\delta))/\log(10)$ otherwise. Keep zero estimates, represented by $\ell=-\infty$ in log space.

Report ratios, absolute and log-relative errors, relative standard deviation, mean and median, zero counts, and counts within a factor of ten. Group by size and tilt. The lower-budget study's working scale-recovery criterion was **at least 12 of 24 runs** within a factor of ten, and it expressed the crude budget as a fraction of $1/p_{\rm ref}$. These are conventions of that study. Its recorded checkpoint mapping remains distinct from later choices of which checkpoints to feature.

## Reproduction entry points

Both designs use [the particle-scaling worker](../scripts/slurm_avon_particle_scaling.sh) and `same_run_reuse run-fresh` with a checksummed source bundle, original ladder calibration, and three workers. The fixed-endpoint worker handles eight replicates per setting. The historical fixed-endpoint submission requested six hours, with a worker limit of 21,000 s. The smaller-ladder submission used `STUDY_CONFIG=mdlm_owt_low_particle_budget.yaml`, `RESULT_NAME=camera_ready_low_particle_budget`, and `SOFT_MAX_SECONDS=6900` with a two-hour allocation. Both used one GPU and four CPUs per run, with 20 GB host RAM per worker.

Use the [Avon runbook](AVON_RUNBOOK.md) for setup. The result bundles retain per-seed populations, worker manifests, CSVs and completion checks. [Particle-scaling plots](../plots/plot_particle_scaling.py) and [combined-budget analysis](../analysis/combine_particle_budgets.py) read saved outputs.
