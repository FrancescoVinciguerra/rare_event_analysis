# RollBack SMC budget pilots and schedule exploration

These studies share the [unconditional 32-step MDLM profile](../../configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml), terminal categorical completion, first-100-token evaluator, and potential $\Phi=\log q$. The non-self-normalized SMC and reverse-SNIS estimates use the same population. Weighting and the normalizer update precede optional systematic resampling and MH, with terminal resampling skipped. The events $q\ge 0.9$ and $q\ge 0.99$ have finite IID references of 1,516/200,000 and 444/200,000, respectively. The corresponding approximate Wilson 95% intervals are [0.007209, 0.007970] and [0.002023, 0.002436]. The thresholds are classifier scores.

| Stage | Configuration and comparison | Recorded execution |
|---|---|---|
| Early budget pilot | [Low-budget YAML](../../configs/mdlm_owt_camera_ready_low_budget.yaml). Five population sizes, eight planned runs per size. The initial plan matched native transitions. The delivered runner also supported measured-time matching with independent crude streams and the same SMC populations. | [Job 2075174](../../artifacts/camera_ready_job_status.json) timed out at 02:00:06 with 30 of 40 SMC populations completed. |
| Schedule and mutation exploration | [Exploration YAML](../../configs/mdlm_owt_exploration_v2.yaml). Half-step ladder through tilt 3. One fresh transition-matched crude estimate per SMC run. | [Job 2076203](../../artifacts/camera_ready_exploration_v2_job_status.json) completed 24 runs in 01:18:52. |
| Measured-time follow-up | [Follow-up YAML](../../configs/mdlm_owt_wall_clock_followup.yaml). Reuse those 24 SMC populations and their measured durations. Generate independent crude controls at the matched time. | [Job 2076362](../../artifacts/wall_clock_followup_job_status.json) completed 24 crude controls in 00:41:32. |

The job records give the outcomes above. The original preparation notes described proposed submissions. Allocations were one GPU, four CPUs and 20 GB for the early pilot, and three GPUs, 12 CPUs and 60 GB for each later stage. The job limits were two hours, two hours and one hour, respectively. Each run used one GPU and four CPUs. See the [Avon runbook](AVON_RUNBOOK.md) for submission and resume procedures.

## Early pilot: transition accounting and discovery budgets

Count 33 native transitions per initial path and $d+1$ per proposed rollback, including rejections. Keep DDPM-only counts, model forward calls, and classifier calls separate. For the early pilot, mean depth was 16.8 and there were 12 proposals per particle, so

$$
\mathbb E[\text{native transitions}]=N[33+12(16.8+1)]=246.6N.
$$

The transition-matched crude run uses $\lfloor\text{actual transitions}/33\rfloor$ complete paths. For the five sizes in the YAML, the expected costs were 3,945.6, 7,891.2, 15,782.4, 31,564.8 and 63,129.6 transitions. Before flooring, these correspond to about 119.56, 239.13, 478.25, 956.51 and 1,913.02 crude paths. At 100 and 200 crude paths the predicted probabilities of zero hits are about 47% and 22% for $q\ge0.9$, and 80% and 64% for $q\ge0.99$.

The historical `publication_figures_v2_20260823_final2/csv/direct_cost.csv` for 200,000 completions recorded 33,041.4424 s for generation and 1,217.6117 s for scoring, about 0.1712953 s per path. This excludes startup and queueing. Using the finite IID references, the mean waiting count is $1/p_{\rm ref}$, and a 95% chance of a hit requires $\lceil\log(0.05)/\log(1-p_{\rm ref})\rceil$ paths.

| Score threshold | Mean paths to first hit | Approximate time | Paths for 95% discovery | Approximate time |
|---|---:|---:|---:|---:|
| 0.9 | 131.93 | 22.6 s | 394 | 67.5 s |
| 0.99 | 450.45 | 77.2 s | 1,348 | 230.9 s |

These are design calculations from historical throughput. A hit establishes discovery, and an accurate probability estimate requires further evidence. The expected crude hit counts across the configured sizes range from 0.91 to 14.50 for $q\ge0.9$ and from 0.27 to 4.25 for $q\ge0.99$. The early job's 30 completed populations are partial pilot evidence.

## Exploration: half-step schedule and overlap

| Case | Particles | MH moves per level | Expected transitions | Expected crude paths before flooring |
|---|---:|---:|---:|---:|
| A | 64 | 4 | 29,452.8 | 892.5 |
| B | 64 | 8 | 56,793.6 | 1,721.0 |
| C | 128 | 4 | 58,905.6 | 1,785.0 |

A versus B varies mutation count, A versus C varies population size, and B versus C compares more mutations with more particles at similar expected cost. The eight runs per case were interleaved A/B/C, with sampler seed `1400000000 + 10000*case_index + 1000000*run_index` and separate crude stream `925081`. Assignment by run index modulo three gives workers 9, 9 and 6 runs, each balanced across cases. Estimates use tilt 3 and remain separate by case and seed.

An overlap check used 5,746 saved crude trajectories from the earlier $N=128$ setting. The ideal-target diagnostic $[\mathbb E_{\pi_a}G]^2/\mathbb E_{\pi_a}[G^2]$ uses moments of $q^\lambda$ under the base law. The first step from 0 to 0.5 gave about **5.3%** on that sample. This calculation describes target overlap. Finite-population ESS, genealogical dependence, and MH mixing require separate checks.

The explicit schedule implementation was checked against the original linear schedule with the same seeds, particles, weights, genealogy and acceptances. The run ledger stores each weighting input, log increment and $\log\widehat Z$ for recurrence checks, plus acceptance by level and depth, forward and reverse ESS, maximum weights, genealogy and reverse mass. Worker directories keep independent runs separate. The report checks study identity, source hashes and duplicates before merging worker tables.

## Measured-time controls

The follow-up reuses synchronized complete SMC times, averaging **156.76 s**, **302.25 s** and **311.76 s** for A, B and C. The control is assigned to the original physical GPU by comparing the SLURM GPU allocation with the saved warm-up record. Each GPU has a 16-path crude warm-up and three independent 128-path calibration batches. These are kept separate from the estimates.

Before drawing each fresh control, set $n_{\rm MC}=\lfloor t_{\rm SMC}/c\rfloor$, where $c$ is the median calibrated seconds per crude path. The follow-up uses stream `925082`, distinct from the transition controls and calibration. Measure actual time, save the ratio $t_{\rm MC}/t_{\rm SMC}$, and flag deviations above 5% without filtering or rerunning based on the estimate. The early pilot's delivered time-matched arm likewise used independent calibration and new draws, with the same SMC population as its transition-matched arm. Per-estimate SMC timing includes generation, scoring, weighting, resampling, MH and both reverse corrections. Startup, warm-up and file I/O are accounted for separately.

## Reports and reproduction

- Use the [early pilot launcher](../scripts/submit_avon_camera_ready_low_budget.sh), [exploration launcher](../scripts/submit_avon_exploration_v2.sh), or [follow-up launcher](../scripts/submit_avon_wall_clock.sh) with its linked YAML. The historical exploration worker used `--max-seconds 6600 --workers 3`. The preparation records preserve the original source packages and SHA-256 checks. The `plan --study` subcommands of `low_budget` and `exploratory` load no models.
- The corresponding offline plotters are [early pilot](../plots/plot_camera_ready_low_budget.py), [exploration](../plots/plot_camera_ready_exploration.py), and [follow-up](../plots/plot_wall_clock_followup.py). Use the saved `runs.csv`. Populations, baseline scores and per-run diagnostics remain available through the [results index](../../../shared/RESULTS_INDEX.md).
- The two metrics are $\log_{10}(1+|\widehat p/p_{\rm ref}-1|)$ and $\widehat p/p_{\rm ref}$. Retain zeros and full tails, show the zoom exclusions, distinguish the median line from the all-run mean diamond, and report sample counts. Boxes span quartiles and whiskers extend to 1.5 IQR. The early pilot used log-error zoom 0–0.4, with relative-error tick labels at 0%, 25%, 50%, 75%, 100% and 150%, and ratio zoom 0–2, with references at $\log_{10}2$ and 1, a ratio band 0.5–1.5, and a full ratio axis linear through 1 and logarithmic above it. Show finite-reference uncertainty and use the same vertical scale across events for each view. Keep cohorts, thresholds and matching policies separate.
- Reports can be rebuilt from saved rows, with checkpoint counts identifying partial runs. The early pilot's HTML embeds its figures, and the PNGs are rendered locally from the same rows. Selection after a pilot requires validation with new seeds.
