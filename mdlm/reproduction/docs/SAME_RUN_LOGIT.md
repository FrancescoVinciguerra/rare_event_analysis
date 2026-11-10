# Continuing SMC ladders: log probability, MC reuse, and logit

## Shared design

The [original study](../../configs/mdlm_owt_same_run_ladder.yaml) and the [logit follow-up](../../configs/mdlm_owt_same_run_logit.yaml) use one $N=128$ ladder per seed, with **12 seeds** $2{,}000{,}000{,}000+1{,}000{,}000i$ for $i=0,\ldots,11$. The ladder goes from 0 to 8 in half-steps, with four RollBack–MH moves per level. Each seed yields **65 populations**: the initial state and one after each of 64 moves. The native path costs 33 transitions, including the terminal categorical draw.

Both events, $q\ge0.99$ and $q\ge0.999$, use the same populations and finite IID references of 444/200,000 and 26/200,000. Weight once per level, update $\widehat Z$ before mutation, use systematic resampling when ESS/$N<0.5$, and skip terminal resampling. The recorded populations are the ones actually used in the continuing run. The two estimators use the current tilt and are reported separately at each checkpoint, without terminal copies, averaging over levels or stopping on discovery.

One event particle identifies a positive checkpoint. Compare the 16 end-of-level populations across independent seeds. Label positive-only and all-run views, show zooms and full tails, count zeros, and distinguish mean from median. Also retain the first-event checkpoint and all per-move diagnostics, including times, native transitions, acceptance, ESS, reverse weights, genealogy and normalizers. Checkpoints within a seed are dependent.

## Execution and baseline history

| Stage | Baseline policy | Recorded outcome |
|---|---|---|
| Original $\Phi=\log q$, [job 2079596](../../artifacts/same_run_ladder_original_job_status.json) | Fresh crude baselines planned at end-of-level checkpoints and at first discovery. Fix the count before drawing samples using independent calibration, and measure actual time on the same GPU. | Cancelled after 01:58:51 during baseline generation. The monitor recorded three completed SMC ladders, three terminal crude baselines, 40 checkpoint crude baselines, zero complete comparison runs, and no result archive. The completed SMC states and crude draws were preserved. |
| Resume, [job 2079904](../../artifacts/same_run_ladder_job_status.json) | Run the nine missing SMC seeds and reuse saved IID draws for all checkpoints. No new crude warm-up or calibration. | Completed all 12 ladders in 00:42:49. The resume verified the original source hashes and retained existing populations, crude draws and manifests. |
| Logit, [job 2081343](../../artifacts/same_run_logit_job_2081343_status.json) and [job 2081350](../../artifacts/same_run_logit_job_status.json) | Repeat the 12 seeds with only the potential changed. Reuse the same crude pool, ordering and seed blocks. The prefix lengths can differ because SMC runtimes differ. | First attempt failed in 45 s before model load. The retry completed all 12 ladders and its scientific CSVs. The batch exit was 1 because Matplotlib was missing during plotting. |

The reused pool is **192 baseline files with 354,420 IID trajectories** from the gradual-budget study. The independent 200,000-draw reference stays separate. Each planned seed has a fixed disjoint block of **29,535** trajectories in deterministic file order. A checkpoint uses the first

$$
n_{\rm MC}=\left\lfloor
  \frac{\text{measured cumulative SMC seconds}}
       {\text{calibrated crude seconds per path}}
\right\rfloor
$$

draws in that block. If the requested count exceeds its block, the code raises an error. Prefixes are nested within a seed and disjoint across seeds. The calibration applies to the same physical GPU and four-CPU allocation. These are **calibrated time equivalents**, distinct from the original fresh crude runs with measured wall times.

All three stages used three GPUs for independent seeds, with one GPU and four CPUs per estimate and 60 GB host RAM for the job. The original job capped OMP, MKL, OpenBLAS and NumExpr at four threads per task. It allowed 12 hours, or 36 allocated GPU-hours. The resume and logit stages allowed two hours on the original node. Keep warm-up and calibration separate from per-estimate time.

## Potential change and validation

The original `smc.phi: log_probability` uses $\log q$. The `logit` option computes $\log(q/(1-q))$ from the difference of the two classifier logits, without clipping probabilities or inserting epsilon. The score $q$ and event definition stay separate. The new potential is used consistently in

$$
G_k=\exp((\lambda_k-\lambda_{k-1})\Phi),\qquad
\alpha_k=\min(1,\exp(\lambda_k(\Phi'-\Phi))),\qquad
\text{reverse factor}=\exp(-\lambda_k\Phi).
$$

Mutation leaves weights fixed. The observer uses sigmoid for logit and exp for log-probability, with a timed recheck of the original classifier probability near event thresholds. Shared seeds allow a paired comparison, although paths diverge once selection differs. Equal numerical tilts have different strength under the two potentials.

The first logit job picked up an older package of the same name through pytest's search path. The corrected job explicitly uses `pytest_snapshot.ini` and the frozen source root, checks the imported package path, and passed 18 local checks under simulated SLURM settings. The failed source snapshot was preserved. The retry's figures were generated locally from the saved CSVs, with no new SMC or MC. The current batch script defers rendering if Matplotlib is unavailable.

Offline validation checked **1,560 checkpoint/event pairs**, including normalizers, weights, native counts and crude prefixes against the frozen pool. The [results index](../../../shared/RESULTS_INDEX.md) links the report, checkpoint estimates and validation. The main outputs include `checkpoint_runs.csv`, `first_event_runs.csv`, `timing_steps.csv` for the logit run, `saved_MC_blocks.json` and `comparison_policy.json`.

## Reproduction entry points

- The original [submitter](../scripts/submit_avon_same_run_ladder.sh) runs the [ladder worker](../scripts/slurm_avon_same_run_ladder.sh) with `accuracy_budget run --max-seconds 41400 --workers 3`. The [resume worker](../scripts/slurm_avon_same_run_reuse.sh) uses `same_run_reuse run`.
- The [logit worker](../scripts/slurm_avon_same_run_phi.sh) uses `same_run_reuse run-fresh`, with `--max-seconds 6300 --workers 3` and the original ladder as the calibration source. Its frozen bundle and hashes identify the executed code. Use the [Avon runbook](AVON_RUNBOOK.md) for cluster setup.
- The [offline verifier](../analysis/verify_same_run_ladder_results.py) accepts `--results` for the logit run, `--study` for its YAML, and `--mc-pool` for the saved gradual-budget pool. It reads saved data without loading models.
