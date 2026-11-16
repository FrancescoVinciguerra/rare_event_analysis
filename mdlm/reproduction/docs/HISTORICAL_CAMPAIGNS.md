# Earlier MDLM campaigns

These methods and commands describe earlier MDLM experiments. The current paper campaign is in the [reproduction guide](../README.md).

This package tests the rare-event estimator on the unconditional distribution
of the frozen MDLM-OWT model. It does **not** reproduce the prompt-conditioned
FK Steering experiment and its result must not be compared with FK's reported
prompt-conditioned toxicity rate.

## Base distribution

The decoder starts from

```text
Y_0 = [MASK]^128
```

with no BOS, EOS, prompt, or clamped prefix. This matches MDLM commit
`c112c526d193436838c98d81455ee51f90309470`: `Diffusion._sample_prior`
returns `mask_index * ones(*batch_dims)`, and the diffusion `_sample` method
uses it for the full configured model length. The BOS insertion in that source
belongs only to the separate autoregressive sampler.

The decoder manifest records `generation_mode: unconditional`,
`fixed_prefix_token_ids: []`, and `prompt: null`. Changing from a fixed-prefix
decoder to unconditional generation necessarily changes `decoder_id`, so old
prompt-conditioned artifacts cannot be reused.

The native SUBS DDPM propagation remains stochastic. Any masks remaining after
the last DDPM step are removed by a categorical draw from the model
distribution (`final_noise_removal_method: stochastic_categorical`), not by
argmax. The FK backend follows the original MDLM FK Steering implementation:
at each DDPM update it draws four independent categorical predictions from
the model's current `p(x_0 | x_t)`, copying already revealed tokens. At reward
checkpoints all four temporary texts are scored and their log-probability
rewards are combined with `logmeanexp`, that is, the logarithm of their mean
positive-class probability. These predictions never replace the propagated
particle. Trajectory backends instead score their genuinely sampled terminal
paths and do not use this intermediate-text mechanism.

The first estimative benchmark fixes this decoder at 32 DDPM steps. It estimates
`P_{X ~ P0^(32)}(E(X)=1)` and must not be described as the standard 1000-step
MDLM probability. A later 1000-step benchmark requires a distinct `decoder_id`
and its own IID direct baseline.

## Toxicity event

The classifier and RoBERTa retokenization/truncation convention are inherited
from the FK code, but their input is generated text only:

- reward: positive-class log probability on the first configured MDLM tokens;
- event: classifier `argmax == 1` after the generated text is retokenized by
  pinned `roberta-large` and truncated to 50 tokens.

The classifier, revisions, tokenization, truncation, text scope, and event rule
form the `evaluator_id`.

## Terminal-lambda calibration

`mdlm/configs/legacy/mdlm_owt_unconditional_lambda_calibration.yaml`
runs separate SMC
components at terminal lambdas 0.5, 1, 2, and 4 with 64 particles. Lambda zero
is a separate base-sampler diagnostic. Calibration components are never
aggregated and their mean is not an estimator.

For a component ending at `lambda_j`, checkpoint levels are 4, 8, ..., 32 and

```text
beta_j,t = lambda_j * t / 32.
```

Levels 4 through 28 use the four disposable multinomial `x_0` predictions from
the original FK MDLM procedure. Level 32 is evaluated on the true terminal
stochastic completion.
The incremental log potential is

```text
beta_j,t * R_t - beta_j,previous * R_previous,
```

so the full product telescopes to `exp(lambda_j * R_32)`. The calibration
artifacts keep reverse-SNIS ESS, genealogical diversity, resampling, reward,
classifier, runtime, NFE, memory, and finite-value diagnostics separate for
every lambda.

`mdlm/configs/legacy/mdlm_owt_unconditional_estimation_template.yaml`
is deliberately
provisional. Its terminal lambdas must be reviewed after calibration. Only
estimation components use the uniform component average; lambda zero is
excluded and particles are never pooled across components.

## IID direct baseline

Direct samples are IID draws `X_i ~ P0`. The estimator is

```text
p_hat_MC = (1/N) sum_i 1_E(X_i).
```

The report includes the binomial standard error and Wilson interval. There is
no prompt index, prompt mixture, stratified variance, or macro-average.

Generation and scoring remain separate and resumable:

```bash
python -m diffusion_toxicity_smc_experiment.generate_direct \
  --config mdlm/configs/legacy/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current \
  --target-samples 2

python -m diffusion_toxicity_smc_experiment.score_direct \
  --config mdlm/configs/legacy/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current
```

Artifacts are append-only:

```text
artifacts/current/mdlm_owt/direct/<decoder_id>/
  manifest.json
  generations/shard_<first>_<last>.jsonl
  evaluations/<evaluator_id>/
    manifest.json
    scores_<first>_<last>.jsonl
    summary.json                 # immutable first summary for legacy readers
    summaries/summary_<N>.json   # append-only snapshots as the run grows
```

Rows have `conditioning_mode: unconditional` and never contain a synthetic
prompt. Resume scans global sample indices, rejects duplicate IDs/indices, and
generates only missing indices up to `target_samples`.

## SMC estimator

Use the [MDLM estimator](../../README.md#single-estimate) for a single probability estimate. When changing RollBack–MH, preserve these invariants:

- Keep the realized prefix and regenerate the suffix with the native decoder. The acceptance ratio assumes a state-independent rollback-depth law. A state-dependent law requires the forward/reverse depth-selection ratio.
- Use the same potential in weighting, MH and reverse correction. The event still uses the classifier score. Update weights and the normalizer before resampling or mutation. The scheduled path skips terminal resampling. A checkpoint from a continuing ladder may have resampled at the same tilt.
- In the scheduled path, use the current tilt and weighted population. The non-self-normalized estimate uses the estimated normalizer. Reverse-SNIS uses the ratio of reverse-weighted sums.

Focused checks cover [schedule equivalence and normalizers](../../tests/test_exploratory.py), [detailed balance and reverse correction](../../tests/test_potentials.py), and [checkpoint behavior](../../tests/test_population_checkpoints.py).

For tilted component `j`, reverse-SNIS is normalized separately:

```text
          sum_i W_j,i exp(-lambda_j r(X_j,i)) 1_E(X_j,i)
p_hat_j = -------------------------------------------------
              sum_i W_j,i exp(-lambda_j r(X_j,i))
```

Only after every ratio is formed do we aggregate:

```text
p_hat = sum_j a_j p_hat_j.
```

The generic aggregation function accepts any non-negative simplex vector.
Current configurations select `component_weight_policy: uniform`, requiring
`a_j=1/J`. Smoke profiles use tilted lambdas `[5, 10]` and weights `[0.5,
0.5]` only to exercise the convex-combination code; these values are not
claimed to be optimal.

`lambda=0` is listed under `diagnostic_lambdas`, saved separately, checked
against the unconditional base sampler, and excluded from the final convex
combination. Including it later requires an explicit entry in `tilted_lambdas`
and a corresponding weight.

```bash
python -m diffusion_toxicity_smc_experiment.run_smc \
  --config mdlm/configs/legacy/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current

python -m diffusion_toxicity_smc_experiment.compare \
  --config mdlm/configs/legacy/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current
```

Component artifacts retain terminal text/tokens, rewards, event scores,
forward and reverse weights, ESS, genealogy, runtime, NFE, and peak CUDA
memory. Reports distinguish tilted components from diagnostic components and
contain no prompt aggregation.

## Serious three-sampler comparison

`configs/legacy/mdlm_owt_unconditional_serious_comparison.yaml` records the historical
three-sampler comparison, including Waste-Free, without changing the decoder,
reward, event, lambdas, or uniform component weights:

```text
samplers:        FK bootstrap, trajectory resample-move, trajectory waste-free
particles:       64, 128, 256, 512
independent runs: 8 per sampler/budget
tilted components per run: lambda = 0.25, 0.5, 0.75, 1.0
direct reference: 200,000 IID samples, scored on the same first 100 MDLM tokens
```

Every tilted component has an independent seed; methods also use disjoint seed
ranges. Direct generation is append-only and uses an independent deterministic
seed for every shard. Existing 32-sample shards are retained, while the 200,000
extension uses 512-sample shards to avoid quadratic GPFS metadata and inventory
overhead. The result is a high-precision Monte Carlo reference with binomial
uncertainty, not an exact ground truth.

Classifier evaluation uses fixed `max_length` padding and pads the last
microbatch with discarded dummy rows. This makes the numerical reward
`phi(x)` independent of the other particles in a batch, which is required by
the trajectory-MH acceptance ratio. The policy is part of `evaluator_id`.

The report writes JSON and CSV tables plus SVG box plots for estimates,
absolute error, runtime, genealogical diversity, reverse ESS, and MH acceptance.
It also writes error-versus-runtime and error-versus-decoder-particle-step
frontiers. Equal-runtime and equal-decoder-work rows are selected from the
measured budget grid; classifier texts are reported separately rather than
hidden in an arbitrary combined cost.

The three-method suite is read-only. The saved results and original suite
identity remain available for reporting. The launcher still supports rebuilding
the IID reference in stages and reporting from completed artifacts:

```bash
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh direct-reference
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh score-reference
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh report
```

These commands are dry-runs unless `--submit` is appended. Each invocation can
submit at most one job. The original sampling code is in the frozen reviewer
bundle.

## Submission probability study

The submission-oriented experiment treats the complete rare-event probability
estimator as the contribution.  FK and trajectory-MH are interchangeable
tilted-population engines; Waste-Free is not part of this study.  The existing
200,000 IID completions and the completed eight-seed N=512 runs are reused as a
pilot.  Their stored continuous classifier score defines the nested events

```text
E_a(x) = 1{event_toxic_probability(x) >= a},
a in {0.5, 0.9, 0.99, 0.999}.
```

No completion is regenerated to change `a`.  For every lambda component, the
code reuses its terminal particles and reverse importance weights and forms a
separately normalized reverse-SNIS estimate.  The pilot fits, independently for
each sampler and threshold, regularized inverse-variance weights

```text
v_tilde_j = max((1-rho) s_j^2 + rho mean_k(s_k^2), floor)
a_j       = (1/v_tilde_j) / sum_k(1/v_tilde_k).
```

Here `s_j^2` is the variance across the eight old, independent pilot runs,
`rho=0.25`, and the floor is 5% of their mean variance.  These are
minimum-pilot-variance weights under independent components; they are not
claimed to minimize finite-sample MSE.  They are frozen before test artifacts
are read.

The long test uses 32 entirely new runs for FK and 32 for trajectory-MH, each
with four independent N=512 lambda populations.  Uniform and pilot-fitted
weights are applied to the same new component estimates, giving a paired and
leakage-free comparison.  The sampling configuration itself remains uniform;
weight optimization is a report-level aggregation choice and does not change
the decoder, SMC transition, reward, event score, or random populations.

Run one stage at a time.  Every command is a dry-run until `--submit` is added:

```bash
bash mdlm/reproduction/scripts/submit_avon_submission_probability.sh pilot
bash mdlm/reproduction/scripts/submit_avon_submission_probability.sh long-smc
bash mdlm/reproduction/scripts/submit_avon_submission_probability.sh report
```

The CPU-only pilot must complete first.  Inspect `direct_thresholds.csv` and
`pilot_weights.csv` before submitting the GPU stage; this determines whether
the most extreme threshold has enough IID discoveries to be interpretable.
The final CPU report writes run/component CSV files, a JSON summary, and paired
uniform-versus-pilot-weight SVG box plots for every threshold.  It also reports
runtime, decoder particle-steps, classifier calls, reverse ESS, genealogical
diversity, and peak CUDA memory without folding them into one arbitrary cost.

## Earlier 288-run Avon experiment

The [archived campaign configuration](../../configs/legacy/mdlm_owt_paper_final_scaling.yaml)
defines the 12 particle sizes, 24 seeds per size, the tilt grid, and the
sweep and resampling policy. The four observation tilts for classifier score
thresholds 0.99, 0.995, 0.999 and 0.9995 are 0.5, 1, 2.5 and 6.5,
respectively. Each observation uses the population after the fourth MH
sweep and its own cumulative synchronized runtime.

### Crude Monte Carlo reference

The [pooled reference counts](../../artifacts/current/camera_ready_progressive_tail_particle_scaling/across_budget/pooled_reference_figures/pooled_crude_reference.json)
come from 975,775 native generations across four cohorts.

| Cohort | Samples |
| --- | ---: |
| Original direct generation | 200,000 |
| Gradual-budget campaign | 354,420 |
| Accuracy-budget pilot | 121,355 |
| Crude-tail reference campaign | 300,000 |

For each threshold, divide the number of scores at or above the threshold by
975,775. The 475,775 scores from the gradual-budget and accuracy-budget
cohorts also supply the matched-MC baseline. MC counts use throughput
calibrated on the same physical GPU with four CPUs. Replicate ranks use
disjoint blocks, with nested prefixes across their budget points.

[Direct generation](../../src/diffusion_toxicity_smc_experiment/generate_direct.py)
and [scoring](../../src/diffusion_toxicity_smc_experiment/score_direct.py) provide
the native MC entry points. The archived audit, `mdlm/reproduction/analysis/count_pooled_crude_thresholds.py`,
recounts the three saved cohorts and takes the original direct cohort's
counts and seed inventory from the Avon audit. An offline recount directly
from all four raw cohorts and a single regeneration recipe for their
original batches are pending.

### Rebuild the comparison on CPU

From the repository root, unpack the [frozen source bundle](../frozen/paper_final_scaling.tar.gz)
and run the report on the saved artifact tree.

```bash
mkdir -p /path/to/frozen
tar -xzf mdlm/reproduction/frozen/paper_final_scaling.tar.gz -C /path/to/frozen
export PYTHONPATH="/path/to/frozen/src:$PWD/shared"
study=/path/to/frozen/configs/mdlm_owt_paper_final_scaling.yaml
python -m diffusion_toxicity_smc_experiment.particle_scaling report \
  --study "$study" --output-root /path/to/artifacts/current
```

The input tree contains `camera_ready_paper_final_scaling` with 288 saved
populations and timing calibrations, and the `camera_ready_gradual_budget`
and `camera_ready_accuracy_budget_pilot` directories with manifests and
baseline scores. The report writes `runs.csv`, `summary.csv`,
`comparison_policy.json` and `report_status.json`. The figure export from
the final `runs.csv` is pending.

### Rerun RollBack–SMC on Avon

Job `2149818` combines 96 original terminal runs at N = 320, 384, 448 and
512 from job `2134669` with 192 new runs. The first 12 seed values at each
of N = 128 and N = 256 come from earlier campaigns and are run with the
terminal policy at 6.5. The `reuse_manifest.json` identifies each copied
population and its SHA-256 hash. The job writes the campaign archive
`camera_ready_paper_final_scaling_results.tar.gz`.

Use the [Avon runbook](AVON_RUNBOOK.md) to set up the GPU environment.
Set `EXPERIMENT_DIR` to the unpacked bundle, `OUTPUT_ROOT` to the artifact
tree and `CALIBRATION_ROOT` to the saved `camera_ready_same_run_ladder`
directory. Preview before submitting.

```bash
bash "$EXPERIMENT_DIR/scripts/submit_avon_paper_final_scaling.sh" --dry-run
bash "$EXPERIMENT_DIR/scripts/submit_avon_paper_final_scaling.sh" --submit
```

For a full GPU rerun, set `PAPER_FINAL_REUSE=0` and use a separate output
root with the same saved MC pools. Use `--resume` for an incomplete job.

## Historical staged launchers

Review the exact account, partition, resources, time limit, config, paths and commands printed by the dry-run. Obtain authorization for the specific submission. A scheduler limit is an allocation bound. Use observed timings for forecasts.

| Workflow | Launcher and stage rules |
|---|---|
| Smoke | Smoke launcher (`mdlm/reproduction/scripts/submit_avon_smoke_pipeline.sh`). Use `technical --dry-run` or `scientific --dry-run` to preview the pipeline. The scientific profile needs a partition that supports the printed four-hour direct and twelve-hour SMC requests. |
| First 32-step experiment | First-experiment launcher (`mdlm/reproduction/scripts/submit_avon_first_experiment.sh`). Start with `direct-256`, then score and review before calibration, reporting or extending the same direct run to 1,000. The helper has no 10,000-sample stage. The [historical calibration notes](#terminal-lambda-calibration) give the scientific sequence. |
| Historical three-sampler comparison | Serious-comparison launcher (`mdlm/reproduction/scripts/submit_avon_serious_comparison.sh`) supports IID reference generation, scoring, and CPU reporting from saved SMC results. |
| Submission probability | Submission launcher (`mdlm/reproduction/scripts/submit_avon_submission_probability.sh`). Run `pilot`, `long-smc` and `report` separately. Verify the pilot's source IDs, seed separation, finite estimates and IID discovery counts before authorizing the long stage. |

The non-smoke launchers in the table default to preview and accept `--submit` for one selected stage. The smoke launcher uses a different interface. Preview the selected stage with `--stage STAGE --dry-run`. When authorized, use `--stage STAGE` without `--dry-run`. The former multi-job `--submit` mode is disabled for this launcher. Smoke order is `prefetch → direct → score → smc → compare`. For a dependent stage, give the preceding job ID with `--afterok` where the submission limit permits it, or use `--no-dependency` after verifying completion. Prefetch has no predecessor.

For example, this command only previews the technical prefetch stage:

```bash
bash mdlm/reproduction/scripts/submit_avon_smoke_pipeline.sh technical --stage prefetch --dry-run
```

The submission study's documented allocations were two CPU cores and 4 GB for the pilot (20 minutes) and report (30 minutes), and one GPU, four CPUs and 20 GB for the long stage (two days). The planning estimate for the long runs was about 23 GPU-hours in total. Verify the current launcher and cluster values before use.

## Retired checkout files

The archived jobs retain their saved results, manifests and frozen execution
sources. To verify one of those jobs, use the code bundle from its manifest.

The [historical source supplement](../frozen/historical-local-sources.zip)
holds the 13 unique working-tree versions retired during the checkout
cleanup and the crude-tail launcher recovered from a local snapshot.
Its `INDEX.json` identifies the original path, SHA-256 and
recovery source for each of the 65 retired files. Other entries refer to
existing frozen files or Git commit `86bd1855a5379bc9d6b2a26195d0e361eab5ee3f`.
Older YAMLs used by regression tests are grouped under
[`configs/legacy/`](../../configs/legacy/).

To reconstruct the old checkout layout for a historical command, use a scratch
copy. Restore each source to its original path from the source listed in
`INDEX.json`. The Git entries can be retrieved with
`git show <commit>:<path>`. For example, to regenerate the earlier threshold
0.995 tables, restore `add_threshold_0995.py`, `analyze_relative_variance.py`
and `plot_same_run_ladder.py` in the paths below, then run
`python mdlm/reproduction/analysis/add_threshold_0995.py` from the scratch
repository root with the saved artifacts. Copy any required YAMLs from
`configs/legacy/` to their original `configs/` paths for scripts that
expect the historical layout.

**`mdlm/configs/`**

- `mdlm_owt_gradual_budget.yaml`
- `mdlm_owt_strong_tilt.yaml`
- `mdlm_owt_unconditional_final_analysis.yaml`
- `mdlm_owt_wall_clock_followup.yaml`

**`mdlm/reproduction/analysis/`**

- `add_threshold_0995.py`
- `analyze_deeper_tilt_budget.py`
- `analyze_particle_scaling_lambda1.py`
- `analyze_relative_variance.py`
- `analyze_threshold_09995.py`
- `combine_particle_budgets.py`
- `compare_averaged_runs.py`
- `count_pooled_crude_thresholds.py`
- `extend_progressive_across_budget.py`
- `offline_terminal_tail_analysis.py`
- `pairing_sensitivity.py`
- `plot_budget_caption_pairs.py`
- `plot_budget_compact.py`
- `plot_budget_standalone.py`
- `plot_comparison_designs.py`
- `plot_publication_across_budget.py`
- `verify_same_run_ladder_results.py`

**`mdlm/reproduction/plots/`**

- `offline_publication_figures.py`
- `plot_accuracy_budget.py`
- `plot_averaged_runs.py`
- `plot_gradual_budget.py`
- `plot_particle_scaling.py`
- `plot_same_run_ladder.py`
- `plot_selected_tilts_by_budget.py`
- `plot_wall_clock_followup.py`

**`mdlm/reproduction/scripts/`**

- `cpu_timing_pilot.py`
- `slurm_avon_accuracy_budget.sh`
- `slurm_avon_camera_ready_low_budget.sh`
- `slurm_avon_cpu_timing_pilot.sh`
- `slurm_avon_crude_tail_reference.sh`
- `slurm_avon_exploration_v2.sh`
- `slurm_avon_final_component_export.sh`
- `slurm_avon_final_experiment.sh`
- `slurm_avon_gradual_budget.sh`
- `slurm_avon_offline_publication_figures.sh`
- `slurm_avon_particle_scaling.sh`
- `slurm_avon_prefetch_models.sh`
- `slurm_avon_same_run_ladder.sh`
- `slurm_avon_same_run_phi.sh`
- `slurm_avon_same_run_reuse.sh`
- `slurm_avon_serious_smc.sh`
- `slurm_avon_smoke_compare.sh`
- `slurm_avon_smoke_direct.sh`
- `slurm_avon_smoke_score.sh`
- `slurm_avon_smoke_smc.sh`
- `slurm_avon_strong_tilt.sh`
- `slurm_avon_terminal_tail_analysis.sh`
- `slurm_compare.sh`
- `submit_avon_accuracy_budget.sh`
- `submit_avon_camera_ready_low_budget.sh`
- `submit_avon_exploration_v2.sh`
- `submit_avon_final_experiment.sh`
- `submit_avon_gradual_budget.sh`
- `submit_avon_paper_final_scaling.sh`
- `submit_avon_same_run_ladder.sh`
- `submit_avon_smoke_pipeline.sh`
- `submit_avon_strong_tilt.sh`

**`mdlm/src/diffusion_toxicity_smc_experiment/`**

- `component_export.py`
- `crude_tail_reference.py`
- `paper_final_scaling.py`
- `prefetch_models.py`
