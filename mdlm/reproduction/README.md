# Reproducing MDLM campaigns

Use these files to reproduce a named study, rebuild plots, or submit a Slurm
job. To run a single estimate, use the [MDLM quick start](../README.md).

| Folder | Contents |
| --- | --- |
| [docs/](docs/) | Derivations, the Avon runbook, and the protocol for each named campaign |
| [scripts/](scripts/) | Slurm launchers, environment checks, and the CPU timing pilot |
| [analysis/](analysis/) | Recompute comparisons and summaries from saved populations or run-level CSVs |
| [plots/](plots/) | Render the figures from analysis results |

For the algorithm, start with [trajectory RollBack–MH](docs/TRAJECTORY_RESAMPLE_MOVE.md).
For cluster execution, start with the [Avon runbook](docs/AVON_RUNBOOK.md).
The remaining documents cover specific studies and estimator variants.

See [the kernel and estimator](docs/TRAJECTORY_RESAMPLE_MOVE.md) for the derivation.

Run commands from the repository root. The import name remains
`diffusion_toxicity_smc_experiment`; only its source location changed.

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

`mdlm/configs/mdlm_owt_unconditional_lambda_calibration.yaml`
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

`mdlm/configs/mdlm_owt_unconditional_estimation_template.yaml`
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
  --config mdlm/configs/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current \
  --target-samples 2

python -m diffusion_toxicity_smc_experiment.score_direct \
  --config mdlm/configs/mdlm_owt_unconditional_smoke_technical.yaml \
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

The tilted sampler is an explicit parameter of the complete probability
estimator. `fk_bootstrap` is the preserved FK-style baseline. The alternative
`trajectory_resample_move` samples full native MDLM paths, tempers each path
from zero to its component lambda, and applies exact Metropolis-Hastings
rejuvenation by returning to checkpoints actually visited by that trajectory.
It does not change token logits, select token positions, remask artificial
subsets, or introduce a new decoder policy. See
`docs/TRAJECTORY_RESAMPLE_MOVE.md` for the derivation and configuration.

`trajectory_waste_free` uses the same targets and exact rollback-MH kernel but
represents every tempering level with `M` chains of `P` retained states, where
`N=M*P`.  All retained states contribute to the next normalizer increment and
seed resampling.  See `docs/WASTE_FREE_TRAJECTORY_SMC.md`.

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
  --config mdlm/configs/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current

python -m diffusion_toxicity_smc_experiment.compare \
  --config mdlm/configs/mdlm_owt_unconditional_smoke_technical.yaml \
  --output-root mdlm/artifacts/current
```

Component artifacts retain terminal text/tokens, rewards, event scores,
forward and reverse weights, ESS, genealogy, runtime, NFE, and peak CUDA
memory. Reports distinguish tilted components from diagnostic components and
contain no prompt aggregation.

## Serious three-sampler comparison

`configs/mdlm_owt_unconditional_serious_comparison.yaml` fixes a resumable
comparison of the three available tilted samplers without changing the decoder,
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

Avon submission is deliberately stage-by-stage:

```bash
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh method-validation
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh direct-reference
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh score-reference
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh smc-grid
bash mdlm/reproduction/scripts/submit_avon_serious_comparison.sh report
```

These commands are dry-runs unless `--submit` is appended. `method-validation`
runs both N=16 and N=64 for all three real GPU kernels with two independent
runs before the long grid. Each invocation can submit at most one job and no
dependency chain is created. Direct shards, score shards, completed SMC
components, and complete grid points are reused on resume.

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
