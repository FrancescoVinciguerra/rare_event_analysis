# Rare language-model completions with suffix SMC

## What problem are we solving?

Direct sampling from a language model rarely visits very unusual completions.
This repository studies whether a fixed compute budget can be directed toward
high Automated Readability Index (ARI) completions, and whether those tilted
samples still contain enough overlap to estimate tail probabilities under the
original model distribution.

The two questions are related but different:

1. Can the sampler discover and represent high-ARI completions?
2. Can self-normalized importance sampling (SNIS) reliably correct the final
   tilted population back to the base model?

The current experiments show useful high-ARI sampling in some configurations.
Tail estimates are only credible when independent-seed stability and corrected
importance ESS also support them. This repository does not claim that the
method is generally more efficient than Transition Path Sampling (TPS).

## Inspiration and starting point

[Rare Event Analysis of Large Language Models](https://arxiv.org/abs/2602.06791)
introduced the project problem, exponentially tilted completion targets, and
an annealed TPS/MBAR workflow. We retain the exponential tilt but investigate a
particle method on complete completions.

[Zanella (2020)](https://doi.org/10.1080/01621459.2019.1585255) motivated earlier
experiments with local informed proposals and balancing functions. The active
sampler does **not** use a locally normalized balancing function and is not
presented as an application of a theorem from that paper. It uses an asymmetric
language-model suffix proposal with an ordinary, directly derived MH correction.

The SMC and importance-sampling terminology follows standard Monte Carlo work,
including [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
These references provide context and inspiration; they do not replace checking
the exact kernel implemented here.

## What we changed

The active method keeps each particle as a complete, fixed-length generation.
Across a schedule of bias parameters it:

- mutates every particle by retaining a prefix and regenerating a suffix from
  the base LM at temperature one;
- applies the exact direct Metropolis-Hastings acceptance probability;
- updates incremental SMC weights in log-space;
- accumulates each tilted normalizing-constant estimate before weight
  normalization and resampling;
- resamples systematically when ESS is low, except at the final level;
- batches suffix generation by suffix length on the model device;
- uses the final weighted population for tilted expectations and SNIS tail
  estimates under the original model.

Legacy Barker, balanced-proposal, mixture-MCMC, and duplicate full-sequence SMC
implementations have been removed. They remain recoverable from Git history.

## Current algorithm

For a prompt `c`, completion `x`, and observable `phi`, the target is

```text
pi_lambda(x) proportional to p_M(x | c) exp(-lambda phi(x)).
```

Negative `lambda` favors larger values of `phi` (ARI in the original
experiments). At SMC transition `k`, mutation targets
`pi_(lambda[k-1])`; particles are then reweighted from `lambda[k-1]` to
`lambda[k]`. The direct-MH suffix acceptance is

```text
min(1, exp(-lambda * (phi(proposal) - phi(current)))).
```

This simplification is exact here because completion length is fixed, suffix
length selection is state-independent, and the proposal is exactly the base LM
conditional at temperature one, without top-k/top-p filtering or penalties.
The full derivation, detailed balance proof, SMC weights, and assumptions are in
[THEORY.md](THEORY.md).

## Why the current algorithm is correct

Correctness is established directly for this code, not inherited from a cited
algorithm. [THEORY.md](THEORY.md) writes the implemented suffix proposal,
computes its forward/reverse MH ratio, shows the LM cancellation, proves
detailed balance and stationarity, derives the SMC incremental weights, and
derives the SNIS correction including cancellation of `Z_lambda`.

Tests also compare the simplified acceptance with the unsimplified MH ratio and
check stationarity on a finite enumerable state space. These checks support the
derivation but do not by themselves guarantee efficient finite runs.

## Estimating tail probabilities

For one final weighted population, the SNIS correction back to the base model is

```text
log r_i = log W_i + lambda_final * phi_i
R_i     = exp(log r_i - logsumexp(log r))
P_hat_pM(phi >= a) = sum_i R_i 1{phi_i >= a}.
```

The output distinguishes final SMC ESS from corrected importance ESS. The first
diagnoses degeneracy while representing the tilted target; the second diagnoses
overlap after correcting back to `p_M`. An aggressive tilt can improve discovery
while making the corrected estimate unusable.

The tail-validation workflow combines independent capped-ARI terminal
populations as a normalized deterministic mixture. Its balance heuristic uses
both the final SMC weights and the estimated component normalizers; all events
are still evaluated with raw ARI. This is a plug-in SNIS estimator, not an
exactly unbiased finite-sample estimator. The direct formula is in
[THEORY.md](THEORY.md). Its primary reference can be generated once with the
same generation and scoring primitives used by SMC.

## Repository structure

```text
rea_llms/                              shared generation, ARI, diagnostics, and SNIS
suffix_direct_mh_smc_experiment/       active sampler and command-line runners
qwen_safety_smc_experiment/            Qwen3Guard proxy observable and safe smoke tests
notebooks/REA_Suffix_SMC_Quickstart.ipynb
                                       primary Colab validation workflow
official_annealed_tps_experiment/      optional adapter to the official TPS code
smc_tps_comparison/                    optional, separate SMC/TPS benchmark
notebooks/REA_SMC_vs_Official_TPS_Benchmark.ipynb
tests/                                 focused numerical and pipeline tests
results/                               curated summaries only
external/rea_of_llms_official/         pinned official repository submodule
```

The TPS benchmark is optional. The SMC and tail-validation runners do not import
or execute TPS, MBAR, or direct sampling.

## Colab quickstart

Open the primary notebook directly:

[Open REA Suffix SMC in Colab](https://colab.research.google.com/github/FrancescoVinciguerra/rea_of_llms_supplementary_code-main/blob/main/notebooks/REA_Suffix_SMC_Quickstart.ipynb)

The notebook has one parameter cell and is fixed to
`ARI_CONVENTION='official_public'`. Its two production switches,
`RUN_OFFICIAL_PUBLIC_DIRECT` and `RUN_OFFICIAL_PUBLIC_SMC`, are false by
default, so opening it never starts an expensive run. Run the direct phase
once, then the SMC phase; with both switches false the notebook reloads the
completed archives and performs analysis only. The separate `RUN_DRIVE_AUDIT`
switch performs read-only metadata inspection.

Only resumable scientific data are persisted under
`MyDrive/rea_official_public_validation`: `component_runs/*.npz` and the
reproducible IID direct ARI reference. The main table compares IID direct and
the SMC mixture at thresholds 6, 7, and 8. Component reverse diagnostics and
exactly two plots are displayed inline; no analysis CSV, PNG, or report is
written. The reason the released paper arrays are not used as a numerical
baseline is documented in
[`docs/DATA_REUSE_AND_ARI_AUDIT.md`](docs/DATA_REUSE_AND_ARI_AUDIT.md).

The optional comparison notebook is
[REA_SMC_vs_Official_TPS_Benchmark.ipynb](notebooks/REA_SMC_vs_Official_TPS_Benchmark.ipynb).
Clone submodules before that benchmark with `git submodule update --init`.

The separate safety-infrastructure smoke uses the public Qwen3Guard-Stream
proxy and does not run a scientific tail experiment:
[open the Qwen safety smoke in Colab](https://colab.research.google.com/github/FrancescoVinciguerra/rea_of_llms_supplementary_code-main/blob/main/qwen_safety_smc_experiment/notebooks/colab_smoke.ipynb).

## Running one experiment locally

Install the project and run one SMC population:

```bash
uv sync
uv run python suffix_direct_mh_smc_experiment/run_experiment.py \
  --device auto \
  --num-particles 64 \
  --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-1 \
  --mcmc-steps-per-level 5 \
  --suffix-sizes 1,2,4,8,16,32,64,100 \
  --seed 12345
```

Run independent seeds and estimate base-model tails:

```bash
uv run python suffix_direct_mh_smc_experiment/run_tail_estimation.py \
  --device auto \
  --seeds 12345,12346,12347 \
  --num-particles 256 \
  --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-0.75,-1,-1.5,-2 \
  --mcmc-steps-per-level 10 \
  --tail-thresholds 8,10,12,15
```

Use `--help` for output paths, resume behavior, suffix schedules, overlap
criteria, and terminal-lambda sweep options.

Build the one-time matching IID direct reference (the 200,000-sample default
is not run by tests or setup):

```bash
uv run python -m rea_llms.current_direct_reference \
  --device cuda \
  --num-samples 200000 \
  --batch-size 256 \
  --max-new-tokens 100 \
  --thresholds 6,7,8,10 \
  --output-dir /path/to/rea_tail_probability_validation/current_direct_reference
```

Validate a deterministic mixture against its matching IID direct reference:

```bash
uv run python suffix_direct_mh_smc_experiment/run_tail_validation.py \
  --device cuda \
  --seed-groups 200,300,400 \
  --num-particles 256 \
  --max-new-tokens 100 \
  --lambda-step 0.25 \
  --mcmc-steps-per-level 10 \
  --thresholds 6,7,8,10
```

## Outputs and minimal metrics

A single SMC run writes `summary.json`, `samples.jsonl`, and `report.md` under
the chosen output directory. The summary reports:

- weighted and unweighted final ARI statistics;
- raw and weighted tilted-target tail summaries;
- acceptance rate, resampling count, ESS/N, ancestry, and runtime;
- generated suffix-token count and peak CUDA memory when available.

Tail estimation adds per-threshold SNIS estimates under `p_M`, corrected
importance ESS/N, overlap flags, and independent-seed aggregate summaries.
The Colab notebook keeps these analyses in memory and displays them inline.
The command-line validator can still write CSV, report, and figure artifacts
when a persistent machine-readable audit is specifically needed.
Generated outputs are ignored by Git. Only compact results that support a stated
conclusion should be recorded in `results/EXPERIMENT_LOG.md`.

## Current limitations

- ARI evaluation uses exact decoded text and `textstat`, so it remains a CPU
  boundary inside each MCMC step.
- Suffix moves can be highly autocorrelated, especially at strong tilts.
- Resampling introduces genealogical dependence among particles.
- SNIS is generally biased at finite particle count and can fail under poor
  overlap even when the SMC population has healthy target ESS.
- A finite run has no automatic convergence or efficiency guarantee. Use
  independent seeds and inspect acceptance, ancestry, both ESS diagnostics,
  and sensitivity to the lambda/suffix schedules.

## References

Full bibliographic entries and the role of each source are listed in
[THEORY.md](THEORY.md). The principal starting-point paper is Dorman et al.,
*Rare Event Analysis of Large Language Models* (2026).
