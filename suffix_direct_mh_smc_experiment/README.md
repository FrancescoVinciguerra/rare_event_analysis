# Suffix direct-MH SMC

This package is the active research pipeline. It samples complete, fixed-length
language-model completions along an exponential-tilt schedule and estimates
base-model ARI tails from the final weighted population.

The mathematical specification and correctness derivation are in
[`THEORY.md`](../THEORY.md). In particular, the active proposal is an exact
temperature-one LM suffix proposal corrected by direct Metropolis-Hastings. It
does not use Barker acceptance, a balancing function, reverse teacher forcing,
or a vocabulary-wide proposal normalizer.

## Files

- `smc_sampler.py`: particle state, batched suffix generation, direct-MH
  mutation, SMC weighting, ESS, resampling, and serialization.
- `run_experiment.py`: one independent SMC run.
- `run_tail_estimation.py`: multi-seed SNIS tail estimates and optional terminal
  lambda schedule sweep.
- `run_tail_validation.py`: independent cap/lambda components, SMC normalizer
  estimates, deterministic-mixture MIS, and comparison with official direct
  statistics.

## One run

```bash
python suffix_direct_mh_smc_experiment/run_experiment.py \
  --device auto --num-particles 64 --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-1 --mcmc-steps-per-level 5 \
  --suffix-sizes 1,2,4,8,16,32,64,100 --seed 12345
```

Mutation at level `k` leaves `pi_(lambda[k-1])` invariant. Reweighting then
applies `exp(-(lambda[k]-lambda[k-1])*phi)`. Resampling may occur below the ESS
threshold at non-final levels only; final normalized log weights are retained.

## Tail estimation

```bash
python suffix_direct_mh_smc_experiment/run_tail_estimation.py \
  --device auto --seeds 12345,12346,12347 \
  --num-particles 256 --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-0.75,-1,-1.5,-2 \
  --mcmc-steps-per-level 10 --tail-thresholds 8,10,12,15
```

Each seed is an independent SMC population. The report keeps final SMC ESS and
corrected importance ESS separate. Generated run directories are ignored by
Git; curate only compact, interpretable summaries under `results/`.

## Official-reference validation

```bash
python suffix_direct_mh_smc_experiment/run_tail_validation.py \
  --device cuda --seed-groups 200,300,400 \
  --num-particles 256 --max-new-tokens 100 \
  --lambda-step 0.25 --mcmc-steps-per-level 10 \
  --thresholds 6,7,8,10
```

Every `(cap, terminal lambda, seed)` component is a separate complete SMC run.
The runner stores one compact `.npz` archive per component and reuses it only
when its full algorithm/configuration metadata matches. It combines final SMC
weights with estimated normalizing constants in log-space; no final resampling,
direct generation, TPS, or MBAR is performed.

Before loading a model, the runner checks model, tokenizer, prompt, completion
length, sampling temperature, truncation, EOS, and fixed-length settings against
the official reference metadata. Exact official/current ARI equivalence remains
unverified because the paper's custom edge-case implementation and original
completions are unavailable. Production is blocked unless
`--allow-unverified-observable-reference` is passed; that flag requests an
approximate cross-implementation comparison, never exact ground truth.

Audit an existing cache without loading the model or generating missing runs:

```bash
python suffix_direct_mh_smc_experiment/run_tail_validation.py \
  --audit-only --seed-groups 200 \
  --output-root results/tail_probability_validation
```

The audit classifies each candidate as `exactly_reusable`,
`comparable_but_not_cache_equivalent`, `incompatible`, or `unverifiable`.
Only unique exact matches enter the mixture denominator and corrected ESS. It
also reports component-wise reverse estimates and compares SMC and reverse
normalizing-constant diagnostics in log-space.
