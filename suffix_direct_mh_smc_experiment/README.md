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
