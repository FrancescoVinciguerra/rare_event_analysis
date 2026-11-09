# Suffix direct-MH SMC

This package contains the active rare-completion sampler.

- `smc_sampler.py` implements particle state, suffix regeneration, direct-MH
  mutation, incremental weighting, ESS, resampling, and serialization.
- `run_experiment.py` runs one independent SMC population with either ARI or a
  Qwen3Guard-Gen terminal observable.
- `run_tail_estimation.py` runs independent seeds and computes SNIS tail
  estimates plus overlap diagnostics.
- `run_tail_validation.py` runs independent `(cap, lambda_j)` components,
  computes one separately normalized `F_j` per component, averages the `F_j`
  values, and compares them with a matching IID direct reference.

Terminal resampling is disabled by default. Each `F_j` uses the normalized
terminal SMC weights together with the reverse-tilt factor. Pass
`--resample-final` to obtain an unweighted terminal population for compatibility;
this is valid but adds a final resampling source of Monte Carlo variance.

The proposal is the temperature-one base-LM conditional for a regenerated
suffix. The completion length is fixed and the suffix-length law is
state-independent, yielding the acceptance probability derived in
[`REA_algorithms_minimal.tex`](../../shared/paper/REA_algorithms_minimal.tex).

```bash
python llm/suffix_direct_mh_smc_experiment/run_experiment.py \
  --device auto --num-particles 64 --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-1 --mcmc-steps-per-level 5 \
  --suffix-sizes 1,2,4,8,16,32,64,100 --seed 12345
```

For multi-seed ARI tail estimation:

```bash
python llm/suffix_direct_mh_smc_experiment/run_tail_estimation.py \
  --device auto --seeds 12345,12346,12347 \
  --num-particles 256 --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-0.75,-1,-1.5,-2 \
  --mcmc-steps-per-level 10 --tail-thresholds 8,10,12,15
```

The main ARI workflow is
[`02_ari_smc_snis.ipynb`](../notebooks/02_ari_smc_snis.ipynb). Final SMC ESS
and corrected importance ESS are reported separately. Generated run
directories are ignored by Git.
