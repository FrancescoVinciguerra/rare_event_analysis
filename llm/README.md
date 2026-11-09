# Autoregressive LLM experiments

This area contains the readability and QwenGuard demonstrations for
autoregressive language models.

- [Suffix direct-MH SMC](suffix_direct_mh_smc_experiment/README.md)
- [Annealed TPS and MBAR](official_annealed_tps_experiment/README.md)
- [Saved tail validation](results/tail_probability_validation/)

Install from the repository root with `python -m pip install -e .`.
The original TPS reference is the [pinned submodule](external/rea_of_llms_official/).

## Notebooks

1. [Original TPS plus MBAR](notebooks/01_original_tps_mbar.ipynb)

   Runs independent annealed Transition Path Sampling chains using the pinned
   official implementation semantics, then applies MBAR across bias windows to
   estimate normalizing constants, overlap, and base-model tail probabilities.

2. [Suffix SMC plus averaged SNIS](notebooks/02_ari_smc_snis.ipynb)

   Runs IID direct sampling and our SMC pipeline on the same TinyStories ARI
   setting. For each terminal component `j`, it computes a separately
   normalized importance-sampling estimate `F_j(a)`. The final estimate is the
   average of the `F_j`; samples from different components are not pooled into
   one importance ratio. The notebook reports per-lambda tilting, corrected
   ESS, variance, relative error, boxplots, and direct-sampling comparisons.

3. [QwenGuard-Gen observable demonstration](notebooks/03_qwen_guard_smc.ipynb)

   Replaces ARI with Qwen3Guard-Gen Unsafe log-odds in a small direct-versus-SMC
   experiment. It displays complete responses and terminal category/refusal
   labels. This notebook demonstrates the mechanism; it is not a calibrated
   harmful-output probability estimate.

Historical notebooks that use GitHub authentication read the optional
`GITHUB_TOKEN` from Colab secrets.

## Estimator

For component `j`, with cap `C_j` and terminal bias `lambda_j`, SMC targets

```text
q_j(x) proportional to p_M(x | c) exp(-lambda_j psi_Cj(x)).
```

For event `f_a(x) = 1{ARI_raw(x) >= a}` and normalized terminal SMC weights
`W_ji`, the component estimate is

```text
F_j(a) = sum_i W_ji exp(lambda_j psi_Cj(X_ji)) f_a(X_ji)
         ------------------------------------------------------
         sum_i W_ji exp(lambda_j psi_Cj(X_ji)).
```

The unknown component normalizer cancels inside this self-normalized ratio.
By default the final SMC step is not resampled, avoiding an extra source of
Monte Carlo variance. With `--resample-final`, `W_ji = 1/N_j` after resampling
and the weights cancel, recovering the earlier unweighted formula.
Only afterward are components combined:

```text
P_hat(ARI_raw >= a) = (1 / J) sum_j F_j(a).
```

The mathematical derivation and current empirical results are in
[REA_algorithms_minimal.tex](../shared/paper/REA_algorithms_minimal.tex).
