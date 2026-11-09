# Rare-event estimation for language models

This repository contains rare-event probability-estimation experiments for
language models. The original experiments use Automated Readability Index
(ARI); the masked-diffusion toxicity experiment is isolated in its own
directory.

To run one MDLM RollBack SMC estimate, start with the
[MDLM quick start](experiments/mdlm_toxicity_rare_event/README.md#quick-start).

The research checkout keeps the campaign drivers and results together. To export
the standalone estimator source tree from an installed checkout, run
`python tools/export_mdlm_source.py`. The ZIP contains the estimator package,
one resolved configuration, its dependencies, and a short README.

## Experiments

- **[Results index, Gaussian toy notebooks, and individual runs](experiments/RESULTS_INDEX.md)**:
  saved experiment outputs, plots, reproducible code, and completion status.
- [`experiments/gaussian_rollback_smc/`](experiments/gaussian_rollback_smc/README.md):
  Gaussian full-trajectory RollBack SMC without resampling, theory and results.
- [`experiments/gaussian_rollback_resampling/`](experiments/gaussian_rollback_resampling/README.md):
  Gaussian ESS/systematic SMC, independent-run results, and a self-contained Colab notebook.
- [`experiments/gaussian_rollback_gpu_previous/`](experiments/gaussian_rollback_gpu_previous/):
  saved Gaussian GPU and deep-tilt experiments with their notebooks and run data.
- [`experiments/mdlm_toxicity_rare_event/`](experiments/mdlm_toxicity_rare_event/README.md):
  unconditional MDLM-OWT generation and RollBack SMC estimation of toxicity tails.
- [`suffix_direct_mh_smc_experiment/`](suffix_direct_mh_smc_experiment/):
  suffix direct-MH SMC for the ARI experiment.
- [`official_annealed_tps_experiment/`](official_annealed_tps_experiment/):
  adapter for the official annealed TPS implementation.

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

The repository is public. Some historical language-model notebooks include
GitHub authentication through the `GITHUB_TOKEN` Colab secret. The Gaussian
Colab notebooks are self-contained and do not require repository authentication.

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
[REA_algorithms_minimal.tex](REA_algorithms_minimal.tex).

## Code Layout

```text
external/rea_of_llms_official/          pinned official supplementary code
official_annealed_tps_experiment/       TPS adapter and MBAR analysis
suffix_direct_mh_smc_experiment/        SMC and multi-component F_j runners
experiments/mdlm_toxicity_rare_event/   isolated MDLM toxicity experiment
rea_llms/                               generation, observables, and estimators
notebooks/                              the three executable experiments
tests/                                  numerical and notebook checks
```

Install and test locally with:

```bash
uv sync
uv run pytest -q
```

Generated completions, caches, and large result directories are ignored by
Git. Historical experiments remain recoverable from repository history.
