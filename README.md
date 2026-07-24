# REA of LLMs - Minimal Research Workspace

This repository contains a small, readable workspace for rare-event experiments with language-model completions. It is intentionally limited to the code needed to run and document three experimental samplers, plus selected results that are suitable for review.

Default model setup:

- model: `roneneldan/TinyStories-8M`
- tokenizer: `EleutherAI/gpt-neo-125M`
- default prompt: `Once upon a time, in a big forest, there lived a rhinoc`
- observable: exact `textstat` Automated Readability Index (ARI)

## Implemented Algorithms

### Mixture MCMC proposal + SNIS to `p_M`

`mixture_mcmc_experiment/` samples from an unnormalized mixture of tilted full-completion distributions,

```text
Gamma_mix(x) = sum_k rho_k p_M(x | c) exp(-lambda_k phi(x)).
```

Samples are then reweighted with self-normalized importance sampling (SNIS) to estimate quantities under the base model `p_M`. Direct ancestral sampling is not run by default. It is opt-in with `--run-direct-baseline`.

### Full-sequence SMC sampler to `p_lambda`

`full_sequence_smc_experiment/` keeps each particle as a complete generated completion. It mutates particles with full-sequence independence proposals from the base model and updates weights across a lambda schedule. It reports ESS/N, acceptance rates, and ARI tail summaries for the final weighted population.

This is not token-level SIS, does not use vocabulary-wide tilted proposals, and does not run TPS or MBAR.

### Masked Barker full-sequence SMC

`masked_barker_smc_experiment/` is an independent SMC pipeline over complete,
fixed-length completions. At each mutation step it draws a state-independent
token mask, resamples the selected tokens left-to-right with the LLM, evaluates
the exact forward and reverse masked-proposal probabilities, symmetrizes the
proposal, and applies Barker acceptance for the current tilted target.

Weights are updated across the lambda schedule in log-space. Systematic
ESS-adaptive resampling is enabled by default. The implementation does not
evaluate target normalizers, does not construct a normalized `Q_g`, and does
not depend on TPS, MBAR, direct sampling, mixture MCMC, or the earlier
full-sequence SMC pipeline. See
`masked_barker_smc_experiment/README.md` for the detailed-balance argument and
mask schedule syntax.

## Repository Map

- `rea_llms/models.py`: model/tokenizer loading and device selection.
- `rea_llms/observables/`: ARI observable and score container.
- `rea_llms/experiment_log.py`: shared result logging, ARI tail summaries, and selected-result publishing.
- `full_sequence_smc_experiment/run_smc_sampler.py`: CLI for full-sequence SMC.
- `masked_barker_smc_experiment/run_experiment.py`: CLI for masked Barker SMC.
- `mixture_mcmc_experiment/run_experiment.py`: CLI for mixture MCMC.
- `notebooks/REA_Colab_Quickstart.ipynb`: Colab quickstart.
- `results/EXPERIMENT_LOG.md`: cumulative log of selected runs.
- `results/selected_runs/`: only curated run summaries and small sample previews.
- `tests/`: lightweight unit tests.

## Colab Quickstart

Upload or clone the repository in Colab, then run:

```bash
pip install -r requirements-colab.txt
```

Open `notebooks/REA_Colab_Quickstart.ipynb` for a clean end-to-end notebook with GPU checks, a substantial masked Barker SMC run, validation plots, Drive output paths, and selected-result publishing instructions.

The quickstart uses an `ALGORITHMS_TO_RUN` list and defaults to
`["masked_barker_smc"]`, so the new pipeline can run by itself.

To compare against the official supplementary code from the paper, clone with submodules:

```bash
git clone --recurse-submodules https://github.com/FrancescoVinciguerra/rea_of_llms_supplementary_code-main
```

The official paper code is tracked as a Git submodule at `external/rea_of_llms_official`. Open `notebooks/compare_with_official_colab.ipynb` in Colab to run this repository's SMC pipeline and the official notebook-based TPS/MBAR examples in the same runtime, with shared seeds and comparable small-example budgets.

## Smoke Runs

Full-sequence SMC:

```bash
python full_sequence_smc_experiment/run_smc_sampler.py \
  --device auto \
  --num-particles 2 \
  --max-new-tokens 2 \
  --lambdas 0,0 \
  --mcmc-steps-per-level 1 \
  --output-dir outputs/full_sequence_smoke
```

Mixture MCMC:

```bash
python mixture_mcmc_experiment/run_experiment.py \
  --device auto \
  --num-iterations 2 \
  --burn-in 0 \
  --thinning 1 \
  --max-new-tokens 2 \
  --lambdas=-0.5,0.0 \
  --output-dir outputs/mixture_smoke
```

Masked Barker SMC:

```bash
python masked_barker_smc_experiment/run_experiment.py \
  --device auto \
  --num-particles 4 \
  --max-new-tokens 4 \
  --lambdas 0,0 \
  --mcmc-steps-per-level 1 \
  --mask-mode deterministic \
  --mask-sizes 1 \
  --deterministic-mask-size-schedule 1 \
  --output-dir outputs/masked_barker_smoke
```

## Logging and Selected Results

Every run always writes to its `--output-dir`. To document a run for GitHub review:

```bash
python mixture_mcmc_experiment/run_experiment.py \
  --run-name mixture_example \
  --experiment-log results/EXPERIMENT_LOG.md \
  --publish-result-dir results/selected_runs
```

The selected result directory receives only:

- `summary.json`
- `report.md`
- `samples_preview.jsonl`

Large full sample files are intentionally not copied to `results/selected_runs/`.

## What Not To Commit

Do not commit:

- virtual environments;
- Python caches;
- Hugging Face caches or model weights;
- complete large `samples.jsonl` files;
- temporary run directories;
- unreviewed raw results.

Commit only selected result summaries that support a stated conclusion in `results/EXPERIMENT_LOG.md`.
