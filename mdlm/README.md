# MDLM rare-event estimation

## Quick start

Run these commands from the repository root.

```bash
python -m pip install -e .
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode plan
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode smoke
```

[reproduce.py](reproduce.py) runs the canonical RollBack–SMC and crude MC
experiment. Set the particle grid, seeds, checkpoints and comparison policy in
[paper.yaml](configs/paper.yaml). Use `--mode run` for the experiment and
`--mode analysis` to build the tables and figures from saved populations.
The [reproduction guide](reproduction/README.md) covers crude MC, resume,
archived inputs and the GPU integration test.

## Repository layout

- [src/](src/diffusion_toxicity_smc_experiment/) contains the reusable sampler,
  estimators and analysis used to produce tables and figures.
- [tests/](tests/) covers the paper workflow, the sampler and historical
  numerical regressions. Those regressions use [legacy presets](configs/legacy/).
- The [historical campaign notes](reproduction/docs/HISTORICAL_CAMPAIGNS.md)
  identify the frozen sources and recovery commands. The
  [results index](../shared/RESULTS_INDEX.md#mdlm-toxicity-experiments) links to
  saved results and figures.

## Single estimate

```bash
python -m diffusion_toxicity_smc_experiment.estimate --particles 2 --tilt 0.5
```

The estimator loads about 1.2 GB of pinned model weights on first use. It
defaults to CUDA. Set `--device cpu` for CPU or `--offline` for a prefilled
model cache. Defaults are 64 particles, tilt 1, threshold 0.995 and seed 42.
The JSON output includes the non-self-normalized estimate `p_hat_smc` and
the event count.

## Standalone source export

From an installed checkout, run `python mdlm/tools/export_mdlm_source.py`
to package the single-estimate command with its resolved configuration.
