# Unconditional MDLM toxicity rare-event benchmark

## Quick start

From the repository root, with Python 3.11 and a PyTorch environment:

```bash
python -m pip install -e .
python -m diffusion_toxicity_smc_experiment.estimate --particles 2 --tilt 0.5
```

The first run downloads about 1.2 GB of pinned model weights. It uses CUDA by
default. Add `--device cpu` for CPU or `--offline` for a prefilled model cache.
The example checks that the pipeline runs. Omit `--particles` and `--tilt`
to use 64 particles and terminal tilt 1. Set the event with `--threshold`
(default 0.995) and the random seed with `--seed` (default 42).
The JSON output includes the non-self-normalized estimate `p_hat_smc`,
the seed, the event count, and the runtime.

This entry point uses the 32-step native MDLM decoder, a logit score,
tilt increments of 0.5, and four RollBack–MH moves per level. Systematic
resampling is triggered by ESS/N below 0.5 and skipped at the final level.
A single estimate can be zero when its terminal population contains no event.

## Repository layout

| Start here | Contents |
| --- | --- |
| [src/](src/diffusion_toxicity_smc_experiment/) | The estimator and reusable experiment code. `estimate.py` is the single-run entry point. |
| [configs/](configs/) | Pinned model settings and named campaign configurations. The single-run command picks its own configuration. |
| [artifacts/](artifacts/README.md) | Saved trajectories, particles, weights, checkpoints, CSVs, and reports. Each campaign's images are in its `plots/` folder. |
| [reproduction/](reproduction/README.md) | Study protocols, Slurm launchers, offline analysis, and figure renderers. |
| [tests/](tests/) | Sampler, numerical, and launcher checks. |

The standalone source exporter is the smallest reproducible code bundle. It
copies the 12 MDLM Python modules used by the estimator, two shared utility
files, one resolved configuration, the dependency list, and the quick start.

```bash
python mdlm/tools/export_mdlm_source.py
```

For the RollBack–MH derivation, see the [method notes](reproduction/docs/TRAJECTORY_RESAMPLE_MOVE.md).
For the main figures, start with the [results index](../shared/RESULTS_INDEX.md#matched-budget-figures).
