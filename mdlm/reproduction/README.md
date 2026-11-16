# Reproduce the MDLM paper experiment

Start here to recover the paper's crude MC reference, RollBack–SMC comparison
and figures. For one probability estimate, use the [MDLM quick start](../README.md).

## Paper experiment with terminal tilt 6.5

From the repository root, run `python -m pip install -e .` for the Python
dependencies. The [paper YAML](../configs/paper.yaml) controls the 384-run
grid, seeds, ladder, checkpoint observations, crude cohorts and figures.
Set `model.runtime.device` in the YAML to the intended device. Figure
rendering uses a LaTeX installation when `figures.usetex` is true.

```bash
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode plan
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode smoke
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode crude
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode run
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode resume
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode analysis
```

The synthetic smoke profile uses the actual SMC driver with a tiny CPU
decoder and scores. `run` and `resume` fill the missing ladders and reuse
complete outputs at the same path. A ladder interrupted before its atomic
save is rerun from its original seed. Use `--output` to select a separate
artifact directory. Add `--allow-network` to permit downloading the pinned
models on a fresh machine. Each execution with pending ladders measures
independent crude calibration on the device running those ladders.
On a cluster, run the same command inside the allocated GPU job.

`crude` generates the configured fixed-size score cohorts, counts events,
and saves `reference.json`. The seeds for a fresh reproduction are in the
YAML. To recount or analyse an existing score bundle, supply `--crude-root`
with its `scores_manifest.json` and `batches/seed<S>.json` files. The manifest
records the ordered batch inventory, including cohort, seed, sample count,
matched-pool membership, decoder/evaluator IDs and SHA-256 of each batch.
`analysis` reads saved ladders and calibrations on CPU and writes
`runs.csv`, `summary.csv` and PNGs in `figures/with_captions` and
`figures/without_captions`. It requires a complete grid and valid crude data.

Audit an unpacked historical Avon experiment before using any saved run.

```bash
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode audit \
  --archive /path/to/unpacked-experiment \
  --frozen-source mdlm/reproduction/frozen/paper_final_scaling.tar.gz
```

`compatibility.json` lists the planned and archived numerical seed for each
replicate and the compatibility checks. In `run` or `resume`, the same
archive options copy only verified populations to the new output tree.
For comparisons using archived costs, supply the original ordered MC and
reference scores via `--crude-root`. Figure groups identify archived and
newly measured hardware separately.

### Validate the pinned models on a GPU

On an existing CUDA environment, cache the model and tokenizer revisions
in `paper.yaml`, including the toxicity classifier and the trimming
tokenizer. Use CUDA-enabled PyTorch and an allocated GPU. This test uses
the same `reproduce.py` pipeline with two short ladders in a pytest
temporary directory.

```bash
python -m pytest -q -s -m gpu mdlm/tests/test_paper_reproduction.py::test_pinned_models_on_gpu
```

Earlier experiments and their recovery commands are in the
[historical campaign notes](docs/HISTORICAL_CAMPAIGNS.md).

## SMC estimator

RollBack–MH keeps the realized prefix and regenerates the suffix with the
native decoder. The acceptance ratio uses a state-independent rollback-depth
law. The sampler uses the same logit potential in weighting, MH and reverse
correction. It updates weights and the normalizer before resampling and
mutation. Each checkpoint is analyzed at its own tilt, with its saved weights
and normalizer.

The focused tests cover [schedule and normalizers](../tests/test_exploratory.py),
[detailed balance and reverse correction](../tests/test_potentials.py),
[checkpoint behavior](../tests/test_population_checkpoints.py), and the
[paper seed plan](../tests/test_paper_final_scaling.py).
