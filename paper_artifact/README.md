# MDLM rare-event estimation

**Pre-release 0.1.0rc1.** This frozen snapshot contains the code and
configuration for the 384-run RollBack–SMC experiment. The final numerical
data are pending. [RELEASE.json](RELEASE.json) records the source commit,
file checksums and dependency versions.

## Install and check

From this directory, create a Python 3.11+ virtual environment and activate
it. Install a [PyTorch 2.6+ wheel](https://pytorch.org/get-started/locally/)
for your device. A CPU wheel is sufficient for planning, smoke tests and
analysis.

```bash
python -m pip install -e .
python reproduce.py --config paper.yaml --mode plan
python reproduce.py --config paper.yaml --mode smoke
```

The smoke test writes synthetic checkpoints, CSVs and PNGs under
`outputs/smoke/`. For regression tests, run
`python -m pip install -e ".[test]"` and `python -m pytest -q`.

## Recompute the published figures on CPU

When the final data are available, put the released bundle in `data/`.
It will contain `runs/` with 384 ladders, saved populations and their
independent timing calibrations, and `crude/` with the ordered score
batches for all 975,775 native samples. The release manifest will record
their provenance and checksums.

```bash
python reproduce.py --config paper.yaml --mode analysis --data data --output outputs/published
```

Analysis reads the saved checkpoints and crude scores without loading or
downloading models. It writes `runs.csv`, `summary.csv`, `reference.json`
and PNGs into the output directory. The canonical PNG style requires
LaTeX and `dvipng` on PATH. The synthetic smoke profile uses Matplotlib
without TeX.

## Rerun the experiment

The reference model stack uses Python 3.11.5, PyTorch 2.6.0 with CUDA 11.8
and Transformers 4.57.6. On a compatible NVIDIA GPU, install the reference
wheel before the project with
`python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118`.
Use a CUDA device with memory for the decoder, classifier and configured
microbatches. The reference GPU class was the 24 GB NVIDIA Quadro RTX 6000.

```bash
python reproduce.py --config paper.yaml --mode crude --allow-network
python reproduce.py --config paper.yaml --mode run
python reproduce.py --config paper.yaml --mode resume
python reproduce.py --config paper.yaml --mode analysis
```

`--allow-network` permits downloads of the pinned public models at first
use. Set `HF_HOME` to reuse an existing cache. `run` can also generate the
crude pool. Each execution with pending ladders measures its own independent
calibration. Use `resume` after interruption with the same configuration
and output directory. It reuses complete ladders and restarts an incomplete
ladder from its original seed.

[paper.yaml](paper.yaml) defines 24 independent runs for each particle
count from 32 to 512, spaced by 32. Each run tempers from 0 to 6.5 in steps
of 0.5 with four RollBack–MH moves per level. It records intermediate
checkpoints for the non-self-normalized SMC and reverse-SNIS estimates
and matches the crude MC comparisons by calibrated compute. By default,
the experiment writes to `outputs/paper/`. To run from another working
directory, pass absolute paths to `reproduce.py` and `--config`.
