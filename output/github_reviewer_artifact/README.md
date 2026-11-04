# Rare-event probability estimation for masked diffusion language models

This anonymous reviewer artifact contains the minimal code, configurations,
derived data, and tests needed to inspect the estimator and reproduce the
paper's tables and plots. It does not contain model checkpoints, Hugging Face
credentials, cluster-specific scripts, raw logs, or the 200,000 generated
completions.

## Artifact map

```text
analysis/                  CPU-only figure reproduction
experiments/.../configs/   frozen scientific configurations
data/final_analysis/       run-level estimates and final report tables
data/components/           per-lambda component diagnostics
docs/                      Trajectory-MH derivation
experiments/.../src/       MDLM decoder, FK SMC, Trajectory-MH, and reporting code
experiments/.../tests/     deterministic regression tests
rea_llms/statistics.py     Wilson interval helper
```

The released data are compact derived artifacts. Each row corresponds to an
independent run or tilted component; particles are never pooled across lambda
components before forming their separately normalized reverse-SNIS estimates.
The direct reference was computed from 200,000 IID MDLM samples.

The implementation builds on the MDLM reverse process of Sahoo et al. (2024)
and uses the Feynman--Kac steering construction of Singhal et al. (2025) as the
particle baseline. Exact bibliographic entries are provided in the accompanying
paper; this artifact reproduces the experimental implementation rather than the
authors' original repositories.

## Reproduce all paper plots (CPU only)

Python 3.10 or newer is required. No GPU, model download, or toxicity
classifier is needed for this step.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python analysis/reproduce_main_figures.py
python analysis/reproduce_component_figures.py
python analysis/verify_released_results.py
```

On Windows, activate with `.venv\Scripts\activate`. The commands write figures
to `reproduced_figures/` and do not modify the released CSV files.

## Run deterministic tests (CPU)

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The optional real-model test is skipped unless
`RUN_MDLM_GPU_INTEGRATION=1` is explicitly set. The ordinary test suite uses
small synthetic transition kernels and downloads nothing.

## Optional GPU reproduction

The frozen configuration uses unconditional `[MASK]^128` generation, 32
denoising steps, stochastic SUBS/DDPM propagation, four independent tilted
components at lambda 0.25, 0.50, 0.75, and 1.00, and uniform component weights.
The two tilted-population engines are FK bootstrap SMC and path-space
Trajectory-MH SMC. The decoder and evaluator revisions are pinned in
`experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_unconditional_common.yaml`.

The verified reference environment was Python 3.11.5, PyTorch 2.6.0+cu118,
CUDA runtime 11.8, Transformers 4.57.6, and an NVIDIA Quadro RTX 6000. On a
different system, install a PyTorch 2.6-or-newer wheel compatible with the
local NVIDIA driver. Do not assume bfloat16 or FlashAttention support.

After arranging checkpoint access through the normal Hugging Face mechanism:

```bash
python -m diffusion_toxicity_smc_experiment.prefetch_models \
  --config experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_unconditional_estimation_100tok.yaml \
  --output-root outputs

python -m diffusion_toxicity_smc_experiment.generate_direct \
  --config experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_unconditional_direct_32.yaml \
  --output-root outputs --target-samples 2 --allow-network

python -m diffusion_toxicity_smc_experiment.score_direct \
  --config experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_unconditional_estimation_100tok.yaml \
  --output-root outputs --allow-network

python -m diffusion_toxicity_smc_experiment.run_smc \
  --config experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_unconditional_estimation_100tok.yaml \
  --output-root outputs --allow-network
```

These are small functional examples, not commands for reproducing the full
GPU budget. Output storage is append-only: existing completed samples are
reused and incompatible manifests are rejected.

## Released-result scope

The primary event is the classifier score threshold 0.99. Thresholds 0.50,
0.90, and 0.999 are also reported, with 0.999 treated as exploratory. The
classifier defines the observable; the paper evaluates the probability
estimator rather than claiming that the classifier is a complete semantic
definition of harmfulness.

See `data/DATA_CARD.md` for provenance, fields, limitations, and the distinction
between public derived data and omitted raw generations.
