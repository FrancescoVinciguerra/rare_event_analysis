#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_JOB_ID:?Run setup only inside an allocated Avon compute node}"
: "${AVON_MODULES:?Set AVON_MODULES after the read-only preflight}"

PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiments/mdlm_toxicity_rare_event}"

if ! type module >/dev/null 2>&1; then
  echo "The environment-modules command is unavailable; stop without guessing an initialization path." >&2
  exit 1
fi
module purge
read -r -a avon_module_list <<< "${AVON_MODULES}"
module load "${avon_module_list[@]}"

if [[ ! -f "${EXPERIMENT_DIR}/src/diffusion_toxicity_smc_experiment/requirements-avon.txt" ]]; then
  echo "Repository not found at ${PROJECT_DIR}" >&2
  exit 1
fi
if [[ -e "${VENV_DIR}" ]]; then
  echo "Refusing to replace existing environment: ${VENV_DIR}" >&2
  exit 1
fi

mkdir -p \
  "${EXPERIMENT_DIR}/artifacts/current" \
  "${EXPERIMENT_DIR}/artifacts/logs" \
  "${HF_HOME}" \
  "$(dirname "${VENV_DIR}")"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --no-cache-dir --upgrade pip==25.1.1
python -m pip install --no-cache-dir \
  torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir \
  -r "${EXPERIMENT_DIR}/src/diffusion_toxicity_smc_experiment/requirements-avon.txt"
python -m pip install --no-deps --editable "${PROJECT_DIR}"
python -m pip check

echo "Environment created at ${VENV_DIR}"
echo "No shell startup file was modified."
