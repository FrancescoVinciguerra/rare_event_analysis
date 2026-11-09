#!/usr/bin/env bash
#SBATCH --job-name=mdlm-component-export
#SBATCH --account=stats
#SBATCH --partition=devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mdlm/artifacts/logs/%x-%j.out
#SBATCH --error=mdlm/artifacts/logs/%x-%j.err

set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
export VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
export HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/mdlm}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export AVON_MODULES="${AVON_MODULES:-GCCcore/13.2.0 Python/3.11.5}"

source "${EXPERIMENT_DIR}/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_cpu_diagnostics

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

destination="${OUTPUT_ROOT}/mdlm_owt/final_component_export/component_estimates.csv"
python -m diffusion_toxicity_smc_experiment.component_export \
  --study "${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_final_analysis.yaml" \
  --output-root "${OUTPUT_ROOT}" \
  --destination "${destination}"
