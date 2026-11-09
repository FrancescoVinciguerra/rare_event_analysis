#!/usr/bin/env bash
#SBATCH --job-name=mdlm-terminal-tail
#SBATCH --account=stats
#SBATCH --partition=devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/mdlm-terminal-tail-%j.out
#SBATCH --error=logs/mdlm-terminal-tail-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
EXPERIMENT_DIR="${PROJECT_DIR}/mdlm"
ARTIFACT_ROOT="${EXPERIMENT_DIR}/artifacts/current/mdlm_owt"
REPORT_DIR="${ARTIFACT_ROOT}/final_analysis/ea63ee1cc76dfee98fe8da9416250d4b6db8538395e1bd7b90ed894bd3985572"
OUTPUT_DIR="${ARTIFACT_ROOT}/offline_terminal_tail_analysis/terminal_tail_v2_20260823"
VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
AVON_MODULES="${AVON_MODULES:-GCCcore/13.2.0 Python/3.11.5}"

export PROJECT_DIR VENV_DIR HF_HOME AVON_MODULES
source "${EXPERIMENT_DIR}/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_cpu_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "analysis_only=true"
echo "generation_or_smc=false"
echo "artifact_root=${ARTIFACT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"

python "${EXPERIMENT_DIR}/reproduction/analysis/offline_terminal_tail_analysis.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --skip-figures
