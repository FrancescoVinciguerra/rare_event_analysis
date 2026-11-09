#!/usr/bin/env bash
#SBATCH --job-name=mdlm-submission-probability
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
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/mdlm/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_cpu_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m diffusion_toxicity_smc_experiment.submission_probability \
  "${SUBMISSION_PROBABILITY_COMMAND:?Set SUBMISSION_PROBABILITY_COMMAND}" \
  --study "${SUBMISSION_STUDY_CONFIG:?Set SUBMISSION_STUDY_CONFIG}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"

