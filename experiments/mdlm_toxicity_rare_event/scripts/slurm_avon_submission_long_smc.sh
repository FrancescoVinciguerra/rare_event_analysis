#!/usr/bin/env bash
#SBATCH --job-name=mdlm-submission-long-smc
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.out
#SBATCH --error=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/experiments/mdlm_toxicity_rare_event/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The pilot weights are frozen before any independent test population is read.
python -m diffusion_toxicity_smc_experiment.submission_probability verify-pilot \
  --study "${SUBMISSION_STUDY_CONFIG:?Set SUBMISSION_STUDY_CONFIG}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"

python -m diffusion_toxicity_smc_experiment.serious_comparison run \
  --suite "${SUBMISSION_LONG_SUITE_CONFIG:?Set SUBMISSION_LONG_SUITE_CONFIG}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

