#!/usr/bin/env bash
#SBATCH --job-name=mdlm-direct
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH --output=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%A_%a.out
#SBATCH --error=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%A_%a.err

set -euo pipefail
: "${TARGET_SAMPLES:?Set the explicit unconditional target; no production default is provided}"
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/experiments/mdlm_toxicity_rare_event/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m diffusion_toxicity_smc_experiment.generate_direct \
  --config "${DIRECT_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct.yaml}" \
  --target-samples "${TARGET_SAMPLES}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${SLURM_ARRAY_TASK_COUNT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
