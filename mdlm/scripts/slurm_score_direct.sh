#!/usr/bin/env bash
#SBATCH --job-name=mdlm-score
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=mdlm/artifacts/logs/%x-%j.out
#SBATCH --error=mdlm/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/mdlm/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  --config "${EVALUATOR_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct.yaml}"
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
)
if [[ -n "${DIRECT_RUN:-}" ]]; then
  args+=(--direct-run "${DIRECT_RUN}")
fi
python -m diffusion_toxicity_smc_experiment.score_direct "${args[@]}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
