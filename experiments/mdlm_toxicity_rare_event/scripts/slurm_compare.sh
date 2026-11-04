#!/usr/bin/env bash
#SBATCH --job-name=mdlm-compare
#SBATCH --partition=gpu-devel
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.out
#SBATCH --error=experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.err

set -euo pipefail
: "${DIRECT_RUN:?Set DIRECT_RUN to an existing direct run}"
: "${SMC_RUN:?Set SMC_RUN to an existing SMC run}"
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/experiments/mdlm_toxicity_rare_event/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  --config "${COMPARE_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_smc.yaml}"
  --direct-run "${DIRECT_RUN}"
  --smc-run "${SMC_RUN}"
)
if [[ -n "${COMPARISON_OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "${COMPARISON_OUTPUT_DIR}")
fi
python -m diffusion_toxicity_smc_experiment.compare "${args[@]}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
