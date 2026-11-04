#!/usr/bin/env bash
#SBATCH --job-name=mdlm-smoke-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.out
#SBATCH --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/experiments/mdlm_toxicity_rare_event/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m diffusion_toxicity_smc_experiment.score_direct \
  --config "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG}" \
  --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
