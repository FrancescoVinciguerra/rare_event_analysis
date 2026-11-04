#!/usr/bin/env bash
#SBATCH --job-name=mdlm-smoke-direct
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
python -m diffusion_toxicity_smc_experiment.generate_direct \
  --config "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG}" \
  --target-samples 2 \
  --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}" \
  --task-index 0 \
  --num-tasks 1
echo "Re-running the resume scan; this must generate zero new samples."
python -m diffusion_toxicity_smc_experiment.generate_direct \
  --config "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG}" \
  --target-samples 2 \
  --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}" \
  --task-index 0 \
  --num-tasks 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
