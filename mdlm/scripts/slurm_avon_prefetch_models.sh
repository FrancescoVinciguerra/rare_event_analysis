#!/usr/bin/env bash
#SBATCH --job-name=mdlm-prefetch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out
#SBATCH --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/mdlm/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
python -m diffusion_toxicity_smc_experiment.prefetch_models \
  --config "${BENCHMARK_CONFIG:?Set BENCHMARK_CONFIG}" \
  --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
