#!/usr/bin/env bash
#SBATCH --job-name=mdlm-calibration-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out
#SBATCH --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/mdlm/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_cpu_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m diffusion_toxicity_smc_experiment.calibration_report \
  --config "${CALIBRATION_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_lambda_calibration.yaml}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
