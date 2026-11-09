#!/usr/bin/env bash
#SBATCH --job-name=mdlm-final-confirmatory
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --output=mdlm/artifacts/logs/%x-%j.out
#SBATCH --error=mdlm/artifacts/logs/%x-%j.err

set -euo pipefail
source "${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}/mdlm/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
from pathlib import Path

import diffusion_toxicity_smc_experiment.final_analysis as final_analysis

expected_root = (Path(os.environ["EXPERIMENT_DIR"]) / "src").resolve()
module_path = Path(final_analysis.__file__).resolve()
if expected_root not in module_path.parents:
    raise RuntimeError(
        f"wrong final_analysis import: expected below {expected_root}, got {module_path}"
    )
print(f"final_analysis_module={module_path}")
print("final_analysis_import_smoke=PASS")
PY

python -m diffusion_toxicity_smc_experiment.serious_comparison run \
  --suite "${FINAL_CONFIRMATORY_SUITE_CONFIG:?Set FINAL_CONFIRMATORY_SUITE_CONFIG}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"

python -m diffusion_toxicity_smc_experiment.final_analysis \
  --study "${FINAL_ANALYSIS_CONFIG:?Set FINAL_ANALYSIS_CONFIG}" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
