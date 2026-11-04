#!/usr/bin/env bash
set -euo pipefail
source "${PROJECT_DIR:?Set PROJECT_DIR}/experiments/mdlm_toxicity_rare_event/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
finalize() {
  result=$?
  trap - EXIT
  set +e
  root="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
  if compgen -G "${root}/camera_ready_low_budget_v1/runs/N*_seed*.json" > /dev/null; then
    python -m diffusion_toxicity_smc_experiment.low_budget report \
      --study "${EXPERIMENT_DIR}/configs/mdlm_owt_camera_ready_low_budget.yaml" --output-root "$root"
    if [[ $? -ne 0 && $result -eq 0 ]]; then result=1; fi
    tar -czf "${root}/camera_ready_low_budget_v1_results.tar.gz" -C "$root" camera_ready_low_budget_v1
    if [[ $? -ne 0 && $result -eq 0 ]]; then result=1; fi
    echo "Report: ${root}/camera_ready_low_budget_v1/REPORT.html"
    echo "Download: ${root}/camera_ready_low_budget_v1_results.tar.gz"
  fi
  exit "$result"
}
trap finalize EXIT
trap 'exit 143' TERM
python -m diffusion_toxicity_smc_experiment.low_budget run \
  --study "${EXPERIMENT_DIR}/configs/mdlm_owt_camera_ready_low_budget.yaml" \
  --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
