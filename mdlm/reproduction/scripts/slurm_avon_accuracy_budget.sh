#!/usr/bin/env bash
set -euo pipefail
source "${PROJECT_DIR:?Set PROJECT_DIR}/mdlm/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
finalize() {
  result=$?; trap - EXIT; set +e
  root="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
  if compgen -G "$root/camera_ready_accuracy_budget_pilot/workers/worker_*/runs/*.json" >/dev/null; then
    python -m diffusion_toxicity_smc_experiment.accuracy_budget report \
      --study "$EXPERIMENT_DIR/configs/mdlm_owt_accuracy_budget_pilot.yaml" --output-root "$root"
    if python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("matplotlib") is None)'; then
      python "$EXPERIMENT_DIR/reproduction/plots/plot_accuracy_budget.py" --results "$root/camera_ready_accuracy_budget_pilot"
    else
      echo 'CSV/HTML ready; PNG figures will be rendered locally after downloading, without changing the Avon environment.'
    fi
    tar -czf "$root/camera_ready_accuracy_budget_pilot_results.tar.gz" -C "$root" camera_ready_accuracy_budget_pilot
    echo "Report: $root/camera_ready_accuracy_budget_pilot/REPORT.html"
  fi
  exit "$result"
}
trap finalize EXIT
trap 'exit 143' TERM
pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G \
    --gres=gpu:quadro_rtx_6000:1 \
    --output="$LOG_DIR/mdlm-accuracy-${SLURM_JOB_ID}-worker${worker}.out" \
    --error="$LOG_DIR/mdlm-accuracy-${SLURM_JOB_ID}-worker${worker}.err" \
    python -m diffusion_toxicity_smc_experiment.accuracy_budget run \
      --study "$EXPERIMENT_DIR/configs/mdlm_owt_accuracy_budget_pilot.yaml" \
      --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}" \
      --max-seconds 20700 --workers 3 --worker-id "$worker" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
