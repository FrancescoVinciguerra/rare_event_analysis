#!/usr/bin/env bash
set -euo pipefail
source "${PROJECT_DIR:?Set PROJECT_DIR}/mdlm/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
finalize() {
  result=$?; trap - EXIT; set +e
  analysis_result=0
  root="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
  if compgen -G "$root/camera_ready_same_run_ladder/workers/worker_*/populations/*.json" >/dev/null; then
    python -m diffusion_toxicity_smc_experiment.same_run_reuse report \
      --study "$EXPERIMENT_DIR/configs/mdlm_owt_same_run_ladder.yaml" --output-root "$root" \
      --pool-root "$root/camera_ready_gradual_budget" || analysis_result=$?
    tar -czf "$root/camera_ready_same_run_ladder_results.tar.gz" -C "$root" camera_ready_same_run_ladder
    echo "Report: $root/camera_ready_same_run_ladder/REPORT.html"
  fi
  if test "$analysis_result" -ne 0; then exit "$analysis_result"; fi
  exit "$result"
}
trap finalize EXIT
trap 'exit 143' TERM
pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G \
    --gres=gpu:quadro_rtx_6000:1 \
    --output="$LOG_DIR/mdlm-reuse-ladder-${SLURM_JOB_ID}-worker${worker}.out" \
    --error="$LOG_DIR/mdlm-reuse-ladder-${SLURM_JOB_ID}-worker${worker}.err" \
    python -m diffusion_toxicity_smc_experiment.same_run_reuse run \
      --study "$EXPERIMENT_DIR/configs/mdlm_owt_same_run_ladder.yaml" \
      --output-root "${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}" \
      --max-seconds 6300 --workers 3 --worker-id "$worker" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
