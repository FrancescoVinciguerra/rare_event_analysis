#!/usr/bin/env bash
set -euo pipefail
source "${PROJECT_DIR:?}/mdlm/reproduction/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
finalize() {
  result=$?;trap - EXIT;set +e
  python -m diffusion_toxicity_smc_experiment.wall_clock_followup report \
    --study "$EXPERIMENT_DIR/configs/mdlm_owt_wall_clock_followup.yaml" --output-root "$OUTPUT_ROOT"
  if [[ -d "$OUTPUT_ROOT/camera_ready_exploration_v2_wall_clock" ]]; then
    tar -czf "$OUTPUT_ROOT/camera_ready_exploration_v2_wall_clock_results.tar.gz" -C "$OUTPUT_ROOT" camera_ready_exploration_v2_wall_clock
  fi
  exit "$result"
}
trap finalize EXIT
pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G --gres=gpu:quadro_rtx_6000:1 \
    --output="$LOG_DIR/mdlm-wall-clock-${SLURM_JOB_ID}-worker${worker}.out" \
    --error="$LOG_DIR/mdlm-wall-clock-${SLURM_JOB_ID}-worker${worker}.err" \
    python -m diffusion_toxicity_smc_experiment.wall_clock_followup run \
      --study "$EXPERIMENT_DIR/configs/mdlm_owt_wall_clock_followup.yaml" --output-root "$OUTPUT_ROOT" --worker-id "$worker" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}";do wait "$pid" || result=1;done
exit "$result"
