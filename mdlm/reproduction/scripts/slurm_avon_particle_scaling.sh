#!/usr/bin/env bash
set -euo pipefail
source "${EXPERIMENT_DIR:?Set the frozen experiment directory}/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
: "${OUTPUT_ROOT:?Set the output root}"
: "${CALIBRATION_ROOT:?Set the original calibrated ladder directory}"
study="$EXPERIMENT_DIR/configs/${STUDY_CONFIG:-mdlm_owt_particle_scaling.yaml}"
name="${RESULT_NAME:-camera_ready_particle_scaling}"
soft_limit="${SOFT_MAX_SECONDS:-21000}"
analysis=(python -m diffusion_toxicity_smc_experiment.particle_scaling)

tests=(
  "$EXPERIMENT_DIR/tests/test_potentials.py"
  "$EXPERIMENT_DIR/tests/test_population_checkpoints.py"
  "$EXPERIMENT_DIR/tests/test_same_run_reuse.py"
)
if [[ "$name" == "camera_ready_low_particle_budget" || "$name" == "camera_ready_ultra_low_particle_budget" || "$name" == "camera_ready_deeper_tilt_budget" || "$name" == "camera_ready_extreme_tail_scaling" || "$name" == "camera_ready_progressive_tail_particle_scaling" ]]; then
  tests+=("$EXPERIMENT_DIR/tests/test_low_particle_budget.py")
fi
python -m pytest -q -c "$EXPERIMENT_DIR/pytest_snapshot.ini" --rootdir "$EXPERIMENT_DIR" "${tests[@]}"
"${analysis[@]}" verify-pool --study "$study" --output-root "$OUTPUT_ROOT"

finish() {
  local status=$?
  trap - EXIT
  set +e
  "${analysis[@]}" report --study "$study" --output-root "$OUTPUT_ROOT" || status=$?
  tar -czf "$OUTPUT_ROOT/${name}_results.tar.gz" -C "$OUTPUT_ROOT" "$name" || status=$?
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G \
    --gres=gpu:quadro_rtx_6000:1 \
    --output="$LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-worker${worker}.out" \
    --error="$LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-worker${worker}.err" \
    python -m diffusion_toxicity_smc_experiment.same_run_reuse run-fresh \
      --study "$study" --output-root "$OUTPUT_ROOT" --calibration-root "$CALIBRATION_ROOT" \
      --max-seconds "$soft_limit" --workers 3 --worker-id "$worker" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
