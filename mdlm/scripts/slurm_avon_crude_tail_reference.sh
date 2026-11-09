#!/usr/bin/env bash
set -euo pipefail
source "${EXPERIMENT_DIR:?Set the frozen experiment directory}/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
name="${RESULT_NAME:?Set the campaign name}"
root="${OUTPUT_ROOT:?Set the output root}/$name"
study="$EXPERIMENT_DIR/configs/${STUDY_CONFIG:?Set the frozen study config}"
analysis=(python -m diffusion_toxicity_smc_experiment.crude_tail_reference)
"${analysis[@]}" initialize --study "$study" --output "$root"

finish() {
  local status=$?
  trap - EXIT
  set +e
  "${analysis[@]}" report --study "$study" --output "$root" || status=$?
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
    "${analysis[@]}" run --study "$study" --output "$root" --worker "$worker" \
      --max-seconds "${SOFT_MAX_SECONDS:-25200}" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
