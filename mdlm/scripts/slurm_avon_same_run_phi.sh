#!/usr/bin/env bash
# A separate, frozen source snapshot is supplied as EXPERIMENT_DIR. The old
# experiment source and results on Avon are never overwritten.
set -euo pipefail
source "${EXPERIMENT_DIR:?Set the frozen code snapshot}/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${CALIBRATION_ROOT:?Set the previous completed ladder directory}"
study="$EXPERIMENT_DIR/configs/mdlm_owt_same_run_logit.yaml"
name=camera_ready_same_run_logit

finalize() {
  result=$?; trap - EXIT; set +e
  analysis_result=0
  if compgen -G "$OUTPUT_ROOT/$name/workers/worker_*/populations/*.json" >/dev/null; then
    python -m diffusion_toxicity_smc_experiment.same_run_reuse report \
      --study "$study" --output-root "$OUTPUT_ROOT" \
      --pool-root "$OUTPUT_ROOT/camera_ready_gradual_budget" || analysis_result=$?
    if test "$analysis_result" -eq 0; then
      python "$EXPERIMENT_DIR/analysis/same_run_timing.py" --results "$OUTPUT_ROOT/$name" || analysis_result=$?
      if python -c "import importlib.util; raise SystemExit(importlib.util.find_spec('matplotlib') is None)"; then
        python "$EXPERIMENT_DIR/plots/plot_same_run_ladder.py" --results "$OUTPUT_ROOT/$name" || analysis_result=$?
      else
        echo 'Matplotlib unavailable in this HPC environment: CSVs/report saved; render plots locally after download.'
      fi
    fi
    tar -czf "$OUTPUT_ROOT/${name}_results.tar.gz" -C "$OUTPUT_ROOT" "$name" || analysis_result=$?
    echo "Report: $OUTPUT_ROOT/$name/REPORT.html"
    python - "$OUTPUT_ROOT/$name/report_status.json" <<'PY' || analysis_result=$?
import json,sys
state=json.load(open(sys.argv[1]))
if state['completed_runs']!=state['expected_runs']:
    raise SystemExit('Partial study: saved runs are preserved; resume is required')
PY
  fi
  if test "$analysis_result" -ne 0; then exit "$analysis_result"; fi
  exit "$result"
}
trap finalize EXIT
trap 'exit 143' TERM

# Real model inference starts only inside the allocated worker steps below.
# First repeat the CPU contract checks inside the allocated batch resources.
python -m pytest -q -c "$EXPERIMENT_DIR/pytest_snapshot.ini" --rootdir "$EXPERIMENT_DIR" \
  "$EXPERIMENT_DIR/tests/test_potentials.py" \
  "$EXPERIMENT_DIR/tests/test_population_checkpoints.py" \
  "$EXPERIMENT_DIR/tests/test_same_run_reuse.py" \
  "$EXPERIMENT_DIR/tests/test_same_run_timing.py"

pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G \
    --gres=gpu:quadro_rtx_6000:1 \
    --output="$LOG_DIR/mdlm-logit-${SLURM_JOB_ID}-worker${worker}.out" \
    --error="$LOG_DIR/mdlm-logit-${SLURM_JOB_ID}-worker${worker}.err" \
    python -m diffusion_toxicity_smc_experiment.same_run_reuse run-fresh \
      --study "$study" --output-root "$OUTPUT_ROOT" \
      --calibration-root "$CALIBRATION_ROOT" \
      --max-seconds 6300 --workers 3 --worker-id "$worker" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
