#!/usr/bin/env bash
# Reviewed fixed design; exactly one submission, dry-run by default.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --submit ) ]]; then
  echo 'Usage: submit_avon_exploration_v2.sh [--submit]' >&2; exit 2
fi
: "${AVON_ACCOUNT:?Verify current account}"
: "${AVON_GPU_PARTITION:?Verify current partition}"
: "${AVON_QOS:?Verify current QoS}"
: "${AVON_GPU_GRES:?Verify current GRES}"
: "${AVON_MODULES:?Verify modules}"
: "${PROJECT_DIR:?Verify project path}"
: "${VENV_DIR:?Verify environment}"
: "${HF_HOME:?Verify model cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiments/mdlm_toxicity_rare_event}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
job="${EXPERIMENT_DIR}/scripts/slurm_avon_exploration_v2.sh"
[[ "$AVON_GPU_GRES" == 'gpu:quadro_rtx_6000:3' ]] || { echo 'This reviewed plan requests three GPUs' >&2;exit 2; }
command=(sbatch --parsable --account="$AVON_ACCOUNT" --partition="$AVON_GPU_PARTITION"
  --qos="$AVON_QOS" --gres="$AVON_GPU_GRES" --nodes=1 --ntasks=3
  --cpus-per-task=4 --mem=60G --time=02:00:00 --job-name=mdlm-exploration-v2
  --chdir="$PROJECT_DIR" --export=ALL
  --output="${LOG_DIR}/%x-%j.out" --error="${LOG_DIR}/%x-%j.err" "$job")
echo 'Exploration: A N64/R6/MH4; B N64/R6/MH8; C N128/R6/MH4; uniform tilt increments of 0.5.'
echo 'Eight runs each, 24 total, interleaved. Two thresholds, one population, two estimators.'
echo 'Resources: THREE GPUs on one node, 12 CPU total, 60 GB, two-hour hard limit; resumable, no resubmission.'
echo "Output: ${OUTPUT_ROOT}/camera_ready_exploration_v2; previous pilot preserved."
echo "Scientific command per GPU: python -m diffusion_toxicity_smc_experiment.exploratory run --study ${EXPERIMENT_DIR}/configs/mdlm_owt_exploration_v2.yaml --output-root ${OUTPUT_ROOT} --max-seconds 6600 --workers 3 --worker-id WORKER_ID"
printf 'command=';printf '%q ' "${command[@]}";printf '\n'
if [[ $# -eq 0 ]]; then echo 'DRY RUN: no submission.';exit 0;fi
for path in "$job" "$VENV_DIR/bin/activate" "$HF_HOME/hub" "$LOG_DIR" "$OUTPUT_ROOT" \
  "${EXPERIMENT_DIR}/configs/mdlm_owt_exploration_v2.yaml"; do
  [[ -e "$path" ]] || { echo "Missing $path" >&2;exit 1; }
done
active=$(squeue -h -u stuhvh -o '%i|%j|%T')
if [[ -n "$active" ]]; then printf 'STOP: active jobs, inspect before submitting:\n%s\n' "$active";exit 3;fi
result="$("${command[@]}")"
[[ "${result%%;*}" =~ ^[0-9]+$ ]] || { echo "Unexpected submission response: $result. Check squeue before any retry." >&2;exit 1; }
echo "job_id=${result%%;*}"
