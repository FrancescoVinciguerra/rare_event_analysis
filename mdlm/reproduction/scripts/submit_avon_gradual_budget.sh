#!/usr/bin/env bash
# Exactly one reviewed submission; no automatic retries. Dry run by default.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --submit ) ]]; then
  echo 'Usage: submit_avon_gradual_budget.sh [--submit]' >&2; exit 2
fi
: "${AVON_ACCOUNT:?Verify current account}"
: "${AVON_GPU_PARTITION:?Verify current partition}"
: "${AVON_QOS:?Verify current QoS}"
: "${AVON_GPU_GRES:?Verify current GRES}"
: "${AVON_MODULES:?Verify modules}"
: "${PROJECT_DIR:?Verify project directory}"
: "${VENV_DIR:?Verify environment}"
: "${HF_HOME:?Verify model cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/mdlm}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
[[ "$AVON_GPU_GRES" == 'gpu:quadro_rtx_6000:3' ]] || { echo 'Reviewed pilot requests three GPUs' >&2; exit 2; }
job="$EXPERIMENT_DIR/reproduction/scripts/slurm_avon_gradual_budget.sh"
command=(sbatch --parsable --no-requeue --account="$AVON_ACCOUNT" --partition="$AVON_GPU_PARTITION" --qos="$AVON_QOS"
  --gres="$AVON_GPU_GRES" --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=60G --time=17:00:00
  --job-name=mdlm-gradual-budget --chdir="$PROJECT_DIR" --export=ALL
  --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "$job")
echo 'Resources: 3 GPUs on one node, 12 CPUs total, 60 GB, 17h hard limit. Checkpoints retained, no automatic resubmission.'
echo "Output: $OUTPUT_ROOT/camera_ready_gradual_budget; old results preserved."
printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
if [[ $# -eq 0 ]]; then echo 'DRY RUN: no submission.'; exit 0; fi
for path in "$job" "$VENV_DIR/bin/activate" "$HF_HOME/hub" "$LOG_DIR" "$OUTPUT_ROOT" \
  "$EXPERIMENT_DIR/configs/mdlm_owt_gradual_budget.yaml"; do
  [[ -e "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
active=$(squeue -h -u stuhvh -o '%i|%j|%T')
if [[ -n "$active" ]]; then printf 'STOP: active jobs; inspect before submission:\n%s\n' "$active"; exit 3; fi
result="$("${command[@]}")"
[[ "${result%%;*}" =~ ^[0-9]+$ ]] || { echo "Unexpected submission response: $result. Check squeue before any retry." >&2; exit 1; }
echo "job_id=${result%%;*}"
