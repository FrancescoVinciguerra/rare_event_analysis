#!/usr/bin/env bash
# Dry-run by default. Submit one pilot job only after reviewed preflight.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --submit ) ]]; then
  echo 'Usage: submit_avon_camera_ready_low_budget.sh [--submit]' >&2; exit 2
fi
: "${AVON_ACCOUNT:?Verify account in current preflight}"
: "${AVON_GPU_PARTITION:?Verify GPU partition in current preflight}"
: "${AVON_QOS:?Verify QoS in current preflight}"
: "${AVON_GPU_GRES:?Verify GPU GRES in current preflight}"
: "${AVON_MODULES:?Verify modules in current preflight}"
: "${PROJECT_DIR:?Verify remote project path}"
: "${VENV_DIR:?Verify remote environment path}"
: "${HF_HOME:?Verify remote model cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/mdlm}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
job="${EXPERIMENT_DIR}/reproduction/scripts/slurm_avon_camera_ready_low_budget.sh"
command=(sbatch --parsable --account="${AVON_ACCOUNT}" --partition="${AVON_GPU_PARTITION}"
  --qos="${AVON_QOS}" --gres="${AVON_GPU_GRES}" --nodes=1 --ntasks=1
  --cpus-per-task=4 --mem=20G --time=02:00:00 --job-name=mdlm-camera-low-budget
  --chdir="${PROJECT_DIR}" --export=ALL
  --output="${LOG_DIR}/%x-%j.out" --error="${LOG_DIR}/%x-%j.err" "$job")
echo 'Pilot: 8 independent populations at each N=16,32,64,128,256; tilts 1,2,3, four MH moves per level; score thresholds 0.9,0.99.'
echo 'Resources: 1 GPU, 4 CPU, 20 GB host RAM, 2 hours maximum.'
echo "Working directory: ${PROJECT_DIR}"
echo "Results: ${OUTPUT_ROOT}/camera_ready_low_budget_v1"
echo "Scientific command: python -m diffusion_toxicity_smc_experiment.low_budget run --study ${EXPERIMENT_DIR}/configs/mdlm_owt_camera_ready_low_budget.yaml --output-root ${OUTPUT_ROOT}"
printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
if [[ $# -eq 0 ]]; then echo 'Dry run only: no job submitted.'; exit 0; fi
for path in "$job" "${LOG_DIR}" "${OUTPUT_ROOT}" "${VENV_DIR}/bin/activate" \
  "${EXPERIMENT_DIR}/configs/mdlm_owt_camera_ready_low_budget.yaml"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done
result="$("${command[@]}")"
[[ "${result%%;*}" =~ ^[0-9]+$ ]] || { echo "Unexpected submission output; inspect squeue before retry: $result" >&2; exit 1; }
echo "job_id=${result%%;*}"
