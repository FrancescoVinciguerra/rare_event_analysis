#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--submit") ]]; then
  echo "Usage: submit_avon_final_experiment.sh [--submit]" >&2
  exit 2
fi
submit=false
if [[ $# -eq 1 ]]; then
  submit=true
fi

: "${AVON_ACCOUNT:?Set AVON_ACCOUNT from verified preflight output}"
: "${AVON_QOS:?Set AVON_QOS from verified preflight output}"
: "${AVON_GPU_PARTITION:?Set AVON_GPU_PARTITION from verified preflight output}"
: "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
: "${AVON_MODULES:?Set AVON_MODULES from verified module metadata}"

export PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
export VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
export HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiments/mdlm_toxicity_rare_event}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
export FINAL_MATCHED_SUITE_CONFIG="${FINAL_MATCHED_SUITE_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_submission_matched_runtime.yaml}"
export FINAL_CONFIRMATORY_SUITE_CONFIG="${FINAL_CONFIRMATORY_SUITE_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_final_confirmatory.yaml}"
export FINAL_ANALYSIS_CONFIG="${FINAL_ANALYSIS_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_final_analysis.yaml}"
job_script="${EXPERIMENT_DIR}/scripts/slurm_avon_final_experiment.sh"

command=(
  sbatch --parsable
  --account="${AVON_ACCOUNT}"
  --partition="${AVON_GPU_PARTITION}"
  --qos="${AVON_QOS}"
  --gres="${AVON_GPU_GRES}"
  --nodes=1
  --ntasks=1
  --cpus-per-task=4
  --mem=20G
  --time=2-00:00:00
  --chdir="${PROJECT_DIR}"
  --job-name=mdlm-final-confirmatory
  --export=ALL
  "${job_script}"
)

echo "mode=$([[ "${submit}" == true ]] && echo submit || echo dry-run)"
echo "cluster=avon"
echo "account=${AVON_ACCOUNT}"
echo "partition=${AVON_GPU_PARTITION}"
echo "qos=${AVON_QOS}"
echo "gres=${AVON_GPU_GRES}"
echo "cpus_per_task=4"
echo "memory=20G"
echo "walltime=2-00:00:00"
echo "project=${PROJECT_DIR}"
echo "suite=${FINAL_MATCHED_SUITE_CONFIG}"
echo "confirmatory_suite=${FINAL_CONFIRMATORY_SUITE_CONFIG}"
echo "analysis=${FINAL_ANALYSIS_CONFIG}"
echo "outputs=${OUTPUT_ROOT}"
echo "logs=${LOG_DIR}/mdlm-final-confirmatory-%j.{out,err}"
printf 'command='
printf '%q ' "${command[@]}"
printf '\n'

if [[ "${submit}" != true ]]; then
  echo "Dry run only: no job submitted."
  exit 0
fi

for required_path in \
  "${PROJECT_DIR}" \
  "${VENV_DIR}/bin/activate" \
  "${HF_HOME}" \
  "${OUTPUT_ROOT}" \
  "${LOG_DIR}" \
  "${FINAL_MATCHED_SUITE_CONFIG}" \
  "${FINAL_CONFIRMATORY_SUITE_CONFIG}" \
  "${FINAL_ANALYSIS_CONFIG}" \
  "${job_script}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path is missing: ${required_path}" >&2
    exit 1
  fi
done

submission_output="$("${command[@]}")"
job_id="${submission_output%%;*}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch response: ${submission_output}" >&2
  echo "A job may exist; inspect squeue before retrying." >&2
  exit 1
fi
echo "job_id=${job_id}"
echo "Submitted exactly one job; existing artifacts are resume-only and are not overwritten."
