#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_avon_submission_probability.sh STAGE [--submit]

STAGE is exactly one of:
  pilot     Re-threshold existing artifacts and freeze pilot component weights.
  long-smc  Run 32 new independent tests for FK and Trajectory MH at N=512.
  report    Compare uniform and frozen-pilot aggregation on the new test runs.

The default is a dry-run. Each --submit invocation submits at most one job.
There is no dependency chain or submit-all mode.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi
stage="$1"
submit=false
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--submit" ]] || { usage; exit 2; }
  submit=true
fi

: "${AVON_ACCOUNT:?Set AVON_ACCOUNT from verified preflight output}"
: "${AVON_QOS:?Set AVON_QOS from verified preflight output}"
: "${AVON_MODULES:?Set AVON_MODULES from verified module metadata}"

export PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
export VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
export HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/mdlm}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
export SUBMISSION_STUDY_CONFIG="${SUBMISSION_STUDY_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_submission_probability.yaml}"
export SUBMISSION_LONG_SUITE_CONFIG="${SUBMISSION_LONG_SUITE_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_submission_long.yaml}"

case "$stage" in
  pilot)
    : "${AVON_CPU_PARTITION:?Set AVON_CPU_PARTITION from verified preflight output}"
    export SUBMISSION_PROBABILITY_COMMAND=pilot
    job_name=mdlm-submission-pilot
    partition="${AVON_CPU_PARTITION}"
    gres=""
    cpus=2
    memory=4G
    walltime=00:20:00
    job_script="${EXPERIMENT_DIR}/reproduction/scripts/slurm_avon_submission_probability_cpu.sh"
    ;;
  long-smc)
    : "${AVON_GPU_PARTITION:?Set AVON_GPU_PARTITION from verified preflight output}"
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    job_name=mdlm-submission-long-smc
    partition="${AVON_GPU_PARTITION}"
    gres="${AVON_GPU_GRES}"
    cpus=4
    memory=20G
    walltime=2-00:00:00
    job_script="${EXPERIMENT_DIR}/reproduction/scripts/slurm_avon_submission_long_smc.sh"
    ;;
  report)
    : "${AVON_CPU_PARTITION:?Set AVON_CPU_PARTITION from verified preflight output}"
    export SUBMISSION_PROBABILITY_COMMAND=report
    job_name=mdlm-submission-report
    partition="${AVON_CPU_PARTITION}"
    gres=""
    cpus=2
    memory=4G
    walltime=00:30:00
    job_script="${EXPERIMENT_DIR}/reproduction/scripts/slurm_avon_submission_probability_cpu.sh"
    ;;
  *)
    usage
    exit 2
    ;;
esac

command=(
  sbatch --parsable
  --account="${AVON_ACCOUNT}"
  --partition="${partition}"
  --qos="${AVON_QOS}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${cpus}"
  --mem="${memory}"
  --chdir="${PROJECT_DIR}"
  --job-name="${job_name}"
  --time="${walltime}"
  --export=ALL
)
if [[ -n "$gres" ]]; then
  command+=(--gres="$gres")
fi
command+=("$job_script")

echo "stage=${stage}"
echo "mode=$([[ "$submit" == true ]] && echo submit || echo dry-run)"
echo "account=${AVON_ACCOUNT}"
echo "partition=${partition}"
echo "qos=${AVON_QOS}"
echo "gres=${gres:-none}"
echo "cpus_per_task=${cpus}"
echo "memory=${memory}"
echo "walltime=${walltime}"
echo "project=${PROJECT_DIR}"
echo "study=${SUBMISSION_STUDY_CONFIG}"
echo "long_suite=${SUBMISSION_LONG_SUITE_CONFIG}"
echo "outputs=${OUTPUT_ROOT}"
echo "logs=${LOG_DIR}/${job_name}-%j.{out,err}"
printf 'command='
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$submit" != true ]]; then
  echo "Dry run only: no job submitted."
  exit 0
fi

for required_path in \
  "${PROJECT_DIR}" \
  "${LOG_DIR}" \
  "${OUTPUT_ROOT}" \
  "${VENV_DIR}/bin/activate" \
  "${SUBMISSION_STUDY_CONFIG}" \
  "${SUBMISSION_LONG_SUITE_CONFIG}" \
  "${job_script}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required submission path is missing: ${required_path}" >&2
    exit 1
  fi
done

submission_output="$("${command[@]}")"
job_id="${submission_output%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch response: ${submission_output}" >&2
  echo "A job may exist; inspect squeue before any retry." >&2
  exit 1
fi
echo "job_id=${job_id}"
echo "Submitted exactly one job. Do not submit the next stage until this one succeeds."

