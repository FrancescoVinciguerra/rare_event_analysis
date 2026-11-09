#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_avon_serious_comparison.sh STAGE [--submit]

STAGE is exactly one of:
  method-validation Validate all three real GPU kernels at N=16 and N=64.
  direct-reference  Generate/resume IID direct samples through N=200000.
  score-reference   Score every persisted direct sample; never regenerate.
  smc-grid          Run/resume 3 methods x 4 budgets x 8 independent runs.
  report            Read completed artifacts and create CSV/JSON/SVG reports.

The default is a dry-run. Each --submit invocation submits at most one job.
There is deliberately no dependency chain or submit-all mode.
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
export SERIOUS_SUITE_CONFIG="${SERIOUS_SUITE_CONFIG:-${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_serious_comparison.yaml}"

case "$stage" in
  method-validation)
    : "${AVON_GPU_DEVEL_PARTITION:?Set AVON_GPU_DEVEL_PARTITION from verified preflight output}"
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SERIOUS_SUITE_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_three_smc_gpu_validation.yaml"
    job_name=mdlm-three-smc-validation
    partition="${AVON_GPU_DEVEL_PARTITION}"
    gres="${AVON_GPU_GRES}"
    cpus=4
    memory=20G
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_avon_serious_smc.sh"
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  direct-reference)
    : "${AVON_GPU_PARTITION:?Set AVON_GPU_PARTITION from verified preflight output}"
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export TARGET_SAMPLES=200000
    export DIRECT_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct_32.yaml"
    job_name=mdlm-direct32-200000
    partition="${AVON_GPU_PARTITION}"
    gres="${AVON_GPU_GRES}"
    cpus=4
    memory=16G
    walltime=12:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_generate_direct.sh"
    extra_sbatch=(--array=0)
    log_pattern="${LOG_DIR}/${job_name}-%A_%a.{out,err}"
    ;;
  score-reference)
    : "${AVON_GPU_PARTITION:?Set AVON_GPU_PARTITION from verified preflight output}"
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export EVALUATOR_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_estimation_100tok.yaml"
    job_name=mdlm-score-direct32-200000
    partition="${AVON_GPU_PARTITION}"
    gres="${AVON_GPU_GRES}"
    cpus=4
    memory=16G
    walltime=04:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_score_direct.sh"
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  smc-grid)
    : "${AVON_GPU_PARTITION:?Set AVON_GPU_PARTITION from verified preflight output}"
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    job_name=mdlm-three-smc-serious
    partition="${AVON_GPU_PARTITION}"
    gres="${AVON_GPU_GRES}"
    cpus=4
    memory=20G
    walltime=2-00:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_avon_serious_smc.sh"
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  report)
    : "${AVON_CPU_PARTITION:?Set AVON_CPU_PARTITION from verified preflight output}"
    job_name=mdlm-three-smc-report
    partition="${AVON_CPU_PARTITION}"
    gres=""
    cpus=2
    memory=4G
    walltime=00:30:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_avon_serious_report.sh"
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
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
command+=("${extra_sbatch[@]}" "$job_script")

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
echo "suite=${SERIOUS_SUITE_CONFIG}"
echo "outputs=${OUTPUT_ROOT}"
echo "logs=${log_pattern}"
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
  "${HF_HOME}" \
  "${VENV_DIR}/bin/activate" \
  "${SERIOUS_SUITE_CONFIG}" \
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
echo "Submitted exactly one job; submit no later stage until it completes successfully."
