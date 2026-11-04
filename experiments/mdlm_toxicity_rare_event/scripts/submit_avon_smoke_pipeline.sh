#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_avon_smoke_pipeline.sh technical|scientific --dry-run
  submit_avon_smoke_pipeline.sh technical|scientific --stage STAGE [--afterok JOB_ID|--no-dependency] [--dry-run]

STAGE is one of: prefetch, direct, score, smc, compare.
Each non-dry-run invocation submits at most one job. For stages after prefetch,
provide --afterok JOB_ID, or use --no-dependency only after manually verifying
that the prerequisite stage completed successfully.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

mode="$1"
shift
stage=""
afterok_job_id=""
dry_run=false
no_dependency=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      if [[ $# -lt 2 ]]; then
        echo "--stage requires a value" >&2
        exit 2
      fi
      stage="$2"
      shift 2
      ;;
    --afterok)
      if [[ $# -lt 2 ]]; then
        echo "--afterok requires one numeric SLURM job ID" >&2
        exit 2
      fi
      afterok_job_id="$2"
      shift 2
      ;;
    --no-dependency)
      no_dependency=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --submit)
      echo "Multi-job --submit mode was removed to respect Avon submission limits." >&2
      echo "Submit exactly one job with --stage STAGE." >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -n "${afterok_job_id}" && "${no_dependency}" == true ]]; then
  echo "Use either --afterok or --no-dependency, not both." >&2
  exit 2
fi
if [[ -n "${afterok_job_id}" && ! "${afterok_job_id}" =~ ^[0-9]+$ ]]; then
  echo "--afterok must be one numeric SLURM job ID." >&2
  exit 2
fi
if [[ -z "${stage}" && "${dry_run}" != true ]]; then
  dry_run=true
fi

: "${AVON_ACCOUNT:?Set AVON_ACCOUNT from sacctmgr/scontrol preflight output}"
: "${AVON_PARTITION:?Set AVON_PARTITION from sinfo/scontrol preflight output}"
: "${AVON_GPU_GRES:?Set AVON_GPU_GRES from sinfo/scontrol preflight output}"
: "${AVON_MODULES:?Set AVON_MODULES from module spider output}"

case "${mode}" in
  technical)
    config_name="mdlm_owt_unconditional_smoke_technical.yaml"
    direct_time="00:30:00"
    smc_time="00:45:00"
    ;;
  scientific)
    config_name="mdlm_owt_unconditional_smoke_scientific.yaml"
    direct_time="04:00:00"
    smc_time="12:00:00"
    ;;
  *)
    usage
    exit 2
    ;;
esac

export PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
export VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
export HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiments/mdlm_toxicity_rare_event}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"
export BENCHMARK_CONFIG="${EXPERIMENT_DIR}/configs/${config_name}"
export AVON_ACCOUNT AVON_PARTITION AVON_GPU_GRES AVON_MODULES
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

for path in "${PROJECT_DIR}" "${BENCHMARK_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done
if [[ "${dry_run}" != true ]]; then
  for path in "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_HOME}" "${VENV_DIR}/bin/activate"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required submission path is missing: ${path}" >&2
      exit 1
    fi
  done
fi

common=(
  --account="${AVON_ACCOUNT}"
  --partition="${AVON_PARTITION}"
  --gres="${AVON_GPU_GRES}"
  --export=ALL
)
if [[ -n "${AVON_QOS:-}" ]]; then
  common+=(--qos="${AVON_QOS}")
fi

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

stage_spec() {
  case "$1" in
    prefetch)
      stage_script="${EXPERIMENT_DIR}/scripts/slurm_avon_prefetch_models.sh"
      stage_time="00:45:00"
      prerequisite=""
      ;;
    direct)
      stage_script="${EXPERIMENT_DIR}/scripts/slurm_avon_smoke_direct.sh"
      stage_time="${direct_time}"
      prerequisite="prefetch"
      ;;
    score)
      stage_script="${EXPERIMENT_DIR}/scripts/slurm_avon_smoke_score.sh"
      stage_time="01:00:00"
      prerequisite="direct"
      ;;
    smc)
      stage_script="${EXPERIMENT_DIR}/scripts/slurm_avon_smoke_smc.sh"
      stage_time="${smc_time}"
      prerequisite="score"
      ;;
    compare)
      stage_script="${EXPERIMENT_DIR}/scripts/slurm_avon_smoke_compare.sh"
      stage_time="00:15:00"
      prerequisite="smc"
      ;;
    *)
      echo "Unknown stage: $1" >&2
      usage
      exit 2
      ;;
  esac
}

print_header() {
  echo "Avon ${mode} smoke plan"
  echo "  account=${AVON_ACCOUNT}"
  echo "  partition=${AVON_PARTITION}"
  echo "  qos=${AVON_QOS:-<none>}"
  echo "  gres=${AVON_GPU_GRES}"
  echo "  project=${PROJECT_DIR}"
  echo "  config=${BENCHMARK_CONFIG}"
  echo "  outputs=${OUTPUT_ROOT}"
  echo "  logs=${LOG_DIR}/%x-%j.{out,err}"
  echo "  direct target=2 IID unconditional samples (resume-safe)"
}

print_stage_plan() {
  local requested_stage="$1"
  local dependency_job_id="${2:-}"
  stage_spec "${requested_stage}"
  local command=("${SBATCH_BIN}" --parsable "${common[@]}")
  if [[ -n "${prerequisite}" ]]; then
    if [[ -n "${dependency_job_id}" ]]; then
      command+=(--dependency="afterok:${dependency_job_id}")
    elif [[ "${no_dependency}" != true ]]; then
      command+=(--dependency="afterok:<${prerequisite^^}_JOB_ID>")
    fi
  fi
  command+=(--time="${stage_time}" "${stage_script}")
  echo "  stage=${requested_stage}; prerequisite=${prerequisite:-none}; walltime=${stage_time}"
  print_command "${command[@]}"
}

print_header

if [[ -z "${stage}" ]]; then
  for planned_stage in prefetch direct score smc compare; do
    print_stage_plan "${planned_stage}"
  done
  echo "Dry run only: no job was submitted. Submit one stage per invocation."
  exit 0
fi

stage_spec "${stage}"
if [[ -z "${prerequisite}" && -n "${afterok_job_id}" ]]; then
  echo "The prefetch stage has no prerequisite; remove --afterok." >&2
  exit 2
fi
if [[ -n "${prerequisite}" && -z "${afterok_job_id}" && "${no_dependency}" != true && "${dry_run}" != true ]]; then
  echo "Stage ${stage} requires --afterok JOB_ID for ${prerequisite}, or explicit --no-dependency." >&2
  exit 2
fi

print_stage_plan "${stage}" "${afterok_job_id}"
if [[ "${dry_run}" == true ]]; then
  echo "Dry run only: no job was submitted."
  exit 0
fi

submit_command=("${SBATCH_BIN}" --parsable "${common[@]}")
if [[ -n "${afterok_job_id}" ]]; then
  submit_command+=(--dependency="afterok:${afterok_job_id}")
fi
submit_command+=(--time="${stage_time}" "${stage_script}")

echo "Submitting exactly one stage: ${stage}"
set +e
submission_output=$("${submit_command[@]}" 2>&1)
submission_status=$?
set -e
if [[ ${submission_status} -ne 0 ]]; then
  echo "SLURM submission failed for stage ${stage}:" >&2
  echo "${submission_output}" >&2
  echo "Successfully recorded jobs from this invocation: none." >&2
  echo "No job ID was returned; inspect squeue before retrying in case the client lost the response." >&2
  if [[ -n "${afterok_job_id}" ]]; then
    echo "Existing prerequisite job ${afterok_job_id} was not modified or cancelled; inspect it explicitly." >&2
  fi
  exit "${submission_status}"
fi

submission_output=${submission_output//$'\r'/}
job_id=${submission_output%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an unexpected response: ${submission_output}" >&2
  echo "A job may have been submitted; inspect squeue before retrying." >&2
  exit 1
fi

echo "stage=${stage}"
echo "job_id=${job_id}"
if [[ -n "${afterok_job_id}" ]]; then
  echo "dependency=afterok:${afterok_job_id}"
else
  echo "dependency=none"
fi
echo "Monitor: squeue -j ${job_id}"
