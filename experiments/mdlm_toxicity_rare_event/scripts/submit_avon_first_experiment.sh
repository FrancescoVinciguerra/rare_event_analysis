#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_avon_first_experiment.sh STAGE [--submit]

STAGE is one of:
  direct-256          Generate/resume the first 256 direct samples.
  score-direct        Score every currently stored direct sample.
  lambda-calibration  Run lambda=0 plus calibration lambdas; never aggregate.
  progressive-calibration
                      Run the dense lambda schedule through 2; never aggregate.
  estimation-100tok  Run four independent 100-token SMC estimators and aggregate.
  estimation-100tok-trajectory-mh
                      Run the same estimator with trajectory resample-move.
  estimation-100tok-trajectory-mh-matched-fk
                      Run the matched-cost FK comparison profile.
  calibration-report  Compare calibration components and lambda=0 to direct.
  direct-1000         Extend the same direct run to 1000 samples.

The default is a dry-run. Each --submit invocation submits exactly one job.
The 10,000-sample extension and provisional estimation template are
intentionally not submit-capable here; both require explicit review first.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

stage="$1"
shift
submit=false
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && "$1" == "--submit" ]]; then
    submit=true
  else
    usage
    exit 2
  fi
fi

: "${AVON_ACCOUNT:?Set AVON_ACCOUNT from verified preflight output}"
: "${AVON_PARTITION:?Set AVON_PARTITION from verified preflight output}"
: "${AVON_MODULES:?Set AVON_MODULES from verified module metadata}"

export PROJECT_DIR="${PROJECT_DIR:-/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code}"
export VENV_DIR="${VENV_DIR:-/gpfs/home/stats/stuhvh/venvs/toxicity-smc}"
export HF_HOME="${HF_HOME:-/gpfs/home/stats/stuhvh/hf-cache}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiments/mdlm_toxicity_rare_event}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/artifacts/current}"
export LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/artifacts/logs}"

case "$stage" in
  direct-256)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export TARGET_SAMPLES=256
    export DIRECT_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct_32.yaml"
    job_name=mdlm-direct32-256
    walltime=00:45:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_generate_direct.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=16G
    extra_sbatch=(--array=0)
    log_pattern="${LOG_DIR}/${job_name}-%A_%a.{out,err}"
    ;;
  score-direct)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export EVALUATOR_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct_32.yaml"
    job_name=mdlm-score-direct32
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_score_direct.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=16G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  lambda-calibration)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SMC_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_lambda_calibration.yaml"
    job_name=mdlm-lambda-calibration
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_run_smc.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=16G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  progressive-calibration)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SMC_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_lambda_progressive.yaml"
    job_name=mdlm-lambda-progressive
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_run_smc.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=16G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  estimation-100tok)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SMC_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_estimation_100tok.yaml"
    job_name=mdlm-estimation-100tok
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_run_smc.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=20G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  estimation-100tok-trajectory-mh)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SMC_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
    job_name=mdlm-estimation-100tok-trajectory-mh
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_run_smc.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=20G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  estimation-100tok-trajectory-mh-matched-fk)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export SMC_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh_matched_fk.yaml"
    job_name=mdlm-estimation-100tok-trajectory-mh-matched-fk
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_run_smc.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=20G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  calibration-report)
    export CALIBRATION_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_lambda_calibration.yaml"
    job_name=mdlm-calibration-report
    walltime=00:10:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_avon_calibration_report.sh"
    stage_partition=devel
    stage_gres=""
    stage_cpus=2
    stage_mem=4G
    extra_sbatch=()
    log_pattern="${LOG_DIR}/${job_name}-%j.{out,err}"
    ;;
  direct-1000)
    : "${AVON_GPU_GRES:?Set AVON_GPU_GRES from verified preflight output}"
    export TARGET_SAMPLES=1000
    export DIRECT_CONFIG="${EXPERIMENT_DIR}/configs/mdlm_owt_unconditional_direct_32.yaml"
    job_name=mdlm-direct32-1000
    walltime=01:00:00
    job_script="${EXPERIMENT_DIR}/scripts/slurm_generate_direct.sh"
    stage_partition="${AVON_PARTITION}"
    stage_gres="${AVON_GPU_GRES}"
    stage_cpus=4
    stage_mem=16G
    extra_sbatch=(--array=0)
    log_pattern="${LOG_DIR}/${job_name}-%A_%a.{out,err}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

command=(
  sbatch --parsable
  --account="${AVON_ACCOUNT}"
  --partition="${stage_partition}"
  --qos="${AVON_QOS:-normal}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${stage_cpus}"
  --mem="${stage_mem}"
  --chdir="${PROJECT_DIR}"
  --job-name="${job_name}"
  --time="${walltime}"
  --export=ALL
)
if [[ -n "${stage_gres}" ]]; then
  command+=(--gres="${stage_gres}")
fi
command+=("${extra_sbatch[@]}" "${job_script}")

echo "stage=${stage}"
echo "mode=$([[ "$submit" == true ]] && echo submit || echo dry-run)"
echo "account=${AVON_ACCOUNT}"
echo "partition=${stage_partition}"
echo "qos=${AVON_QOS:-normal}"
echo "gres=${stage_gres:-none}"
echo "cpus_per_task=${stage_cpus}"
echo "memory=${stage_mem}"
echo "project=${PROJECT_DIR}"
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
echo "Submitted exactly one job. Monitor with: squeue -j ${job_id}"
