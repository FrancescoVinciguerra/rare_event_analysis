#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --submit ) ]] || exit 2
: "${PROJECT_DIR:?}" "${AVON_ACCOUNT:?}" "${AVON_GPU_PARTITION:?}" "${AVON_QOS:?}" "${AVON_MODULES:?}" "${VENV_DIR:?}" "${HF_HOME:?}"
export EXPERIMENT_DIR="$PROJECT_DIR/mdlm"
export OUTPUT_ROOT="$EXPERIMENT_DIR/artifacts/current" LOG_DIR="$EXPERIMENT_DIR/artifacts/logs"
command=(sbatch --parsable --account="$AVON_ACCOUNT" --partition="$AVON_GPU_PARTITION" --qos="$AVON_QOS"
  --nodes=1 --nodelist=gpu007 --gres=gpu:quadro_rtx_6000:3 --ntasks=3 --cpus-per-task=4
  --mem=60G --time=01:00:00 --job-name=mdlm-wall-clock --chdir="$PROJECT_DIR" --export=ALL
  --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "$EXPERIMENT_DIR/reproduction/scripts/slurm_avon_wall_clock.sh")
echo 'MC-only follow-up: reuse all 24 SMC populations and their measured per-run times; new independent crude seeds.'
echo 'Three GPUs on ORIGINAL node gpu007, 12 CPU total, 60 GB, one-hour maximum. No SMC rerun.'
printf 'command=';printf '%q ' "${command[@]}";printf '\n'
if [[ $# -eq 0 ]]; then echo 'DRY RUN ONLY';exit 0;fi
[[ -z "$(squeue -h -u stuhvh)" ]] || { echo 'STOP: inspect active jobs before submission';exit 3; }
result="$("${command[@]}")";[[ "${result%%;*}" =~ ^[0-9]+$ ]] || exit 1
echo "job_id=${result%%;*}"
