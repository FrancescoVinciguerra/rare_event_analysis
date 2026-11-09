#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "${EXPERIMENT_DIR:?Set the frozen experiment directory}/scripts/avon_job_common.sh"
avon_activate_environment
avon_print_cpu_diagnostics
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 --cpu-bind=cores python "${CPU_PILOT_DIR:?Set the pilot directory}/cpu_timing_pilot.py" \
  --study "$EXPERIMENT_DIR/configs/mdlm_owt_low_particle_budget.yaml" --output "$CPU_PILOT_DIR"
