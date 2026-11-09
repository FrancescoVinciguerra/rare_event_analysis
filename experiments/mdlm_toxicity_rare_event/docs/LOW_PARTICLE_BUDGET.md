# Same-ladder SMC with 64 and 96 particles

Run one RollBack SMC ladder per seed and size. Use 24 independent SMC seeds
per size, for 48 ladders. The schedule is 0.5, 1, 1.5, 2, 2.5 and 3, with
logit potential and four native RollBack–MH moves per level. Use systematic
resampling at ESS/N below 0.5 and suppress resampling at the final level.

Report q ≥ 0.99 at tilt 0.5, q ≥ 0.995 at tilt 1, and q ≥ 0.999 at
tilts 0.5 and 3. Every estimate uses the actual population after the fourth
MH move at its level. Record synchronized cumulative times and native
transition counts. The crude prefix for each estimate has length equal to
the floor of measured SMC seconds divided by calibrated crude seconds per
path on the same GPU. Use the frozen 121,355-path MC pool and disjoint
blocks per replicate rank. Reuse nested prefixes across checkpoints.
The comparison also records the exact native-transition budget.

Report zero counts, estimates within a factor of ten of the reference,
absolute and logarithmic relative errors, and relative standard deviation.
The working threshold for scale recovery is at least 12 of 24 runs within
a factor of ten. Report the crude budget as a fraction of 1/p_ref.

## Submission on Avon

The checksummed source archive originates in the Windows workspace
`C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main`. It goes to
`stuhvh@avon.scrtp.warwick.ac.uk` via Hammer. The batch job runs in
`/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code`. The new result
directory is `experiments/mdlm_toxicity_rare_event/artifacts/current/camera_ready_low_particle_budget`.
Its `code` subdirectory stores the frozen source. The job writes per-seed
checkpoints, `runs.csv`, `summary.csv`, completion status and a result archive.
Existing result directories are separate. Only scheduler queries and file
transfer use the login node.

```bash
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export OUTPUT_ROOT="$PROJECT_DIR/experiments/mdlm_toxicity_rare_event/artifacts/current"
export EXPERIMENT_DIR="$OUTPUT_ROOT/camera_ready_low_particle_budget/code"
export CALIBRATION_ROOT="$OUTPUT_ROOT/camera_ready_same_run_ladder"
export LOG_DIR="$PROJECT_DIR/experiments/mdlm_toxicity_rare_event/artifacts/logs"
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
export STUDY_CONFIG=mdlm_owt_low_particle_budget.yaml
export RESULT_NAME=camera_ready_low_particle_budget
export SOFT_MAX_SECONDS=6900
sbatch --parsable --no-requeue --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --nodelist=gpu007 --ntasks=3 --cpus-per-task=4 --mem=60G --time=02:00:00 --job-name=mdlm-low-particle-budget --chdir="$PROJECT_DIR" --export=ALL --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "$EXPERIMENT_DIR/scripts/slurm_avon_particle_scaling.sh"
```

Each of the three worker tasks gets one GPU, four CPUs and 20 GB, and runs
the following command with its worker ID.

```bash
python -m diffusion_toxicity_smc_experiment.same_run_reuse run-fresh --study "$EXPERIMENT_DIR/configs/$STUDY_CONFIG" --output-root "$OUTPUT_ROOT" --calibration-root "$CALIBRATION_ROOT" --max-seconds "$SOFT_MAX_SECONDS" --workers 3 --worker-id "$worker"
```
