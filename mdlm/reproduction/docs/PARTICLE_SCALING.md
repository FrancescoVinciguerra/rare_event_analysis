# Population size at fixed tilt

Compare the non-normalized RollBack SMC estimator with crude Monte Carlo
as the population grows from 128 to 160, 192 and 256. Use 24 independent
SMC runs per size and terminal tilt. The earlier 12-run logit ladder serves
as pilot data. The new campaign uses distinct seeds.

- Terminal tilt 0.5 for score thresholds 0.99 and 0.995.
- Terminal tilt 2.5 for score threshold 0.999.
- Both terminal populations also evaluate the other thresholds for secondary
  analysis. The two tilts use separate runs with no resampling at their
  terminal levels. The earlier pilot's intermediate populations could
  resample at those levels before continuing.
- Keep logit, half-step tempering, four RollBack–MH moves per level, uniform
  native rollback depths, and systematic resampling at ESS/N below 0.5.
- Each of the eight configurations has 24 new SMC seeds, for 192 runs.
  Each worker handles eight replicates of each configuration.
- Estimate crude MC from the 121,355 saved native trajectories of the
  completed accuracy-budget pilot. Use disjoint blocks of 5,056 trajectories
  across replicate ranks. Within each rank, reuse prefixes across sizes and
  tilts. The number of trajectories in a prefix is the floor of synchronized
  SMC time divided by the previously calibrated MC seconds per trajectory
  on the same physical GPU with four CPU cores. If a budget exceeds its
  block, report the mismatch instead of silently capping it.
- Reference counts are 444/200,000, 334/298,800, and 26/200,000. The
  accuracy-budget MC streams differ from the gradual-budget MC streams
  used to obtain the reference for 0.995.
- Reconstruct log estimates from the saved log normalizer, log weights
  and potentials. Compute the log relative error with `expm1` and
  `log1p` on the negative log-ratio branch. Preserve zero estimates.
- Report per-run ratios and log errors, RSD, mean and median, zero counts,
  and the number within a factor of ten, grouped by fixed size and tilt.

## Avon submission

Origin: `C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main`.
Destination: `stuhvh@avon.scrtp.warwick.ac.uk` via Hammer.
Working directory: `/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code`.
The deployment uploads a checksummed source archive into
`mdlm/artifacts/current/camera_ready_particle_scaling/code`
and creates a separate result directory. This contains worker manifests,
atomic per-run population files, CSVs and validation of campaign completion.
The result archive is `camera_ready_particle_scaling_results.tar.gz` alongside
the result directory. Scheduler logs go in the existing `artifacts/logs`.
No model inference or analysis runs on the login node.

```bash
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export OUTPUT_ROOT="$PROJECT_DIR/mdlm/artifacts/current"
export EXPERIMENT_DIR="$OUTPUT_ROOT/camera_ready_particle_scaling/code"
export CALIBRATION_ROOT="$OUTPUT_ROOT/camera_ready_same_run_ladder"
export LOG_DIR="$PROJECT_DIR/mdlm/artifacts/logs"
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
sbatch --parsable --no-requeue --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --nodelist=gpu007 --ntasks=3 --cpus-per-task=4 --mem=60G --time=06:00:00 --job-name=mdlm-particle-scaling --chdir="$PROJECT_DIR" --export=ALL --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "$EXPERIMENT_DIR/scripts/slurm_avon_particle_scaling.sh"
```

Scientific command inside each allocated worker, with `worker` set to 0, 1 or 2:

```bash
python -m diffusion_toxicity_smc_experiment.same_run_reuse run-fresh --study "$EXPERIMENT_DIR/configs/mdlm_owt_particle_scaling.yaml" --output-root "$OUTPUT_ROOT" --calibration-root "$CALIBRATION_ROOT" --max-seconds 21000 --workers 3 --worker-id "$worker"
```
