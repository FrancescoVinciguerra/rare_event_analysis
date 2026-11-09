# Avon runbook: unconditional MDLM toxicity smoke benchmark

This runbook prepares two non-production pipelines for the unconditional
MDLM-OWT distribution. It does not reproduce FK Steering's prompt-conditioned
experiment. Do not compare the resulting toxicity probability with FK's
reported prompt-conditioned rate.

No experiment, installation, Python process, model load, or download may run on
an Avon login node. Only the read-only preflight and `sbatch` submission belong
there. No script modifies `.bashrc` or contains a Hugging Face token.

## Current 32-step experiment

The current scientific target is the 32-step decoder probability
`P_{X ~ P0^(32)}(E(X)=1)`, not the standard 1000-step MDLM probability. Direct
and SMC use the same unconditional `[MASK]^128` prior, native stochastic
DDPM/SUBS transitions, stochastic terminal completion, decoder ID, and
evaluator ID.

The first sequence is deliberately staged:

1. generate or resume 256 direct samples;
2. score and inspect them;
3. run the non-aggregating lambda calibration at 256 particles;
4. review runtime, completions, event rate, ESS and genealogy;
5. extend the same direct run to 1000 samples;
6. choose the terminal lambdas and only then finalize the estimation
   configuration.

The helper below defaults to dry-run and submits at most one job per invocation:

```bash
export AVON_ACCOUNT=stats
export AVON_PARTITION=gpu-devel
export AVON_QOS=normal
export AVON_GPU_GRES=gpu:quadro_rtx_6000:1
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'

bash mdlm/scripts/submit_avon_first_experiment.sh direct-256
bash mdlm/scripts/submit_avon_first_experiment.sh score-direct
bash mdlm/scripts/submit_avon_first_experiment.sh lambda-calibration
bash mdlm/scripts/submit_avon_first_experiment.sh calibration-report
bash mdlm/scripts/submit_avon_first_experiment.sh direct-1000
bash mdlm/scripts/submit_avon_first_experiment.sh estimation-100tok
bash mdlm/scripts/submit_avon_first_experiment.sh estimation-100tok-trajectory-mh
bash mdlm/scripts/submit_avon_first_experiment.sh estimation-100tok-trajectory-mh-matched-fk
```

Add `--submit` to exactly one reviewed command. The helper intentionally has no
10,000-sample stage. The dedicated `estimation-100tok` and
`estimation-100tok-trajectory-mh` and
`estimation-100tok-trajectory-mh-matched-fk` stages are submit-capable only
after explicit review. The latter two change only the tilted-sampler backend; see
`docs/TRAJECTORY_RESAMPLE_MOVE.md`. Do not use
`direct-1000` until the 256-sample runtime and artifacts have been reviewed.

## Persistent paths

```bash
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export EXPERIMENT_DIR="${PROJECT_DIR}/mdlm"
export OUTPUT_ROOT="${EXPERIMENT_DIR}/artifacts/current"
export LOG_DIR="${EXPERIMENT_DIR}/artifacts/logs"
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
```

## 1. Read-only preflight after login

```bash
cd /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
bash mdlm/scripts/avon_preflight.sh
```

Verify:

- Avon login hostname and `/gpfs/home/stats/stuhvh` mount/quota;
- account association and allowed QoS;
- partition state, GPU GRES, and maximum wall time;
- the Python 3.11 prerequisite chain from `module spider`;
- CUDA/PyTorch module metadata without loading modules.

Previously observed candidates are account `stats`, QoS `normal`, partitions
`gpu-devel` and `gpu`, GRES `gpu:quadro_rtx_6000:1`, and modules
`GCCcore/13.2.0 Python/3.11.5`. Use them only after the current preflight
confirms them. If a SLURM query fails, stop and diagnose that exact error.

## 2. Export verified scheduler values

Replace values below only with preflight-confirmed values:

```bash
export AVON_ACCOUNT=stats
export AVON_PARTITION=gpu-devel
export AVON_QOS=normal
export AVON_GPU_GRES=gpu:quadro_rtx_6000:1
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
```

Account, partition, QoS, and module choices are not embedded in the new SLURM
jobs. The submit script refuses to continue without the mandatory exports.

## 3. Validate Bash syntax on the login node

This is read-only and executes none of the scripts:

```bash
cd /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
bash -n \
  mdlm/scripts/avon_job_common.sh \
  mdlm/scripts/avon_preflight.sh \
  mdlm/scripts/avon_setup_environment.sh \
  mdlm/scripts/slurm_avon_prefetch_models.sh \
  mdlm/scripts/slurm_avon_smoke_direct.sh \
  mdlm/scripts/slurm_avon_smoke_score.sh \
  mdlm/scripts/slurm_avon_smoke_smc.sh \
  mdlm/scripts/slurm_avon_smoke_compare.sh \
  mdlm/scripts/submit_avon_smoke_pipeline.sh
```

## 4. Create the environment on an allocated compute node

After reviewing the exact request, allocate one GPU with verified values:

```bash
srun \
  --account="${AVON_ACCOUNT}" \
  --partition="${AVON_PARTITION}" \
  --qos="${AVON_QOS}" \
  --gres="${AVON_GPU_GRES}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --pty bash -l
```

Only after the shell is on a compute node:

```bash
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export EXPERIMENT_DIR="${PROJECT_DIR}/mdlm"
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
bash "${EXPERIMENT_DIR}/scripts/avon_setup_environment.sh"
exit
```

The environment pins PyTorch 2.6.0 cu118, required for safe loading of the
classifier checkpoint with Transformers 4.57.6. MDLM uses FP32 and the
classifier uses FP16. FlashAttention, `bfloat16`, bitsandbytes, and native CUDA
compilation are not enabled. If compute nodes lack outbound access, stop; do
not move setup or downloads to the login node.

## 5. Review without submitting

In the login shell, export the persistent paths and verified scheduler values,
then run:

```bash
cd "${PROJECT_DIR}"
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --dry-run
```

For the scientific smoke, first select a verified partition supporting the
printed four-hour direct and twelve-hour SMC requests:

```bash
export AVON_PARTITION=gpu
bash mdlm/scripts/submit_avon_smoke_pipeline.sh scientific --dry-run
```

The full dry run prints account, partition, GRES, GPU/CPU/RAM/time, config,
exact commands, outputs, and logs for every stage. A stage-specific dry run is
also available and submits nothing:

```bash
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage prefetch --dry-run
```

## 6. Submit one stage at a time after dry-run verification

The old multi-job `--submit` mode is intentionally disabled. Each invocation
below performs at most one `sbatch` call and prints one job ID. On a cluster
with a one-job submission limit, wait until each job is `COMPLETED`, verify its
log, and then explicitly submit the next stage without a dependency:

```bash
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage prefetch
# Verify PREFETCH_ID is COMPLETED.
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage direct --no-dependency
# Verify DIRECT_ID is COMPLETED.
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage score --no-dependency
# Verify SCORE_ID is COMPLETED.
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage smc --no-dependency
# Verify SMC_ID is COMPLETED.
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage compare --no-dependency
```

Where the submission limit permits one pending dependent job, preserve SLURM
ordering explicitly by supplying the immediately preceding job ID:

```bash
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage direct --afterok PREFETCH_ID
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage score --afterok DIRECT_ID
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage smc --afterok SCORE_ID
bash mdlm/scripts/submit_avon_smoke_pipeline.sh technical --stage compare --afterok SMC_ID
```

For the scientific profile, first set the verified longer partition and use
the same stage sequence with `scientific`:

```bash
export AVON_PARTITION=gpu
bash mdlm/scripts/submit_avon_smoke_pipeline.sh scientific --stage prefetch
```

The logical order remains:

```text
prefetch -> direct -> score -> SMC -> compare
```

The technical config is
`mdlm/configs/mdlm_owt_unconditional_smoke_technical.yaml`: 128 positions, 32
denoising steps, 2 IID direct samples, 8 particles, 2 independent runs,
tilted lambdas `[5,10]`, uniform weights `[0.5,0.5]`, and diagnostic lambda
`[0]`.

The scientific config is
`mdlm/configs/mdlm_owt_unconditional_smoke_scientific.yaml`: 128 positions, 1,000
steps, 2 IID direct samples, 8 particles, 1 run, the same two tilted lambdas,
and the separate zero diagnostic. Values 5 and 10 exercise the pipeline and
are not asserted to be optimal.

Both start from exactly `[MASK]^128`, with no BOS/EOS, prompt, or fixed prefix.
The upstream MDLM diffusion sampler was verified to use this all-mask prior;
BOS insertion exists only in its autoregressive sampler.

DDPM transitions are stochastic. If masks remain after the last transition,
this benchmark removes them with a categorical model draw rather than argmax.
For FK intermediate rewards, each update draws four categorical `x_0`
predictions from the current MDLM distribution, as in the original FK Steering
MDLM code. Reward checkpoints score all four predictions and aggregate their
log-probability rewards with `logmeanexp`; these temporary predictions are
discarded and never replace the propagated particle.

Expected identifiers will change whenever the decoder manifest changes:

```text
technical decoder_id:  cb78b724cdaa4a7ec1c91955be80de5e169f77ab776a98ec3ae1dc6aa0cfb609
scientific decoder_id: 98719063a58345c23c4726c2e02cb838a447890669c05fef32cfcaa7fecf85fd
evaluator_id:          ae3f157a1521ef7f986ba734d14c0277c96a4754fd6b74693c6d92ebc7b9a60c
```

Direct and SMC comparison rejects any different `decoder_id` or `evaluator_id`.
Old prompt-conditioned artifact directories remain untouched and are never
selected by these new identifiers.

## 7. Monitor logs, memory, and artifacts

Record the one ID printed by each stage invocation:

```bash
squeue -j PREFETCH_ID,DIRECT_ID,SCORE_ID,SMC_ID,COMPARE_ID
sacct -j PREFETCH_ID,DIRECT_ID,SCORE_ID,SMC_ID,COMPARE_ID \
  --format=JobID,JobName,Partition,Account,State,Elapsed,Timelimit,ReqMem,AllocTRES,ExitCode
tail -n 100 mdlm/artifacts/logs/mdlm-smoke-smc-SMC_ID.out
tail -n 100 mdlm/artifacts/logs/mdlm-smoke-smc-SMC_ID.err
grep -R --line-number 'peak_cuda_memory_allocated_bytes' \
  mdlm/artifacts/logs \
  mdlm/artifacts/current/mdlm_owt
find mdlm/artifacts/current/mdlm_owt \
  -maxdepth 7 -type f -printf '%p %s bytes\n' | sort
```

Generation rows store `conditioning_mode: unconditional`, all 128 terminal
tokens, and decoded text. Scoring artifacts contain classifier inputs/logits,
rewards, and events. SMC stores tilted and diagnostic components separately,
including reverse weights, ESS, genealogy, NFE, runtime, and completions.

Successful artifacts verify CUDA/model/classifier loading, all-mask initial
state, empty fixed prefix, no terminal mask, finite values, separate
reverse-SNIS ratios for lambdas 5 and 10, arithmetic-mean aggregation, excluded
zero diagnostic, IID binomial/Wilson direct baseline, matching identifiers,
and duplicate-free resume.

## 8. Resume

After diagnosing an interruption, resubmit only the interrupted stage. Use
`--afterok PREVIOUS_ID` when the prerequisite is still active, or
`--no-dependency` only after verifying it already completed. Direct scans global
sample indices and creates only missing samples up to `target_samples=2`; the
smoke direct job performs a second no-op resume scan. Scoring skips complete
shards, SMC skips complete run/lambda component JSON files, and completed
comparisons are reused. Existing prompt-conditioned or unconditional outputs
are never deleted.

## 9. Scaling after both smoke profiles pass

Increase one dimension at a time:

1. independent SMC runs;
2. particles, while monitoring peak VRAM;
3. the direct `target_samples` pilot size;
4. generation microbatch size only after memory measurements.

The historical `scientific` smoke profile remains a separate 1,000-step decoder
check with its own `decoder_id`; do not mix its artifacts with the reviewed
32-step comparison below. Neither profile changes the stochastic DDPM kernel,
terminal categorical completion, reward/event definition, four-sample FK
multinomial intermediate prediction, or uniform tilted-component weights.

## Serious 32-step three-method comparison

The reviewed long comparison is configured separately in
`configs/mdlm_owt_unconditional_serious_comparison.yaml`. It keeps the validated
32-step decoder and compares `fk_bootstrap`, `trajectory_resample_move`, and
`trajectory_waste_free` at `N={64,128,256,512}`, with eight independent runs
per grid point. Each run retains four independently seeded lambda populations
and the fixed uniform convex combination.

The direct reference contains 200,000 IID completions. It is generated using the
direct 32-step configuration and scored using the exact 100-token evaluator of
the SMC configs. The decoder and evaluator IDs must match before reporting.
The already validated 10,000 samples are reused; only missing indices are
generated. New samples use larger append-only shards, while the report counts
the actually persisted shard seeds instead of inferring them from one shard
size.

After exporting values verified by preflight:

```bash
export AVON_ACCOUNT=stats
export AVON_QOS=normal
export AVON_GPU_PARTITION=gpu
export AVON_GPU_DEVEL_PARTITION=gpu-devel
export AVON_CPU_PARTITION=devel
export AVON_GPU_GRES=gpu:quadro_rtx_6000:1
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
```

inspect each stage before adding `--submit`:

```bash
bash mdlm/scripts/submit_avon_serious_comparison.sh method-validation
bash mdlm/scripts/submit_avon_serious_comparison.sh direct-reference
bash mdlm/scripts/submit_avon_serious_comparison.sh score-reference
bash mdlm/scripts/submit_avon_serious_comparison.sh smc-grid
bash mdlm/scripts/submit_avon_serious_comparison.sh report
```

First require `method-validation` to report `COMPLETED` and `0:0`; it exercises
all three real kernels at N=16 and N=64 with two independent runs. Submit only
the next requested stage after checking its predecessor. No stage submits
another stage. Resubmitting direct generation creates only missing sample
indices; resubmitting the SMC grid reuses completed component JSON files. The
CPU-only report reads existing artifacts and produces CSV, JSON and
dependency-free SVG plots below the suite-specific comparison directory in
`artifacts/current/mdlm_owt/comparisons/`.

## Submission rare-probability study

This follow-up deliberately excludes Waste-Free and submits no dependency
chain. It first reuses the completed serious-comparison artifacts as a frozen
pilot, then runs independent test populations, and finally reports uniform
versus pilot-fitted component aggregation.

Verified resources used by the submitter are:

```text
pilot:    devel, no GPU, 2 CPU, 4 GB, 00:20:00
long-smc: gpu, gpu:quadro_rtx_6000:1, 4 CPU, 20 GB, 2-00:00:00
report:   devel, no GPU, 2 CPU, 4 GB, 00:30:00
```

The two-day GPU value is a SLURM limit. Based on the previous N=512 timings,
32 FK and 32 trajectory-MH runs are expected to require about 23 GPU-hours in
total. Each stage invocation performs at most one `sbatch` call.

After exporting the same verified Avon variables shown above, print the exact
first command without submitting it:

```bash
bash mdlm/scripts/submit_avon_submission_probability.sh pilot
```

Confirm the dry-run reports account `stats`, QoS `normal`, partition `devel`,
no GRES, the project directory under `/gpfs/home/stats/stuhvh/`, and the
submission-study artifact/log paths. Then submit only the pilot:

```bash
bash mdlm/scripts/submit_avon_submission_probability.sh pilot --submit
```

After it is `COMPLETED 0:0`, inspect the continuous-score tail counts and
frozen weights:

```bash
sacct -j PILOT_ID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End
tail -n 100 mdlm/artifacts/logs/mdlm-submission-pilot-PILOT_ID.out
tail -n 100 mdlm/artifacts/logs/mdlm-submission-pilot-PILOT_ID.err
find mdlm/artifacts/current/mdlm_owt/submission_probability \
  -name direct_thresholds.csv -o -name pilot_weights.csv -o -name pilot_weights.json
```

Do not submit the long stage until the pilot confirms the old artifact IDs,
zero pilot/test seed overlap, finite component estimates, and interpretable
IID discovery counts at the retained thresholds. The dry-run and submission
commands are then:

```bash
bash mdlm/scripts/submit_avon_submission_probability.sh long-smc
bash mdlm/scripts/submit_avon_submission_probability.sh long-smc --submit
```

The job verifies the frozen pilot before loading either model. It resumes at
the component-file level and never replaces completed artifacts. Once it has
completed successfully, run the CPU report in the same two-step fashion:

```bash
bash mdlm/scripts/submit_avon_submission_probability.sh report
bash mdlm/scripts/submit_avon_submission_probability.sh report --submit
```

Final tables and plots are written below the study-specific directory in
`artifacts/current/mdlm_owt/submission_probability/`. The report keeps the 8
pilot runs out of all final 32-run metrics. It reports paired squared-error
improvement, variance, RMSE against the finite IID reference, runtime, decoder
particle-steps, classifier texts, reverse ESS, root-ancestor diversity, and
peak GPU memory. The IID Wilson interval remains visible because 200,000
direct draws are a high-precision Monte Carlo reference, not exact truth.
