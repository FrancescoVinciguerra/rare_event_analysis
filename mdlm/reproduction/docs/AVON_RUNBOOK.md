# Avon runbook for MDLM

Use this guide for access, environment setup, staged jobs and recovery. The [reproduction guide](../README.md) owns the decoder, event, calibration and estimator definitions. For specific campaigns see the [budget pilots](EXPLORATION_V2.md), [accuracy-budget screen](ACCURACY_VS_BUDGET.md), [continuing ladders](SAME_RUN_LOGIT.md), and [particle scaling](PARTICLE_SCALING.md).

On the login node, do read-only preflight, scheduler queries, file transfer and authorized submission. Run Python, model loading, dependency installation and downloads on allocated compute nodes. Use the authorized SSH route with host-key verification. Keep credentials out of scripts and Git.

## 1. Preflight and persistent paths

From the repository root on Avon:

```bash
bash mdlm/reproduction/scripts/avon_preflight.sh
```

Verify hostname, project mount and quota, account association, QoS, partition state, GPU GRES, wall time and the Python module prerequisite chain. Inspect CUDA/PyTorch metadata without loading modules on the login node. If a scheduler query fails, diagnose that failure before submitting. Use preflight-confirmed values for all requests.

Set the paths for the checkout and allocated jobs:

```bash
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export EXPERIMENT_DIR="$PROJECT_DIR/mdlm"
export OUTPUT_ROOT="$EXPERIMENT_DIR/artifacts/current"
export LOG_DIR="$EXPERIMENT_DIR/artifacts/logs"
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
```

For a frozen campaign, point `EXPERIMENT_DIR` to its verified source bundle and set `OUTPUT_ROOT` separately. Preserve the bundle's hashes and resolved configuration. For a syntax check use `bash -n` on the relevant launcher and worker. This checks Bash syntax without executing the files.

Launchers require `AVON_ACCOUNT`, `AVON_QOS`, `AVON_GPU_GRES` and `AVON_MODULES` where applicable. Smoke and early staged launchers use `AVON_PARTITION`. Comparison and submission launchers also use `AVON_GPU_PARTITION`, `AVON_GPU_DEVEL_PARTITION` or `AVON_CPU_PARTITION` by stage. Historically observed values were account `stats`, QoS `normal`, partitions `gpu-devel`, `gpu` and `devel`, GRES `gpu:quadro_rtx_6000:1`, and modules `GCCcore/13.2.0 Python/3.11.5`. Confirm them afresh.

## 2. Environment setup on a compute node

After review of the request, allocate an interactive GPU with the verified values. The documented setup used one GPU, two CPUs, 8 GB and 30 minutes:

```bash
srun --account="$AVON_ACCOUNT" --partition="$AVON_PARTITION" \
  --qos="$AVON_QOS" --gres="$AVON_GPU_GRES" --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash -l
```

Confirm that the shell is on a compute node and run:

```bash
bash "$EXPERIMENT_DIR/reproduction/scripts/avon_setup_environment.sh"
```

The [setup script](../scripts/avon_setup_environment.sh) and [Avon requirements](../../src/diffusion_toxicity_smc_experiment/requirements-avon.txt) own the installation recipe, including the PyTorch 2.6.0 cu118 and Transformers 4.57.6 combination used to load the classifier. The tested smoke environment used FP32 for MDLM and FP16 for the classifier, without FlashAttention, bfloat16, bitsandbytes or native CUDA compilation. If outbound access is unavailable on the compute node, stop and resolve access there.

## 3. Preview and submit one stage

Review the exact account, partition, resources, time limit, config, paths and commands printed by the dry-run. Obtain authorization for the specific submission. A scheduler limit is an allocation bound. Use observed timings for forecasts.

| Workflow | Launcher and stage rules |
|---|---|
| Smoke | [Smoke launcher](../scripts/submit_avon_smoke_pipeline.sh). Use `technical --dry-run` or `scientific --dry-run` to preview the pipeline. The scientific profile needs a partition that supports the printed four-hour direct and twelve-hour SMC requests. |
| First 32-step experiment | [First-experiment launcher](../scripts/submit_avon_first_experiment.sh). Start with `direct-256`, then score and review before calibration, reporting or extending the same direct run to 1,000. The helper has no 10,000-sample stage. The [reproduction guide](../README.md#terminal-lambda-calibration) gives the scientific sequence. |
| Three-sampler comparison | [Serious-comparison launcher](../scripts/submit_avon_serious_comparison.sh). Require `method-validation` to complete with exit `0:0` before the direct reference, scoring, SMC grid or CPU report stages. |
| Submission probability | [Submission launcher](../scripts/submit_avon_submission_probability.sh). Run `pilot`, `long-smc` and `report` separately. Verify the pilot's source IDs, seed separation, finite estimates and IID discovery counts before authorizing the long stage. |

The non-smoke launchers in the table default to preview and accept `--submit` for one selected stage. The smoke launcher uses a different interface. Preview the selected stage with `--stage STAGE --dry-run`. When authorized, use `--stage STAGE` without `--dry-run`. The former multi-job `--submit` mode is disabled for this launcher. Smoke order is `prefetch → direct → score → smc → compare`. For a dependent stage, give the preceding job ID with `--afterok` where the submission limit permits it, or use `--no-dependency` after verifying completion. Prefetch has no predecessor.

For example, this command only previews the technical prefetch stage:

```bash
bash mdlm/reproduction/scripts/submit_avon_smoke_pipeline.sh technical --stage prefetch --dry-run
```

The submission study's documented allocations were two CPU cores and 4 GB for the pilot (20 minutes) and report (30 minutes), and one GPU, four CPUs and 20 GB for the long stage (two days). The planning estimate for the long runs was about 23 GPU-hours in total. Verify the current launcher and cluster values before use.

## 4. Verify, monitor and resume

Record the scheduler job ID printed by the launcher. Inspect scheduler status, both log files, memory and actual persisted artifacts:

```bash
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,Account,State,Elapsed,Timelimit,ReqMem,AllocTRES,ExitCode
```

Check model and classifier loading, finite outputs, peak CUDA allocation, and matching decoder/evaluator IDs before comparing results. Smoke should exercise generation, scoring, aggregation and duplicate-free resume. The two smoke profiles use different decoder IDs. The 32-step and 1,000-step outputs represent different native distributions, and the unconditional runs also differ from prompt-conditioned FK Steering. The original smoke verification recorded these expected fingerprints for its configuration:

```text
technical decoder_id:  cb78b724cdaa4a7ec1c91955be80de5e169f77ab776a98ec3ae1dc6aa0cfb609
scientific decoder_id: 98719063a58345c23c4726c2e02cb838a447890669c05fef32cfcaa7fecf85fd
evaluator_id:          ae3f157a1521ef7f986ba734d14c0277c96a4754fd6b74693c6d92ebc7b9a60c
```

For later runs use the IDs in their resolved manifests.

Keep logs and source fingerprints with the run provenance. Check completed populations and report completeness separately from the batch status. A plotting failure can leave valid scientific outputs, as documented in the [logit recovery](SAME_RUN_LOGIT.md#potential-change-and-validation).

After diagnosing an interruption, resubmit only the interrupted stage using compatible config, code, paths and model IDs. Direct generation scans indices and fills missing samples. The smoke direct job also checks that a second resume produces no new samples. Scoring skips complete shards, SMC skips complete component or population files, and reporting reuses completed inputs. Check the manifests, independent seeds and matching policy before combining workers or reusing saved data. Inspect partial results as partial.
