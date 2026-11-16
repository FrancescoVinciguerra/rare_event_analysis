# Avon runbook for MDLM

Use this guide for access, environment setup, staged jobs and recovery. See the [reproduction guide](../README.md) for the estimator and configurations and the [results index](../../../shared/RESULTS_INDEX.md) for saved campaigns.

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

## 3. Preview and run the paper campaign
Review the account, partition, resources, time limit, configuration and paths
before submitting. For the paper campaign, preview the run grid from the
allocated environment and run the same CLI command as on a local machine.

```bash
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode plan
python mdlm/reproduce.py --config mdlm/configs/paper.yaml --mode run
```

Use `--mode smoke` for the synthetic CPU check or the
[pinned-model GPU test](../README.md#validate-the-pinned-models-on-a-gpu)
on an allocated GPU. The [historical notes](HISTORICAL_CAMPAIGNS.md)
describe the earlier staged launchers and where to recover them.

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

Keep logs and source fingerprints with the run provenance. Check completed populations and report completeness separately from the batch status. A plotting failure can leave valid scientific outputs, as documented in the [logit completion record](../../artifacts/current/camera_ready_same_run_logit/COMPLETION_STATUS.json).

After diagnosing an interruption, resubmit only the interrupted stage using compatible config, code, paths and model IDs. Direct generation scans indices and fills missing samples. The smoke direct job also checks that a second resume produces no new samples. Scoring skips complete shards, SMC skips complete component or population files, and reporting reuses completed inputs. Check the manifests, independent seeds and matching policy before combining workers or reusing saved data. Inspect partial results as partial.
