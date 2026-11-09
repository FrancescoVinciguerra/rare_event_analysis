# Camera-ready: low-budget score exceedance probabilities

Status: prepared and locally tested; not submitted to Avon. This is an eight-run
pilot per population size, not a final accuracy study. No previous results change.

## Frozen scientific design

- Same unconditional MDLM-OWT 32-step DDPM decoder, length 128, terminal
  categorical completion, classifier checkpoint and first-100-token evaluator.
- Events: classifier score >= 0.9 and >= 0.99, evaluated on the SAME population.
  These numbers are score thresholds, not true probabilities or terminal tilts.
- Same production RollBack resample-move kernel: three equally spaced internal
  tilt levels 1,2,3 starting at zero, FOUR MH sweeps per level, independent uniform
  rollback choices from the existing MDLM set {4,8,16,24,32}.
- ESS/N < 0.5, systematic resampling, weight -> resample -> mutate;
  no final resampling, Z_hat updated at every level. Mutations leave weights fixed.
- Score remains Phi=log(q); no logit transformation in this experiment.
- Single final population: both unnormalized SMC and reverse-SNIS; no combining
  estimates from multiple lambda populations or levels.
- N=16,32,64,128,256: five scientifically useful population sizes rather than
  forcing 1-2 particle populations to meet tiny waiting-time fractions.
  Eight independent pilot runs each; new recorded sampler seeds and
  separately derived crude seeds. Resume per saved component/run.
- Existing reference: 1516/200000=0.00758 (score >=0.9), 444/200000=0.00222
  (>=0.99). Wilson 95% intervals approximately [0.007209,0.007970] and
  [0.002023,0.002436]. These are finite Monte Carlo references, not exact truth.

## Budget convention

The primary budget counts ALL native transitions: 32 DDPM steps plus one
terminal categorical transition per complete initial path, and d+1 per proposed
suffix (including rejected proposals). The old DDPM-only count is also recorded
separately. Decoder forward-call counts and classifier calls are not confused
with particle transitions. No Gaussian padding is involved in this MDLM model.

Mean rollback depth is 16.8. With 3 levels * 4 moves = 12 proposals per particle:

    E[cost] = N * [33 + 12 * (16.8+1)] = 246.6*N

| N | Expected native transitions | Equivalent crude paths before floor |
|---|---:|---:|
|16|3945.6|119.56|
|32|7891.2|239.13|
|64|15782.4|478.25|
|128|31564.8|956.51|
|256|63129.6|1913.02|

Each run receives a fresh independent crude population of floor(actual_cost/33)
complete native paths, with both thresholds evaluated on it. Terminal transitions
are included for both methods. Remaining transitions are fewer than 33.

At exactly 100/200 crude paths, estimated probabilities of zero discoveries are
about 47%/22% for threshold 0.9 and 80%/64% for threshold 0.99. These are design
predictions using the finite reference, not new experimental results.

## Historical wall-clock estimate for crude MC's first event

The frozen `publication_figures_v2_20260823_final2/csv/direct_cost.csv` records
33041.4424 s generation and 1217.6117 s scoring for 200000 samples: 0.1712953 s
per sample, or 5.84 samples/s. This is a historical GPU throughput estimate,
not a newly timed sequential stopping experiment; it excludes queue time,
model loading and fresh-job startup. Batching and machine load can change
the short-run wall clock. Always measure the fresh pilot on the same device.

Using the finite reference p, mean IID waiting count is 1/p. A 95% discovery
budget is ceil(log(0.05)/log(1-p)). These imply:

| Score threshold | Mean samples to first hit | Approx mean time | Samples for 95% discovery | Approx time |
|---|---:|---:|---:|---:|
|0.9|131.93|22.6 s|394|67.5 s|
|0.99|450.45|77.2 s|1348|230.9 s|

Finding one event is not an accurate probability estimate. Across the five
configured populations, expected crude discoveries range from 0.91 to 14.50
for score >=0.9, and 0.27 to 4.25 for score >=0.99. These are expected budget
factors relative to the mean waiting time, not statistical power guarantees.

## Tests and local commands

    python -m pytest mdlm/tests/test_low_budget.py -q
    python -m diffusion_toxicity_smc_experiment.low_budget plan --study mdlm/configs/mdlm_owt_camera_ready_low_budget.yaml

The plan loads no models and submits nothing. Tests check both corrections against
an exactly known finite tilted target, zero-event estimates, normalized weights,
terminal-transition budget, disjoint streams and frozen distribution identifiers.

## Review before ANY remote write or job submission

The user guide `.codex-private/avon/WARWICK_SCRTP_HPC_GUIDE.md`, section 2,
requires showing the full commands and waiting for confirmation. Re-establish
the previously used tunnel to Avon (local 127.0.0.1:2222) before access. Do not
guess a new SSH route or disable host-key checking. The tunnel was not listening
at preparation time. All cluster values below were used historically and require
a fresh read-only preflight; they are not asserted to be currently valid.

ORIGIN: this Windows PC, repository
`C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main`.

DESTINATION: Avon, `/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code`.

FILES ADDED by the bundle: this document, the low-budget YAML, `low_budget.py`,
the two camera-ready shell scripts, `tests/test_low_budget.py`,
`plots/plot_camera_ready_low_budget.py`, and the portable GPU/report modules
`camera_ready_gpu.py` and `camera_ready_report.py`. No existing
scientific kernel, model, startup file, environment or result is replaced.

After the tunnel is open, read-only preflight from local PowerShell:

```powershell
ssh -i C:/Users/UTENTE/.ssh/id_ed25519_warwick -p 2222 -o HostKeyAlias=avon.scrtp.warwick.ac.uk -o StrictHostKeyChecking=yes -o BatchMode=yes stuhvh@127.0.0.1 'cd /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code && bash mdlm/scripts/avon_preflight.sh'
```

After explicit confirmation, transfer the reviewed new-file-only bundle:

```powershell
scp -i C:/Users/UTENTE/.ssh/id_ed25519_warwick -P 2222 -o HostKeyAlias=avon.scrtp.warwick.ac.uk -o StrictHostKeyChecking=yes mdlm/artifacts/camera_ready_low_budget_v1_code.tar.gz stuhvh@127.0.0.1:/gpfs/home/stats/stuhvh/camera_ready_low_budget_v1_code.tar.gz
```

Remote extraction, refuses to replace any existing file:

```bash
tar --keep-old-files -xzf /gpfs/home/stats/stuhvh/camera_ready_low_budget_v1_code.tar.gz -C /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
```

The prospective scheduler values below must first match the fresh preflight:

```bash
export AVON_ACCOUNT=stats
export AVON_GPU_PARTITION=gpu
export AVON_QOS=normal
export AVON_GPU_GRES=gpu:quadro_rtx_6000:1
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
cd "$PROJECT_DIR"
bash mdlm/scripts/submit_avon_camera_ready_low_budget.sh
```

This prints the exact sbatch and scientific commands without submitting. The job
request is ONE GPU, four CPU, 20 GB host RAM and at most two hours in `gpu`.
This limit is not a runtime forecast; the 40-population pilot is resumable.
No installation/download; cached models only; all Python runs inside SLURM.
After the dry-run and confirmation:

```bash
bash mdlm/scripts/submit_avon_camera_ready_low_budget.sh --submit
```

Expanded proposed job submission (assuming the values above are confirmed):

```bash
sbatch --parsable --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G --time=02:00:00 --job-name=mdlm-camera-low-budget --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/scripts/slurm_avon_camera_ready_low_budget.sh
```

Scientific command inside the allocated job:

```bash
python -m diffusion_toxicity_smc_experiment.low_budget run --study /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/configs/mdlm_owt_camera_ready_low_budget.yaml --output-root /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/current
```

Outputs: `artifacts/current/camera_ready_low_budget_v1/manifest.json`, original
per-component SMC JSON files, `runs/N*_seed*.json`, `runs.csv`, `summary.csv`.
Per-run files retain terminal weights, scores, Z_hat, parent links, ESS,
acceptance, genealogy, seeds and fresh crude scores. Analysis uses reference
ratios and log10(1+abs(estimate/reference-1)), includes all zero estimates, and
must disclose reference uncertainty. Eight pilot runs must not be presented
as a conclusive performance comparison. Choose the final replication count
only after inspecting runtime and stability; do not select it by observed wins.


## Required presentation: same boxplots as the toy experiment

Produce exactly two summary PNG figures at 300 dpi from independent per-run
estimates: (1) log10(1+abs(estimate/reference-1)); (2) estimate/reference.
Each figure has one row per score threshold and comparison policy, with zoom
on the left and the complete distribution on the right. Five grouped x-axis
positions correspond to N=16,32,64,128,256. Blue = unnormalized SMC, orange =
SMC+SNIS, gray = crude MC. Box = Q25-Q75, whiskers 1.5 IQR, black central line
= median, black diamond = arithmetic mean of ALL independent runs. Both are
shown and explicitly distinguished; the mean can lie outside the box.

- All zero estimates remain zero in all calculations and both views.
- All outliers are visible in the complete panel; zoom exclusions are counted
  for each method and each N. Zero counts and total run counts are also shown.
- Log-error zoom: 0-0.4, with ticks labeled by metric AND relative error
  (0%,25%,50%,75%,100%,150%); dashed reference at log10(2), including zero estimates.
- Ratio zoom: 0-2, line at 1, green band 0.5-1.5. Full ratio uses a mixed
  axis, linear 0-1 and logarithmic above 1, with no epsilon substitution.
- The finite 200k IID reference is labeled as a Monte Carlo estimate; show its
  Wilson interval in titles and as a reference band in ratio plots.
- Same vertical scales across events for each view. No averaging across levels,
  events, population sizes or matching policies.

Offline command, on the local analysis machine after downloading real runs.csv:

```bash
python mdlm/plots/plot_camera_ready_low_budget.py --runs PATH_TO_REAL_RESULTS/runs.csv
```

The figures `camera_ready_log_error_boxplots.png` and
`camera_ready_ratio_boxplots.png` go in `plots/`. The plot-audit JSON next to
the input CSV contains the exact
counts. Rendering is tested on clearly labeled synthetic layout fixtures in
temporary test directories only; they are NOT experimental results.

The runner now produces BOTH matching policies with independent baseline streams.
A fresh full-kernel warm-up and three independent 128-path crude timing batches
are excluded from estimates and per-run timing. The median calibrated seconds
per crude completion determines the wall-clock baseline sample count BEFORE its
samples are generated. This avoids stopping on the baseline's own data-dependent
processing time. Matching is approximate: actual time ratios and overruns are
saved and explicitly reported. Each SMC estimate includes full generation,
classifier scores, weighting, resampling, MH and both final corrections, with GPU
synchronization before and after measurement. Startup, warm-up and file I/O are
excluded and separately documented. Both policies retain the same SMC population;
they are not independent additional SMC replications.

Every completed population and each baseline is checkpointed. The job creates
runs.csv and summary.csv progressively and REPORT.html/REPORT.md automatically
at normal completion or on a clean termination with partial data. REPORT.html
contains the two full SVG figures inline, so it is standalone and requires no
browser connection, JavaScript, matplotlib or external assets on Avon. The two
300-dpi PNGs are generated locally by the tested plotting script after downloading
the real CSV. A results tar.gz is written next to the experiment output directory.
The two-hour job limit is NOT a promise that all 40 pilot populations will finish;
partial reports label the completed population count and the same configuration
can resume. Final performance claims require a larger independently seeded study.
