# Private Avon deployment note — MDLM rare-event benchmark

This file is excluded through `.git/info/exclude` and must never be committed.
It supersedes the earlier fixed-count demasking deployment plan.

No remote directory, transfer, allocation, installation, model download or
experiment is authorized by this note.

Before any remote action, rebuild the offline bundle manifest for the current
benchmark. It must contain the pinned artifacts in
`configs/mdlm_owt_unconditional_common.yaml`, including:

- byte-identical MDLM-OWT weights plus the pinned no-FlashAttention wrapper;
- pinned GPT-2 tokenizer;
- pinned `roberta-large` completion-trimming tokenizer;
- pinned Skolkovo toxicity classifier;
- Linux Python 3.11.5 / PyTorch 2.6.0+cu118 wheelhouse.

Persistent paths remain proposed, not created:

```text
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/outputs
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/logs
/gpfs/home/stats/stuhvh/hf-cache/mdlm-rare-event
/gpfs/home/stats/stuhvh/wheelhouse/mdlm-rare-event
/gpfs/home/stats/stuhvh/venvs/mdlm-rare-event
```

After local bundle sizes and hashes are known, show the exact transfer command
and obtain approval. After read-only verification on Avon, show a separate
`gpu-devel` installation allocation. Only after reviewing its log, show a
separate one-GPU smoke allocation using
`configs/mdlm_owt_unconditional_smoke_technical.yaml`.

Production commands live in `scripts/slurm_*.sh`. They do not hardcode a home
directory. Do not submit them before filling and verifying `PROJECT_DIR`,
`VENV_DIR`, `HF_HOME`, output/log paths, account, partition, GPU, CPU, RAM and
wall time with the user.
