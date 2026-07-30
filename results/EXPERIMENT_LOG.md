# Experiment Log

This is the cumulative index of research runs selected for review. Raw outputs,
model caches, and smoke runs are intentionally excluded from Git.

## Official direct-sampling reference

- Source: Zenodo `10.5281/zenodo.20395631`.
- Verified archive MD5: `83ad4e8f245edf5f45d879ae08450c3e`.
- Direct ARI observations: `4,803,396`, raw and uncapped in the archived file.
- Result: [`REFERENCE_DIRECT_SAMPLING.md`](REFERENCE_DIRECT_SAMPLING.md).
- Interpretation: a high-precision IID baseline for the paper's official ARI
  observable, but not strict ground truth for this repository's `textstat` ARI
  because completions and the custom edge-case implementation are unavailable.

For each future selected run record:

- commit, date, model, tokenizer, prompt, and seed;
- lambda schedule, particles, MCMC steps, suffix-size law, and token length;
- runtime, final SMC ESS/N, corrected importance ESS/N, mean/max ARI, and the
  reported tail thresholds;
- the conclusion supported by the run and the path under `selected_runs/`.

Only small `summary.json`, `report.md`, and sample previews belong in
`selected_runs/`. Keep complete `samples.jsonl` files outside version control.
