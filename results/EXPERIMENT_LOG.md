# Experiment Log

This is the cumulative index of research runs selected for review. Raw outputs,
model caches, and smoke runs are intentionally excluded from Git.

Only runs generated with the repository's reproducible ARI convention belong
in this log. Released paper arrays are excluded from numerical comparisons
because their completion texts and exact ARI implementation are unavailable.

For each future selected run record:

- commit, date, model, tokenizer, prompt, and seed;
- lambda schedule, particles, MCMC steps, suffix-size law, and token length;
- runtime, final SMC ESS/N, corrected importance ESS/N, mean/max ARI, and the
  reported tail thresholds;
- the conclusion supported by the run and the path under `selected_runs/`.

Only small `summary.json`, `report.md`, and sample previews belong in
`selected_runs/`. Keep complete `samples.jsonl` files outside version control.
