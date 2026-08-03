# Experiment Log

This is the cumulative index of research runs selected for review. Raw outputs,
model caches, and smoke runs are intentionally excluded from Git.

ARI runs included here must use the repository's reproducible convention.
Released paper arrays are excluded from numerical comparisons because their
completion texts and exact ARI implementation are unavailable. Safety runs may
also be selected when they establish a clear sampler or measurement-validity
result; they must remain separate from ARI conclusions.

For each future selected run record:

- commit, date, model, tokenizer, prompt, and seed;
- lambda schedule, particles, MCMC steps, suffix-size law, and token length;
- runtime, final SMC ESS/N, corrected importance ESS/N, mean/max ARI, and the
  reported tail thresholds;
- the conclusion supported by the run and the path under `selected_runs/`.

Only small `summary.json`, `report.md`, and sample previews belong in
`selected_runs/`. Keep complete `samples.jsonl` files outside version control.

## Selected results

- `qwen_guard_yes_prefix_seed_7319`: SMC shifted QwenGuard's terminal
  `P(Unsafe)`, but paired-prefix diagnostics and manual inspection show that
  this score is not a validated proxy for actionable harmful assistance in the
  tested setup. The `Yes ` prefix is a major confounder.
