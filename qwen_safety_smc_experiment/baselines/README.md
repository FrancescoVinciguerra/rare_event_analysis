# Direct safety baselines

`historical_direct_baselines.json` records the aggregate direct-sampling
results recovered from the executed Colab notebooks downloaded on 2026-08-03.
Runs are deduplicated by their complete sampling configuration and seed. In
particular, notebook copies `(3)`, `(4)`, and `(5)` contain the same run and do
not count as three independent baselines.

The old notebooks did not serialize all completion texts and token IDs. Their
aggregate results can therefore document prior evidence, but cannot populate
the sample-level analysis used by the active notebook. This limitation is
marked by `raw_samples_available: false`.

Going forward, the notebook stores sample-level direct runs in a private,
gzip-compressed cache. The cache filename contains a fingerprint of the model,
tokenizer, rendered context, fixed assistant prefix, sampling convention,
completion length, and observable. SMC-only settings such as lambda schedules,
particle counts, and mutation steps are deliberately excluded. Distinct seeds
can be appended to one compatible cache; a repeated seed is not added twice.

The cache can contain unsafe model outputs and is ignored by Git. Keep it in a
private Drive directory rather than committing it to the public repository.
