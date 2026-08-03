# Data reuse and ARI convention

## Active validation convention

The active direct and suffix-SMC validation uses `official_public` ARI:

```text
exact prompt
-> tokenize prompt
-> generate exactly 100 completion tokens
-> decode prompt tokens plus completion tokens together
-> remove the first len(prompt) characters from the decoded string
-> reconstruct exact prompt plus decoded completion
-> textstat.automated_readability_index
```

Direct and SMC call the same scoring function. A direct cache is accepted only
when its metadata matches the model, tokenizer, exact prompt, completion length,
temperature, fixed-length policy, and ARI convention requested by the SMC run.

## Why the released paper arrays are not a baseline

The released paper archive contains scalar ARI and log-probability arrays but no
completion texts or token IDs. The paper also reports a custom ARI
implementation whose edge-case handling is not public. Consequently, the
archived values cannot be rescored with this repository's implementation.

A 200,000-sample IID calibration generated from the public code convention did
not agree with the archived tail probabilities within Monte Carlo uncertainty.
The difference therefore cannot be attributed to SMC. The archived arrays are
not used in validation tables, plots, estimators, or automated conclusions.
They remain part of the cited paper's historical record only.

## Reusable artifacts

| Artifact | Reusable for probability estimation | Rescore ARI | Runtime comparison |
| --- | --- | --- | --- |
| Matching IID direct `direct_ari.npz` | Yes | No | Yes when runtime metadata is present |
| Matching compact SMC component | Yes, with its stored weights and log Z | No | Yes when runtime metadata is present |
| Full `samples.jsonl` with token IDs | Yes for its original target | Yes, diagnostically | Metadata-dependent |
| Released paper scalar arrays | No in the active validation | No | No |

Changing the ARI convention changes MH acceptance decisions, incremental SMC
weights, resampling decisions, and normalizing-constant estimates. Old SMC runs
therefore cannot be retargeted by replacing only their final ARI values.

## Colab workflow

`notebooks/REA_Suffix_SMC_Quickstart.ipynb` has three phases:

1. Generate or load one reproducible IID direct reference.
2. Generate or load the requested independent SMC components.
3. Compare the component-averaged SMC estimate with the matching IID direct
   reference in memory.

Only direct arrays, direct metadata, and compact SMC component archives persist
on Drive. Tables and plots stay in the notebook.
