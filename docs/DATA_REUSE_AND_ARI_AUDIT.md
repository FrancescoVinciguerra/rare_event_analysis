# Data reuse and ARI audit

## Scope

This audit was completed before any new SMC, TPS, or direct production run.
It covers the pinned official repository, the downloaded paper archives, the
active suffix SMC formats, and the Drive layout expected by the Colab
quickstart. No data were deleted or rewritten.

The official source inspected here is submodule commit
`9a2ae06323db98496d179cefb294db60a6cd9535`. The relevant public files are:

- `external/rea_of_llms_official/sampling_examples.ipynb` for generation, ARI,
  and TPS;
- `external/rea_of_llms_official/reweighting_example.ipynb` for loading,
  burn-in, Gelman-Rubin filtering, and MBAR;
- `external/rea_of_llms_official/data/paper/README.md` for archive provenance.

## Exact public ARI pipeline

The public sampling notebook implements the following pipeline for ARI:

```text
exact prompt string
-> tokenizer(prompt, return_tensors="pt")
-> model.generate(max_length=116, do_sample=True, temperature=1,
                  top_k=None, eos_token_id=None, pad_token_id=None)
-> tokenizer.decode(full_output_tokens, skip_special_tokens=True)
-> completion = decoded_full[len(prompt):]
-> text_for_ari = prompt + completion
-> textstat.automated_readability_index(text_for_ari)
```

The prompt has 16 tokens, so `max_length=116` gives exactly 100 generated
tokens. The notebook asserts the fixed output length. It does not pass
`top_p`, so the Transformers default applies. It does not pass
`clean_up_tokenization_spaces`, strip whitespace, normalize newlines, alter
case, or otherwise preprocess the reconstructed string.

The exact prompt is:

```python
'Once upon a time, in a big forest, there lived a rhinoc'
```

Its UTF-8 SHA-256 is
`b4b7f60343cd3d1e24d8f2e312810afdaaa28034b9a194c66865b47483f70f59`.
With `EleutherAI/gpt-neo-125M`, the audited prompt token IDs are:

```text
[7454, 2402, 257, 640, 11, 287, 257, 1263, 8222, 11, 612, 5615,
 257, 9529, 259, 420]
```

The public environment lock records `textstat==0.7.13`,
`transformers==5.8.1`, `torch==2.12.0+cpu`, and `pymbar==4.0.3`. The archived
data do not record model or tokenizer revisions, and the public notebook does
not pin them explicitly.

### Public code versus paper observable

The public notebook states that the paper data were produced with a custom
`py-readability`-based implementation that handles edge cases and is not
published. The paper observable also caps ARI at 15 for the biased runs.
Consequently, this repository uses three distinct labels:

| Label | Meaning |
| --- | --- |
| `current` | Decode completion token IDs separately, then prepend the exact prompt before `textstat` ARI. This is the historical suffix-SMC convention. |
| `official_public` | Decode the complete prompt-plus-completion token sequence, character-slice the prompt, prepend the exact prompt, then use `textstat` ARI. |
| `official_paper_ari_unpublished` | Observable represented by the released paper arrays; its exact edge-case implementation is unavailable. |

`current` remains the default, preserving previous experiments and cache
fingerprints. `official_public` is selectable in direct, SMC, TPS, tail, and
analysis runners with `--ari-convention official_public`. The two public
conventions can differ at tokenizer boundaries even when the underlying token
sequence is identical.

## Official data inventory

The downloaded archive is `external_data/rea_paper/paper_data.zip`:

- size: 145,029,767 bytes;
- SHA-256: `3f29f498e2ecfa38816536610155f3069b394a8c25c694eb0204840a652677bd`;
- MD5: `83ad4e8f245edf5f45d879ae08450c3e`.

It contains JSON scalar arrays only. It contains no completion text, token IDs,
seeds, cut points, acceptance indicators, runtime, or hardware metadata.

### Direct ARI

`ari_unbiased_samples.json` contains 4,803,396 finite raw ARI values. The
extracted file SHA-256 is
`88832f5aa8efd5cf6c768a10ad1a3f0b6ce767d46c2ac33035ff81689cf1b82d`.

Audited summary:

| Statistic | Value |
| --- | ---: |
| count | 4,803,396 |
| mean | 2.53390436 |
| standard deviation | 1.34902650 |
| minimum / maximum | -6.10625076 / 45.65370178 |
| median | 2.49363 |
| 99th / 99.5th percentile | 5.87529 / 6.28317 |
| count ARI >= 6 / 7 / 8 / 10 / 15 | 39,066 / 6,497 / 940 / 48 / 15 |

`log_probs_unbiased_samples.json` contains the same number of model
log-probabilities, ranging from about -423.35 to -29.10. Array positions can be
paired with the ARI array, but the original completions cannot be recovered.

These values are reusable for empirical paper-observable tail probabilities,
quantiles, and boxplots. They can provide the lambda-zero state for an MBAR
calculation under the same paper observable. They are not an exact reference
for `official_public` until calibration establishes empirical agreement. They
cannot be rescored and do not support a runtime comparison. Under the documented
fixed 100-token setup, the direct transition count is count times 100; this is
not a measured wall-clock cost.

### TPS ARI

`ari_trajectories.json` is a three-element payload containing nested
trajectories, bias schedules, and `steps_per_bias=40000`. It contains:

- two schedule groups, one negative and one positive;
- ten independent chain arrays per schedule group;
- ten lambda levels per chain;
- 400,001 stored values per chain;
- 8,000,020 scalar values in total.

The negative schedule is `[-0.1, -0.2, ..., -1.0]`; the positive schedule is
`[0.1, 0.2, ..., 1.0]`. The public loader labels only the first 400,000 values
of each chain as ten blocks of 40,000. Thus 8,000,000 values have a public
level assignment and the last value of every chain is silently unused. The
nested structure preserves schedule, chain identity, level blocks, and
temporal order for the labelled portion.

The ARI trajectories range from about -6.1404 to exactly 15, with 65,439 values
at 15. This is consistent with the paper's capped ARI observable. The values
must not be treated as IID. The corresponding log-probability trajectories
have the same nested dimensions and use their own smaller bias schedules.

The ARI TPS data support per-chain burn-in, drift checks, temporal ESS,
Gelman-Rubin diagnostics, and MBAR after preserving chain/level membership.
They do not support rescoring, exact regenerated-token counts, or runtime.
Because cut points were not stored, suffix-token cost can only be approximated
from the documented proposal law; the paper reports roughly 50 generated
tokens per TPS step and roughly 4e8 generated tokens overall.

## How the public MBAR calculation works

The public reweighting notebook applies 10% burn-in independently inside each
level block, rejects levels failing its Gelman-Rubin criterion, flattens the
accepted samples, and calls PyMBAR as:

```python
u_kn = np.outer(biases, all_scores)
mbar = pymbar.MBAR(u_kn, Ns)
```

For targets proportional to `p_M(x) exp(-lambda * phi(x))`, the base-model
term is common to every thermodynamic state and cancels in reduced-potential
differences. MBAR therefore needs `phi(x)`, each target lambda, the per-level
sample counts `N_k`, and the association used to construct those counts. No
separate LM log-probability is required for this ARI reweighting. The original
PyMBAR implementation should be reused rather than duplicated.

## Project-run inventory

No completed SMC component archives, current direct arrays, or TPS production
runs are present in the local working copy. The ignored CSV files under
`results/tail_probability_validation` contain headers only and are not sample
data. Drive artifacts were not available locally, so no result is inferred for
them.

The expected compact SMC component archive stores metadata, `raw_ari`, `phi`,
final log weights, an SMC log-normalizer estimate, and runtime. It does not
store completion text/token IDs, realized suffix sizes, proposal counts, or
hardware metadata. A complete legacy archive remains reusable for the
`current` benchmark. It cannot be rescored, and its weights, resampling
history, and log normalizer cannot be retargeted to `official_public`.

The current direct archive stores a raw-ARI array and strict generation/scoring
metadata, but no completion text/token IDs or comparable runtime metadata. A
full single-run `samples.jsonl` may contain text and token IDs; rescoring such
final states is useful diagnostically but still does not change the target from
which the SMC population was generated.

The Colab quickstart contains one optional read-only audit cell. With
`RUN_DRIVE_AUDIT=True`, it inspects recognized archives in memory and displays
their reuse dimensions. It does not generate samples or write audit reports.

## Reuse matrix

`Yes*` denotes use only under the stored observable convention. `Expected`
denotes a compute estimate from a documented proposal law rather than realized
cut/suffix records.

| Artifact | Probability | MBAR | Temporal ESS | Runtime | Compute count | Rescore phi | Official comparison |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Official direct ARI | Yes* | Yes, lambda zero* | No | No | Yes, fixed-length transitions | No | Paper archive only; `official_public` unverified |
| Official TPS ARI | Yes, through MBAR* | Yes* | Yes | No | Expected only | No | Paper archive only; exact public equivalence unverified |
| Complete compact suffix SMC (`current`) | Yes* | No | No | No comparable hardware record | Expected only | No | No |
| Current direct reference | Yes* | Lambda-zero input if needed* | No | No comparable hardware record | Yes, fixed-length transitions | No | No |
| Full SMC samples with text/tokens | Yes for original target* | No | Depends on trajectory | Metadata-dependent | Metadata-dependent | Yes, diagnostically | Not as a retargeted SMC run |

The programmatic classifier in `rea_llms.data_reuse_audit` records separate
booleans and reasons for probability estimation, MBAR, temporal ESS, runtime,
compute count, rescoring, and official comparison. Missing evidence is reported
as unavailable rather than guessed.

## Benchmarks that must remain separate

### Current-convention benchmark

Combine only the current direct reference, old/current suffix SMC archives, and
TPS runs explicitly generated with `current` ARI.

### Official-public benchmark

Generate direct, TPS, and SMC with `ARI_CONVENTION='official_public'`. Do not
mix old current-SMC weights or paper-observable TPS values into its estimator.

### Paper-archive comparison

Use the released direct and TPS scalar arrays under
`official_paper_ari_unpublished`. Until calibration succeeds, this is a
cross-implementation comparison, not exact validation of `official_public`.

## Calibration plan (not executed)

The quickstart exposes:

```python
ARI_CONVENTION = 'official_public'
RUN_OFFICIAL_PUBLIC_DIRECT = False
OFFICIAL_PUBLIC_DIRECT_NUM_SAMPLES = 200_000
```

When deliberately enabled, this creates a separate fixed-length IID direct
sample using the public decode convention. Compare its tail probabilities,
quantiles, and boxplot statistics with the 4,803,396 archived direct values,
including Monte Carlo uncertainty. Agreement would justify using the archive
as a practical reference, not prove implementation identity. Disagreement
means TPS and SMC must be rerun under one common published convention.

## Minimum future work

1. Run the 200,000-sample `official_public` direct calibration once.
2. If calibration agrees sufficiently, define the tolerance before inspecting
   SMC/TPS outcomes and document that the equivalence is empirical.
3. Otherwise rerun both TPS and suffix SMC with `official_public` ARI.
4. Compare runtime only in newly controlled runs on the same hardware and
   software environment.
5. Keep MBAR inputs homogeneous in observable convention, prompt, model,
   tokenizer, length policy, and generation settings.

The primary quickstart now prepares this calibration together with a separate
nine-component official-public suffix-SMC validation at thresholds 6, 7, and
8. It does not run TPS or MBAR, and both production flags remain false in the
committed notebook.

Unverifiable items remain: the exact unpublished paper ARI edge-case code,
paper model/tokenizer revisions and seeds, per-step TPS cut points, TPS runtime
and hardware, and the contents of Drive archives until the read-only audit is
run in Colab.
