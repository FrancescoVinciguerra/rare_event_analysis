# Official direct-sampling ARI reference

## Provenance

- Paper: [Rare Event Analysis of Large Language Models](https://arxiv.org/abs/2602.06791)
- Dataset: [Zenodo 20395631](https://zenodo.org/records/20395631)
- DOI: `10.5281/zenodo.20395631`
- Zenodo publication date: 2026-05-26
- Archive: `paper_data.zip` (145,029,767 bytes)
- Verified MD5: `83ad4e8f245edf5f45d879ae08450c3e`
- Direct ARI member: `ari_unbiased_samples.json`

The archive checksum exactly matches Zenodo. It contains four JSON files and
no completion text, token IDs, generation metadata, or separate README.

## Official configuration

- Model: `roneneldan/TinyStories-8M` (revision not published)
- Tokenizer: `EleutherAI/gpt-neo-125M` (revision not published)
- Exact prompt (16 tokens): `Once upon a time, in a big forest, there lived a rhinoc`
- `repr(prompt)`: `'Once upon a time, in a big forest, there lived a rhinoc'`
- Prompt SHA-256 (exact UTF-8): `b4b7f60343cd3d1e24d8f2e312810afdaaa28034b9a194c66865b47483f70f59`
- Prompt token IDs: `[7454, 2402, 257, 640, 11, 287, 257, 1263, 8222, 11, 612, 5615, 257, 9529, 259, 420]`
- Completion: 100 fixed tokens
- Generation: `do_sample=True`, temperature 1.0, `top_k=None`, top-p not explicitly set
- EOS and padding IDs are passed as `None`; the demo asserts exact length
- Dataset seed: not published; notebook demo seeds are Torch=42 and Python=1234
- Demo batching: one completion at a time; the paper generation implementation is not published
- Decode convention: decode the full prompt-token plus completion-token sequence with `skip_special_tokens=True`, then set `completion_text = decoded_full_text[len(prompt):]`.
- `clean_up_tokenization_spaces` is not passed explicitly, so behavior depends on the tokenizer/version default.
- Exact public-demo ARI input: `prompt + decoded_full_text[len(prompt):]`. The original prompt string is reattached before calling ARI.
- Source cells: `external/rea_of_llms_official/sampling_examples.ipynb`, zero-based JSON cell 3 (`generate_tokens`, `compute_ari`) and cell 5 (`direct_sampling`).
- Decode fragment: `output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)`
- Prompt-slice fragment: `completion_text = output_text[len(prompt):]`
- ARI fragment: `automated_readability_index(prompt + c)`
- No `.strip()`, whitespace normalization, case conversion, or newline preprocessing is explicit in the public notebook.
- ARI formula: `4.71 * characters/words + 0.5 * words/sentences - 21.43`
- Paper method: custom py-readability implementation with edge-case handling; implementation not published
- Cap: described as a hard cap at 15 and consistent with min(raw_ari, 15), but the implementation and paired raw/capped TPS values are unavailable
- Public notebook demo: `textstat.automated_readability_index`, uncapped. The environment declares `textstat>=0.7.12`, not an exact version; edge cases are delegated to that installed version.
- Log-probability: re-tokenize decoded prompt plus completion, compute model log-softmax, zero prompt-token contributions, gather observed token IDs, and sum
- Published use: Figure 3b direct ARI/log-probability curves, direct data incorporated into MBAR, and Figure 4 direct-sampling Wilson error analysis

## Archive inventory

| member | role | observable | observations | shape | min | max | invalid |
|---|---|---:|---:|---|---:|---:|---:|
| `log_probs_unbiased_samples.json` | direct_iid | log_probability | 4,803,396 | `[4803396]` | -423.346 | -29.0972 | 0 |
| `ari_trajectories.json` | tps_markov_chain_states | ari | 8,000,020 | `[2, [10], [400001]]` | -6.14042 | 15 | 0 |
| `ari_unbiased_samples.json` | direct_iid | ari | 4,803,396 | `[4803396]` | -6.10625 | 45.6537 | 0 |
| `log_probs_trajectories.json` | tps_markov_chain_states | log_probability | 8,000,020 | `[2, [10], [400001]]` | -530.042 | 0 | 0 |

The trajectory files are TPS Markov-chain states, including repeated
current states after rejected proposals; they are not IID direct samples or
unaccepted proposal collections. The direct ARI member is the only official
IID ARI array.

## Count and representation discrepancies

- Experimental-setup text: 4.2 million direct completions.
- Figure 3b caption: 4.1 million direct completions.
- Notebook direct-sampling markdown: 10^6 completions plus the ARI cap gives approximate paper results.
- Notebook reconstruction comment: 4,200,000.
- Current reweighting-notebook formula with the archived trajectory metadata selects 2,200,000 direct values for its histogram.
- Actual Zenodo direct array: **4,803,396**.
- The paper says ARI is capped at 15, and the TPS array is capped at 15.
  The direct array is demonstrably **raw**: max=45.6537, 15 values exceed 15, and 0 equal 15 exactly.

The dataset content is primary for this analysis. No values above 15 were
lost from the direct array; the published cap applies to the TPS target/data.

Two documentation errors are also visible: the official data README calls
the log-probability arrays ARI arrays, and the repository README reverses
the notebook's `USE_EXAMPLE_DATA` setting (`False` selects `data/paper`).

## Direct ARI statistics

- N: **4,803,396**
- Mean: 2.53390436
- Population standard deviation: 1.34902650
- Min / max raw: -6.10625076 / 45.65370178
- Q1 / median / Q3: 1.60161018 / 2.49363327 / 3.41518021

| quantile | ARI |
|---:|---:|
| 0.9 | 4.28444672 |
| 0.95 | 4.81708717 |
| 0.99 | 5.87528992 |
| 0.999 | 7.16181946 |
| 0.9999 | 8.37901966 |

## Empirical upper tails

| threshold | count | probability | Wilson 95% lower | Wilson 95% upper | rule of three |
|---:|---:|---:|---:|---:|---:|
| 6 | 39,066 | 0.0081329959 | 0.0080530679 | 0.0082137106 | - |
| 7 | 6,497 | 0.0013525847 | 0.0013201139 | 0.0013858531 | - |
| 8 | 940 | 0.00019569488 | 0.00018357925 | 0.00020860993 | - |
| 10 | 48 | 9.99293e-06 | 7.5377024e-06 | 1.3247879e-05 | - |
| 12 | 17 | 3.5391627e-06 | 2.2097849e-06 | 5.6682723e-06 | - |
| 15 | 15 | 3.1227906e-06 | 1.8925348e-06 | 5.1527789e-06 | - |

For thresholds up to and including 15, replacing raw ARI by
`min(raw_ari, 15)` preserves every event indicator `ARI >= threshold`.
It does not preserve raw maxima, upper-tail magnitudes, or distributions
conditioned on values above 15.

## Comparability with suffix direct-MH SMC

- Model name, tokenizer name, prompt, 100-token fixed length, and temperature-one ancestral sampling match the current experiment.
- Strict observable equivalence: **not verified**. The archive has no completions or token IDs, and the paper's custom ARI edge-case implementation is not published. The official demo also decodes the full sequence and removes the prompt by character slicing, whereas this repository decodes completion token IDs separately. Recalculation with the repository textstat backend is impossible.
- The central direct boxplot is usable as an official reference. A future
  SMC boxplot must use weighted quantiles from final normalized weights.
- Model/tokenizer revisions, the exact dataset-generation environment, and
  dataset RNG seeds are not published.

## Reference assessment

High-precision empirical baseline for the official direct-sampling observable, but not exact ground truth for this repository's textstat raw-ARI convention until observable equivalence is established.

It is appropriate as a large IID baseline for probabilities under the
official observable at thresholds 6, 8, 10, 12, and 15. It should not be
described as exact ground truth for our raw `textstat` ARI until the custom
edge-case convention or original completions become available.
