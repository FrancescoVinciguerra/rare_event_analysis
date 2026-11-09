# Tail-probability validation

The comparison uses the **official empirical reference**, not exact analytic
ground truth. Only archives classified as `exactly_reusable` can enter the
deterministic mixture.

## Official empirical reference audit

- Summary: `C:\Users\UTENTE\Downloads\rea_of_llms_supplementary_code-main\results\reference_direct_sampling_summary.json`
- Origin notebook: `external/rea_of_llms_official/sampling_examples.ipynb`
- Official repository commit inspected: `9a2ae06323db98496d179cefb294db60a6cd9535`
- Model / revision: `roneneldan/TinyStories-8M` / `not published`
- Tokenizer / revision: `EleutherAI/gpt-neo-125M` / `not published`
- Exact prompt: `Once upon a time, in a big forest, there lived a rhinoc`
- `repr(prompt)`: `'Once upon a time, in a big forest, there lived a rhinoc'`
- Prompt SHA-256: `b4b7f60343cd3d1e24d8f2e312810afdaaa28034b9a194c66865b47483f70f59`
- Prompt token IDs: `[7454, 2402, 257, 640, 11, 287, 257, 1263, 8222, 11, 612, 5615, 257, 9529, 259, 420]`
- Prompt token count: `16`
- Completion: `100` fixed tokens
- Temperature / top-k / top-p: `1.0` / `None` / `not explicitly set; Hugging Face default`
- EOS / padding: `None` / `None`
- Dataset seed policy: `not published`
- Decode convention: decode the full prompt-token plus completion-token sequence with skip_special_tokens=True, then remove len(prompt) characters
- Observable scope: `decoded prompt plus completion`
- Paper ARI implementation: custom py-readability implementation with edge-case handling; implementation not published
- Public demo ARI: `textstat.automated_readability_index, uncapped`
- Paper cap: `15.0`; direct archived ARI is raw.
- Direct samples contain neither completions nor token IDs. Exact equivalence
  with the current textstat/decode convention remains unverified.

## Archive reuse audit

- Candidate archives found: `0`
- Exactly reusable: `0`
- Comparable but not cache-equivalent: `0`
- Incompatible: `0`
- Unverifiable: `0`
- Selected for the mixture: `0`

The mixture denominator, corrected weights, and ESS are constructed only
from the selected paths below. Excluded archives are never passed to the
estimator.

| archive | classification | selected | reason |
|:---|:---|:---:|:---|
| _none available in this output root_ | - | - | - |

## Estimator status

The SMC result is a plug-in deterministic-mixture SNIS estimate because the
component normalizing constants are estimated by SMC.


**Strict observable equivalence is unverified; this is not exact ground truth for the repository ARI convention.**

## Configuration

- Direct observations: `4803396`
- Sampling setup match: `True`
- Strict observable match verified: `False`
- Seed groups: `[]`
- Particles per component: `256`
- MCMC steps per level: `10`
- Lambda step: `0.25`
- ESS resampling threshold: `0.5`
- Independent terminal component runs: `0`
- Reused compatible component runs: `0`
- Final resampling: `false`
- Observable limitation: The archive has no completions or token IDs, and the paper's custom ARI edge-case implementation is not published. The official demo also decodes the full sequence and removes the prompt by character slicing, whereas this repository decodes completion token IDs separately. Recalculation with the repository textstat backend is impossible.

Components per seed group:

| cap | terminal lambda | mixture rho | reused seed groups |
|---:|---:|---:|:---|
| 6 | -0.5 | n_j / sum(n) | none |
| 6 | -1 | n_j / sum(n) | none |
| 6 | -1.5 | n_j / sum(n) | none |
| 7 | -1 | n_j / sum(n) | none |
| 7 | -1.5 | n_j / sum(n) | none |
| 7 | -2 | n_j / sum(n) | none |
| 8 | -1.5 | n_j / sum(n) | none |
| 8 | -2 | n_j / sum(n) | none |
| 8 | -2.5 | n_j / sum(n) | none |
| 10 | -2 | n_j / sum(n) | none |
| 10 | -2.5 | n_j / sum(n) | none |
| 10 | -3 | n_j / sum(n) | none |
| 10 | -3.5 | n_j / sum(n) | none |

## Component diagnostics

Reverse estimates use each component's normalized final SMC weights and
are computed in log-space. `log Z reverse` is a stability diagnostic, not
an exact reference.

| cap | lambda | seed group | threshold | component probability | corrected ESS/N | raw events | log Z SMC | log Z reverse | delta log Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - | - | - | - | - |

## Aggregate comparison

| threshold | direct events | direct | Wilson 95% | mixture mean | seed SD | difference (hat-direct) | ESS | ESS/N | mean runtime (s) |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - | - | - | - | - |

Each seed group is the replication unit. Particles within one SMC run
are not treated as independent replicates. The ARI boxplots use final
SMC weights for tilted targets and do not represent reconstructed p_M.

## Diagnostic interpretation

- No complete seed group is available for mixture estimation.
- The official/current ARI equivalence is unverified; an observable or decoding mismatch remains a live explanation.
- No complete exactly reusable component set is available locally; the numerical source of the discrepancy remains unidentified.
