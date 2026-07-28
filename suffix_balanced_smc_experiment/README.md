# Suffix-balanced full-sequence SMC

This experiment is an independent full-completion SMC pipeline. It does not
call direct sampling, mixture MCMC, TPS, MBAR, or either earlier SMC sampler.
Its unnormalized level target is

\[
\gamma_k(x)=p_M(x\mid c)\exp\{-\lambda_k\phi(x)\}.
\]

## Suffix proposal

For a state-independent suffix length \(m\), write a completion as
\(x=(u,s_x)\), where \(|s_x|=m\). The proposal keeps \(u\) fixed and samples
exactly \(m\) new tokens from the language model:

\[
y=(u,s_y), \qquad R_m(x,y)=p_M(s_y\mid c,u).
\]

For Barker and square-root balancing, the reverse probability is evaluated
exactly by teacher forcing:

\[
R_m(y,x)=p_M(s_x\mid c,u).
\]

At proposal temperature one, the shared-prefix language-model probabilities
cancel, so at mutation target \(\pi_{k-1}\),

\[
z=\log\frac{\pi_{k-1}(y)}{\pi_{k-1}(x)}
=\log R_m(x,y)-\log R_m(y,x)
-\lambda_{k-1}[\phi(y)-\phi(x)].
\]

Other proposal temperatures are rejected explicitly because this cancellation
would no longer be exact.

## Direct Metropolis fast path

With `--balancing-function metropolis`, proposal temperature one, fixed
completion length, and state-independent suffix-size selection, the direct MH
ratio is

\[
\frac{\pi_{k-1}(y)R_m(y,x)}{\pi_{k-1}(x)R_m(x,y)}
=\exp\{-\lambda_{k-1}[\phi(y)-\phi(x)]\}.
\]

Both suffix LM probabilities and the equal forward/reverse suffix-size
selection probabilities cancel. The implementation therefore uses

\[
\log A_{\mathrm{MH}}=
\min\{0,-\lambda_{k-1}[\phi(y)-\phi(x)]\}.
\]

This fast path generates the proposal and evaluates only \(\phi(y)\), reusing
the cached \(\phi(x)\). It does not evaluate full-completion probabilities,
forward or reverse suffix probabilities, teacher forcing, `a_sym`, or an LM
ratio. In particular, it does **not** use the lower-acceptance two-stage rule
`a_sym * min(1, pi(y)/pi(x))`.

`--disable-metropolis-fast-path` computes the algebraically equivalent full
direct-MH ratio for validation and benchmarking. It still performs one direct
MH decision; it does not restore the old two-stage kernel.

## Barker and square-root acceptance

The proposal is first symmetrized with

\[
\log a_{\mathrm{sym}}
=\min\{0,\log R_m(y,x)-\log R_m(x,y)\}.
\]

The configured balancing component is then added:

- Barker: \(\log g(e^z)=\log\sigma(z)\);
- square root: \(\log g_\delta(e^z)=\log\delta_k+z/2\).

The total log acceptance is

\[
\log A=\log a_{\mathrm{sym}}+\log g(e^z).
\]

For square-root balancing, certified bounds `phi_min` and `phi_max` are
required. The automatic safe scale is

\[
\delta_k=
\exp\{-|\lambda_{k-1}|(\phi_{\max}-\phi_{\min})/2\}.
\]

This bound uses the exact identity contributed by symmetrization,
\(\min(0,-d)+d/2=-|d|/2\), for
\(d=\log R_m(x,y)-\log R_m(y,x)\). It therefore certifies that the *combined*
acceptance is at most one without estimating a bound from observed samples.
An explicit `--sqrt-scale` may only reduce this certified scale. The invalid
construction `min(1, sqrt(t))` is never used.

Barker and scaled square root satisfy \(g(t)=t g(1/t)\) in the symmetrized
construction. Direct Metropolis satisfies detailed balance through the usual
MH proposal-ratio correction, which cancels exactly in this suffix setting.

## SMC order and resampling

At transition \(k\), mutation targets \(\pi_{k-1}\). The particles are then
reweighted to level \(k\):

\[
\log w_k^{(i)}=\log w_{k-1}^{(i)}
-(\lambda_k-\lambda_{k-1})\phi(X_k^{(i)}).
\]

Weights are normalized with `torch.logsumexp`. Systematic resampling occurs
only below the configured ESS threshold. The final level is deliberately not
resampled, preserving the final weighted particle approximation.

The standalone CLI defaults to `--ari-mode capped` so existing experiments
retain their original potential. Use `--ari-mode raw` to set
\(\phi(x)=\operatorname{ARI}_{\mathrm{raw}}(x)\); in that mode `--ari-cap` is
recorded for provenance but is not applied. The SMC/TPS comparison runner
requires raw mode and rejects capped configurations.

## Suffix schedules and irreducibility

The stochastic default adapts the multiscale support
`1,2,4,8,16,32,64,T` to the completion length. A fixed distribution uses
`--suffix-sizes` and `--suffix-rhos`; a per-transition distribution uses
semicolon-separated rows in `--suffix-rho-schedule`. Deterministic runs use
`--suffix-size-schedule`, with one row per transition and one value per MCMC
step when desired.

Positive probability for `m=T` gives a global move from every state and is
required by default. `--allow-local-suffix-support` disables that validation
only for deliberate local-kernel studies.

## GPU implementation

`--implementation parallel` is the default. It draws all suffix sizes on the
model device, groups particles by suffix length, regenerates each group in GPU
batches, and, when required, teacher-forces forward and reverse suffix
probabilities together. Metropolis skips that scoring pass entirely.
Acceptance, weight updates, ESS, and resampling are vectorized. Exact textstat
ARI remains one explicit batched CPU boundary per mutation step.

CUDA OOM errors halve the active batch and retry. An OOM at batch size one
fails clearly. The reference implementation uses the same mathematics but
processes proposals one particle at a time. Random-number consumption differs
between grouped and particlewise execution, so fixed-seed trajectories need
not match across implementations; fixed proposals and all probability factors
are tested exactly.

## Performance audit and approved mechanical optimizations

The current implementation is mathematically consistent with the target,
mutation order, direct-MH cancellation, incremental weights, ESS, systematic
resampling, and retained final weights described above. In particular,
`torch.logsumexp`, `torch.cumsum`, and `torch.searchsorted` are already used for
the numerically sensitive SMC operations. Future tokens of one completion
cannot be generated in parallel; the existing parallelism is across particles
that share a suffix length.

After review, three class-1 changes were approved and implemented: fixed-length
generation now writes into a preallocated output tensor, stochastic suffix-size
constants are materialized once per SMC transition, and `run_benchmark.py` can
attribute runtime by phase. No class-2 or class-3 proposal below has been
implemented.

The classifications below mean:

1. **Safe mechanical optimization**: no intended mathematical change and low
   implementation risk, but fixed-seed regression tests are still required.
2. **Requires equivalence testing**: intended to preserve the kernel, but exact
   proposal probabilities, serialized results, or random-number consumption
   could change and must be tested explicitly.
3. **Algorithmic change**: changes the proposal, target, acceptance, observable,
   or particle dependence and is outside this optimization pass.

| code location | potential inefficiency | expected cost | candidate optimization | class | correctness risk | priority | required verification |
|---|---|---|---|---:|---|---|---|
| `_generate_suffix_chunk` | Previously stored one tensor per generated token and concatenated at the end. | Temporary allocations proportional to batch size times suffix length. | **Implemented:** preallocate the fixed-size output tensor and fill one column per autoregressive step. | 1 | Low: in-place writes and tensor views must not affect sampled tokens. | Completed | Exact fixed-seed token equality against the previous list-and-cat path; CUDA benchmark. |
| `_generate_suffix_chunk` | Manual decoding loop may miss optimized cache implementations available through Transformers. | One output-head projection and multinomial draw per token; cache implementation can dominate long suffixes. | Benchmark `model.generate` or a Transformers static/dynamic cache path with unrestricted sampling, temperature 1, no EOS stopping, no penalties, no beams, `top_k=0`, and `top_p=1`. | 2 | High: defaults, stopping rules, cache positions, and RNG batching can change the proposal. | High | Token-level proposal law, full support, fixed length, log-probability checks, stationary-distribution test, and throughput/memory benchmark. |
| `_generate_suffix_chunk` fallback | Repeatedly concatenates and re-evaluates the growing prefix when `base_model`/KV cache is unavailable. | Superlinear work and allocation growth in suffix length. GPT-Neo uses the cached path, so this mainly affects other models. | Use the model's documented `prepare_inputs_for_generation`/cache API or fail clearly for unsupported cache behavior. | 2 | Medium: model-specific cache semantics and position IDs must match a full forward pass. | Medium | Next-token logits and sampled-suffix probabilities versus full-prefix recomputation for each supported architecture. |
| `sample_suffix_sizes` | Previously recreated suffix-size and probability tensors at every MCMC step. | Small repeated host-to-device allocations and copies; noticeable only with many levels/steps. | **Implemented:** materialize immutable per-transition tensors once on the model device. | 1 | Low: the multinomial call and its ordering must remain unchanged. | Completed | Exact fixed-seed suffix-size draws across repeated MCMC steps and schedule tests. |
| `propose_suffixes_parallel` / `score_fixed_suffix_proposals` | `torch.unique(...).cpu().tolist()` synchronizes the GPU before looping over size groups; groups also allocate `index_select`, prompt, context, and proposal tensors. | One synchronization and several group-sized copies per MCMC step. | Reuse the configured sorted support and preallocated index/context buffers, skipping empty groups without changing group order. | 2 | Medium: group order changes RNG consumption and buffer reuse can introduce aliasing. | High | Fixed proposals per size, empirical proposal distribution, acceptance factors, reproducibility within the parallel implementation, and peak memory. |
| `_paired_suffix_log_probabilities_chunk` | Duplicates contexts for forward/reverse teacher forcing and materializes hidden states for both paths. `log_softmax(...).gather(...)` also builds full-vocabulary chunks. | Large activation and vocabulary-logit memory for Barker/sqrt; absent from the Metropolis fast path. | Reuse a cached prefix for the two suffix branches; benchmark selected-token `F.cross_entropy` against `log_softmax` plus `gather`. | 2 | Medium/high: cache branching and fused reductions can change indexing or numerical results. | Medium | Forward/reverse suffix log probabilities against enumeration, acceptance equality, detailed balance, and memory benchmark. |
| `score_observable` | Copies prompt plus every completion to CPU and decodes text for exact ARI. Textstat itself is scalar CPU work. | Major synchronization/CPU bottleneck as particles and MCMC steps grow. | Transfer completion IDs only and share an exact decoded-text batch with downstream consumers. | 2 | High: BPE boundaries, punctuation, repetition, and concatenated words must decode identically to final evaluation. | High | Exact raw/capped ARI equality on randomized real-tokenizer sequences and explicit tokenizer-boundary cases. |
| `track_discoveries` | Transfers and decodes every proposal again after ARI scoring and stores every unique proposal in `best_discoveries`, although only the top 20 are reported. | A second GPU-to-CPU boundary plus potentially unbounded strings and sets. | Reuse decoded text from ARI evaluation; maintain an exact bounded top-results structure while retaining threshold-unique sets and first-seen metadata. Make tracking optional only in a separate approved change. | 2 | Medium/high: rejected proposals, duplicate text, ties, and first-seen times are observable report semantics. | High | Byte-for-byte discovery summaries on fixed proposals, duplicates, ties, accepted/rejected cases, and all thresholds. |
| `apply_suffix_balanced_mutation` | `current.clone()`, grouped copies, and `torch.where` produce full `N x T` token tensors at every MCMC step, even when few proposals are accepted. | Device memory bandwidth and allocation pressure proportional to all particles and completion length. | Update accepted rows with carefully isolated `index_copy_`/masked buffers and reuse proposal storage. | 2 | Medium: in-place aliasing can corrupt rejected states or cached observables. | High | Exact accept mask, tokens, cached `phi`, raw ARI, and ancestry for acceptance rates 0, 1, and mixed. |
| `metropolis_log_acceptance_*`, `kernel_components`, level diagnostics | Finite checks and diagnostics call `.item()`, forcing synchronization. | Several small CUDA stalls per MCMC step/level; usually secondary to generation and ARI. | Aggregate validation flags and diagnostic scalars, then transfer once at a deliberate reporting boundary. | 2 | Medium: failures must remain clear and level timing semantics can shift. | Low | Invalid/non-finite injection tests and equality of every reported diagnostic. |
| `_final_samples` | Re-evaluates exact ARI for final particles, decodes again, and performs several separate device-to-CPU transfers. | Duplicate CPU work and synchronization at the end of every run. | Carry or reconstruct generic score metadata from accepted cached scores and batch final transfers/decoding once. | 2 | Medium: every existing `samples.jsonl` field must remain identical. | Medium | Byte-for-byte sample rows for cached/uncached and raw/capped observables, including resampled ancestry. |
| ARI/discovery caches | Exact ARI cache, discovery dictionaries, and threshold sets have no size bound. | Host memory grows with the number of unique evaluated proposals. | Add an optional bounded/LRU score cache and compact exact discovery structures. | 2 | Medium: eviction must not alter scores; compact discovery must preserve uniqueness and first occurrence. | Medium | Long-run memory profile and exact result equality under forced cache eviction. |
| `run_benchmark.py` | Previously exposed total and per-level runtime but not phase attribution. | Bottlenecks could not be distinguished without ad hoc instrumentation. | **Implemented:** benchmark-only CPU timers/CUDA Events for initial generation, suffix generation/scoring, ARI, discovery, acceptance, and reweight/ESS/resampling. | 1 | Low: profiling must not change particles or synchronize every inner GPU call. | Completed | Profiled/unprofiled particle equality; one final event synchronization; CPU and CUDA benchmark runs. |
| Proposal/observable alternatives | Vocabulary truncation, approximate or incremental non-equivalent ARI, parallel future-token generation, state-dependent suffix selection without proposal correction, or skipping proposals based on ARI. | Could be faster but changes support, target, transition kernel, or reported discovery set. | None in this audit. Treat each as a new method with its own derivation. | 3 | Unacceptable for an equivalence optimization. | Not applicable | New mathematical proof, stationary tests, and separate experimental labeling. |

### Audit benchmark snapshot

The phase-enabled benchmark was run on an NVIDIA GeForce MX570 A with local
TinyStories-8M weights. With 8 particles, 8 completion tokens, one transition,
one MCMC step, and batch size 8, the grouped implementation took 0.344 s versus
2.110 s for the particlewise reference (6.13x in this microbenchmark), with
173,725,184 bytes peak allocated CUDA memory. Its measured phases were 0.142 s
for initial generation, 0.129 s for suffix generation, 0.021 s for suffix
probability scoring, 0.004 s for ARI, 0.001 s for discovery, 0.003 s for
acceptance, and 0.001 s for reweighting/ESS/resampling.

In a less noisy direct-MH check with 32 particles, 20 tokens, two MCMC steps,
and batch size 16, the fast path took 1.532 s versus 1.636 s for the full-ratio
reference (1.07x), matched final token particles, and used 141,718,528 bytes
peak allocated CUDA memory.

These single-run numbers validate that batching and phase attribution are
active; they are not estimates for a long experiment or statistically stable
speedup claims. Profiling uses CPU timers off CUDA and CUDA Events on GPU, with
one final synchronization. It writes no files and does not change the sampler's
CLI or result schemas.

## CLI

```bash
python suffix_balanced_smc_experiment/run_experiment.py \
  --device cuda \
  --ari-mode raw \
  --num-particles 256 \
  --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-0.75,-1,-1.25,-1.5,-1.75,-2 \
  --mcmc-steps-per-level 10 \
  --suffix-mode stochastic \
  --suffix-sizes 1,2,4,8,16,32,64,100 \
  --suffix-rhos 0.20,0.18,0.16,0.14,0.12,0.10,0.06,0.04 \
  --balancing-function barker \
  --batch-size 32 \
  --experiment-label suffix_balanced_long \
  --seed 200
```

Normal runs write only `summary.json`, `samples.jsonl`, and `report.md` under
`outputs/<experiment-label>/seed_<seed>/`. A completed run is not overwritten
unless `--overwrite` is explicit. The summary contains weighted and
unweighted ARI statistics, tail summaries, ESS, acceptance components,
resampling and ancestry diagnostics. It also reports the exact number of
mutation proposals and regenerated proposal tokens, their throughputs, and
peak CUDA memory. Initial particle generation is intentionally excluded from
the proposal and regenerated-token counters for both benchmarked methods.

The focused reference/parallel benchmark is:

```bash
python suffix_balanced_smc_experiment/run_benchmark.py --device cuda
```

It reports only total runtime, particles per second, mean time per level, peak
GPU memory, and speedup over the reference implementation.

The focused direct-MH fast-path benchmark is:

```bash
python suffix_balanced_smc_experiment/run_benchmark.py \
  --benchmark-mode metropolis-fast-path --device cuda
```

Both paths are warmed before timing. The benchmark also requires identical
final token particles at the shared seed before reporting the speedup.

For resumable independent multi-seed Colab runs, open
`notebooks/REA_Suffix_Balanced_SMC_Quickstart.ipynb`.
