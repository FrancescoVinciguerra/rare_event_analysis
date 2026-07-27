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

## CLI

```bash
python suffix_balanced_smc_experiment/run_experiment.py \
  --device cuda \
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
resampling and ancestry diagnostics.

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
