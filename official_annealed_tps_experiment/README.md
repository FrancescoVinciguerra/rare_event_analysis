# Official annealed TPS adapter

This package is a minimal adapter for Algorithm 2 as implemented in the
official supplementary notebook:

- repository: `https://github.com/MLiS-Research/rea_of_llms_supplementary_code`;
- pinned submodule commit: `9a2ae06323db98496d179cefb294db60a6cd9535`;
- source: `sampling_examples.ipynb`;
- replicated functions: `generate_tokens`, `generate_proposal`,
  `compute_tps_acceptance_factor`, `tps_step`, and `biased_sampling`.

The notebook functions are not importable as a Python module. The adapter
therefore ports only their cut-index, suffix-regeneration, acceptance, and
annealing semantics. The official submodule is not modified.

## Exact cut convention

The official notebook defines

```python
choice_idxs = list(range(prompt_token_length, max_length))
idx = random.choice(choice_idxs)
input_tokens_proposal = current_tokens[:, 0:idx]
```

For a completion of length `T`, the relative retained completion-prefix length
is uniform on `0,...,T-1`. Consequently:

- the first completion token can be regenerated;
- suffix size is uniform on `1,...,T`;
- at least one token is always regenerated;
- the empty-suffix case does not occur;
- generated completions remain exactly length `T` with EOS disabled.

The adapter uses the repository's tested cached suffix generator instead of a
scalar call to `model.generate`. Both generate exactly from
`p_M(suffix | prompt, retained prefix)` at temperature one.

For the observable, both benchmark methods deliberately call the same shared
exact token-sequence ARI backend: decode completion token IDs, prepend the
original prompt, then evaluate `textstat`. The official notebook decodes the
full sequence and removes the first `len(prompt)` characters. Those two text
conventions can differ at a BPE boundary, so this is recorded as an
instrumentation difference needed to give SMC and TPS exactly the same target.

## Target and acceptance

At each bias, the target is

\[
\pi_\lambda(x)\propto p_M(x\mid c)e^{-\lambda\phi(x)}.
\]

Because the cut distribution is state-independent, completion length is fixed,
and the suffix is regenerated from the base LM, direct MH simplifies exactly:

\[
\log A(x,y)=\min\{0,-\lambda[\phi(y)-\phi(x)]\}.
\]

The implementation runs one chain per seed, completes every configured step at
every bias, and warm-starts each bias from the final state of the previous one.
It has no particle population, importance weights, weight ESS, or resampling.

The compact trajectory records the state *after* every accept/reject. The
official notebook's `step_info["current_obs"]` is populated before the
transition; using the post-transition value here follows the explicit estimator
definition while changing only this diagnostic indexing, not the Markov chain.

Tail-discovery fields include every generated and evaluated proposal, whether
accepted or rejected, plus the initial state. They are compute-level discovery
diagnostics and are not used as final-target samples.

## Output and diagnostics

A run writes only:

- `summary.json`;
- `trajectory.npz`;
- `report.md`.

The final-target estimator uses only the final-bias row of the trajectory.
Temporal ESS uses Geyer's initial positive sequence. Consecutive TPS samples
are correlated, and completion of a finite chain is not evidence of exact
convergence. Usual MCMC requirements include irreducibility, aperiodicity, and
sufficient chain length.

```bash
python official_annealed_tps_experiment/run_experiment.py \
  --device cuda --ari-mode raw --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-0.75,-1 \
  --steps-per-bias 2000 --seed 200
```
