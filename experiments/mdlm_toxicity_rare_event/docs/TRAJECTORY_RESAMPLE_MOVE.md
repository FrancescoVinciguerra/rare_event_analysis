# Trajectory resample-move SMC

This backend changes only how each tilted component is sampled. The frozen
MDLM decoder, DDPM/SUBS kernel, terminal categorical draw, reward, event,
reverse-SNIS estimator and component aggregation are unchanged.

## Target

Let a complete native decoder trajectory be

```text
tau = (x_T, x_{T-1}, ..., x_preterminal, x_0)
```

with base path probability `P0(tau)`. For one external component with terminal
tilt `lambda`, introduce SMC tempering levels

```text
0 = eta_0 < eta_1 < ... < eta_K = lambda
Pi_k(tau) proportional to P0(tau) exp(eta_k R(x_0)).
```

The incremental weight is

```text
exp((eta_k - eta_{k-1}) R(x_0)).
```

`eta_k` is an SMC tempering level. It is not diffusion time and does not alter
the decoder.

## Exact rollback mutation

For a configured rollback depth `d`, retain the checkpoint actually visited
at decoder step `T-d` and the complete path before it. Regenerate the remaining
`d` DDPM steps and terminal draw with the unchanged native decoder. The old
and proposed paths share the retained checkpoint.

For a fixed, state-independent rollback-depth distribution, the forward and
reverse native suffix probabilities cancel from the Metropolis-Hastings ratio:

```text
alpha(tau, tau') = min(1, exp(eta_k (R(x'_0) - R(x_0)))).
```

Thus every mutation kernel is reversible and invariant for `Pi_k`. Finite
MCMC sweeps do not make particles independent or guarantee exact finite-time
mixing; they do preserve the intended target and can diversify copies produced
by resampling.

No token positions are selected or artificially remasked. Rolling back simply
restores a stored state from the particle's real trajectory.

## GPU execution

Rollback depths are sampled independently per particle. Within a mutation
sweep particles are grouped by equal depth, so each depth is regenerated as a
batched GPU workload. All proposed terminal completions are then scored in one
batched reward call and accepted or rejected vectorially.

The full path uses approximately

```text
N * (T + 2) * L * sizeof(torch.long)
```

bytes. The proposal sweep temporarily holds a second path tensor. Model
activations remain governed by `microbatch_size`.

## Configuration

The current FK implementation is selected explicitly with:

```yaml
smc:
  tilted_sampler:
    type: fk_bootstrap
```

The trajectory backend is selected with:

```yaml
smc:
  tilted_sampler:
    type: trajectory_resample_move
    num_tempering_levels: 8
    mcmc_steps_per_level: 1
    mutation_policy: every_level
    rollback_steps: [4, 8, 16, 24, 32]
    rollback_probabilities: [0.20, 0.20, 0.20, 0.20, 0.20]
```

`mutation_policy: every_level` applies an invariant move at every `eta_k`.
`after_resampling` applies it only when resampling occurred. Rollback
probabilities must be fixed, non-negative and sum to one. State-dependent
routing is deliberately not implemented because it would require its
forward/reverse selection ratio in the acceptance rule.

The ready comparison configuration is
`configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml`. It
inherits the same four tilted components and uniform convex combination as the
existing 100-token FK run. Its decoder and evaluator IDs are identical, while
its SMC ID differs.

For a first compute-matched comparison use
`configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh_matched_fk.yaml`.
It retains 512 particles and mutates only after resampling, with one sweep and
rollback depths 4 or 8. The previous FK run used 1,280 model forward calls per
component. The trajectory base path uses 1,056 at microbatch 16; one balanced
4/8 rollback sweep adds approximately 224, placing a component with one
resampling at approximately the same 1,280 calls. Components with zero or
multiple resampling events will cost less or more, so reports must compare
actual NFE, particle-steps and runtime rather than assuming equality.

## Diagnostics

Component artifacts record the sampler type, trajectory storage, proposals,
acceptances, acceptance rate, selected rollback depths, terminal token-change
fractions, model NFE, resampling genealogy, forward ESS and reverse ESS. Compare
backends both at equal particle count and at equal NFE or GPU time.
