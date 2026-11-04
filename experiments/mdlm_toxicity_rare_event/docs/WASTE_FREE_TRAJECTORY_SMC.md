# Waste-free trajectory SMC

This backend changes only the particle construction inside each tilted
component.  It retains the frozen MDLM decoder, native DDPM/SUBS path law,
rollback-depth distribution, exact rollback-MH acceptance rule, terminal
event, reverse-SNIS correction and external component mixture.

## Targets and incremental weights

For a complete native decoder path `tau`, define

```text
Pi_r(tau) proportional to P0(tau) exp(eta_r R(tau_terminal)),
0 = eta_0 < eta_1 < ... < eta_K = lambda.
```

The incremental potential is unchanged:

```text
G_r(tau) = exp((eta_r - eta_{r-1}) R(tau_terminal)).
```

If the level-`r-1` population has uniform weights, the normalizer increment is

```text
Z_hat_r / Z_hat_{r-1} = (1/N) sum_n G_r(tau_n).
```

The product of these increments is the existing SMC normalizer estimate.

## Waste-free population refresh

Choose a chain length `P >= 2` such that `N` is divisible by `P`, and let

```text
M = N / P.
```

At every transition from `eta_{r-1}` to `eta_r`:

1. Resample exactly `M` seed trajectories from the weighted population that
   represents `Pi_{r-1}`.
2. For every seed, run `P-1` transitions of the existing rollback-MH kernel
   invariant for `Pi_{r-1}`.
3. Retain the seed and every post-transition state.  The `M` chains of length
   `P` form exactly `M*P=N` unweighted states representing `Pi_{r-1}`.
4. Evaluate `G_r` on all `N` retained states, update the normalizer estimate,
   and normalize these weights to represent `Pi_r`.
5. At the following level, resample the next `M` seeds from this weighted
   population.

For chain `m`, the retained states are

```text
tau[m,1], tau[m,2], ..., tau[m,P].
```

Every transition is the same exact kernel used by
`trajectory_resample_move`, at the lambda of the population before the next
incremental potential:

```text
alpha(tau, tau') = min(1, exp(eta_{r-1} * (R(tau') - R(tau)))).
```

A rejected proposal records the current state again.  The rejected proposal
is not a particle.  Repeated states encode the correct MH holding time and may
be stored equivalently as states with multiplicity.

## Why intermediate states matter

All retained states generated at `eta_{r-1}` contribute to the estimate of
`Z(eta_r) / Z(eta_{r-1})` and, after weighting, are candidates for the next
seed resampling.  Thus an accepted state visited before the endpoint of a
chain is not discarded and may found a lineage at the following tempering
level.

## Extended-state justification

Treat one complete chain as a super-particle

```text
z = (tau[1], ..., tau[P]).
```

Its Feynman--Kac potential is

```text
G_wf(z) = (1/P) sum_p G(tau[p]).
```

Consequently, selecting a super-particle proportionally to `G_wf` and then a
position inside it proportionally to `G(tau[p])` is equivalent to selecting
one state from the flattened `N=MP` population proportionally to its ordinary
SMC weight.  The selected state starts a Markov chain whose kernel is invariant
for the current target.  This is Algorithm 2 and the extended Feynman--Kac
construction of Dau and Chopin, not a heuristic reuse rule.

The normalizer increment is the average super-particle potential:

```text
(1/M) sum_m G_wf(z_m) = (1/(M*P)) sum_m sum_p G(tau[m,p]).
```

The `N=MP` states are correlated.  They must not be reported as `N` IID draws.
Diagnostics therefore retain terminal chain identifiers and positions, the
number of seeds, acceptance and self-transition counts, distinct roots and
the complete resampling genealogy.

## Configuration

```yaml
smc:
  final_resampling: false
  tilted_sampler:
    type: trajectory_waste_free
    implementation_version: waste_free_trajectory_rollback_mh_v1
    num_tempering_levels: 8
    mutation_policy: every_level
    chain_length: 4
    rollback_steps: [4, 8, 16, 24, 32]
    rollback_probabilities: [0.20, 0.20, 0.20, 0.20, 0.20]
```

`final_resampling: false` is mandatory for this implementation: as in
Algorithm 2, the terminal result is the final weighted chain population.
`mutation_policy: every_level` is mandatory because the waste-free
construction refreshes the population at every transition, independently of
the ordinary ESS threshold.

The paper develops its main statements with multinomial resampling.  This
implementation keeps the benchmark's configured unbiased resampling scheme
(`systematic` in the current profile) so that the comparison does not silently
change that factor.  The extended-target weighting remains correct, but the
paper's precise multinomial asymptotic-variance theorem is not claimed for the
systematic configuration.  A theorem-matched ablation can explicitly select
`resampling_method: multinomial`.

The initial comparison profile is
`configs/mdlm_owt_unconditional_estimation_100tok_trajectory_waste_free.yaml`.
It uses `N=512`, `P=4`, and `M=128`.  This file prepares an experiment but does
not submit it.

## Compute accounting

At each level the backend evaluates

```text
M * (P - 1)
```

rollback proposals, not `N*P`.  The initial native decoder paths still cost
`N*T` decoder particle-steps.  Reports retain actual NFE, rollback
particle-steps, wall time and peak CUDA memory; comparisons must use those
measured quantities rather than treating correlated states as independent.
