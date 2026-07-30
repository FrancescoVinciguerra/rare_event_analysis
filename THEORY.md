# Theory of the active suffix direct-MH SMC sampler

This note derives the algorithm implemented in
`suffix_direct_mh_smc_experiment/`. Citations identify the origin of ideas and
standard terminology. Correctness claims below follow from the explicit kernel
and weight calculations, not automatically from any cited theorem.

## 1. Problem formulation

Fix a prompt `c`. Let `x = x_1:T` be a completion of exactly `T` tokens and let

```text
p_M(x | c)
```

be the probability assigned by the autoregressive language model. Let `phi(x)`
be an observable evaluated on the decoded prompt plus completion. In the active
experiments it is raw or capped Automated Readability Index (ARI), using the
same exact decode-and-`textstat` convention during mutation and final scoring.

For each finite bias `lambda`, define

```text
gamma_lambda(x) = p_M(x | c) exp(-lambda phi(x)),
Z_lambda        = sum_x gamma_lambda(x),
pi_lambda(x)    = gamma_lambda(x) / Z_lambda.
```

When `lambda < 0`, larger values of `phi` receive more mass. When `lambda > 0`,
smaller values receive more mass. `lambda = 0` recovers `p_M`.

This exponential target and the rare-completion problem are the starting point
provided by Dorman et al. (2026). The active sampler changes the transition and
population method; it does not alter this target.

## 2. Suffix proposal

Choose a suffix length `m` from a fixed distribution `rho` on a configured set
of lengths. Write the current completion as

```text
x = (u, x_s),
```

where `u = x_1:T-m` is retained and `x_s = x_(T-m+1):T` has length `m`. Generate
a fresh suffix autoregressively from the unmodified base model,

```text
y_s ~ p_M(. | c, u),
y   = (u, y_s).
```

For fixed `m`, the proposal probability is

```text
q_m(y | x) = 1{y_1:T-m = x_1:T-m} p_M(y_s | c, u).
```

The implementation has fixed completion length, temperature one, no early EOS
stopping, and no top-k, top-p, repetition penalty, or other logit transform.
Particles with equal `m` are generated together on the GPU, but tokens within a
suffix remain autoregressive.

### Random scan over suffix lengths

The code first samples `m`, then applies the MH kernel associated with `q_m`.
Thus the mutation kernel is

```text
M_lambda = sum_m rho_m M_(lambda,m).
```

`rho_m` is a random-scan probability, not an SMC weight. If each component
kernel leaves `pi_lambda` invariant and `rho` is state-independent, then

```text
sum_x pi_lambda(x) M_lambda(x,y)
  = sum_m rho_m sum_x pi_lambda(x) M_(lambda,m)(x,y)
  = sum_m rho_m pi_lambda(y)
  = pi_lambda(y).
```

Positive probability for `m=T` is required by default so the proposal can
refresh the entire completion. This supports global reachability when the LM
itself has the needed support; it does not guarantee fast mixing.

## 3. Metropolis-Hastings correctness

### General rule

For a fixed suffix length, ordinary Metropolis-Hastings accepts `y` from `x`
with

```text
alpha_lambda,m(x,y)
  = min(1, [pi_lambda(y) q_m(x | y)]
           / [pi_lambda(x) q_m(y | x)]).
```

This is direct MH. The active implementation does not multiply a symmetric
pre-acceptance probability by another Metropolis correction.

### Exact cancellation of language-model terms

For a reachable proposal, `x=(u,x_s)` and `y=(u,y_s)` share the retained prefix.
Autoregressive factorization gives

```text
p_M(x | c) = p_M(u | c) p_M(x_s | c,u),
p_M(y | c) = p_M(u | c) p_M(y_s | c,u),
q_m(y | x) = p_M(y_s | c,u),
q_m(x | y) = p_M(x_s | c,u).
```

Substitution into the full ratio gives

```text
 pi_lambda(y) q_m(x | y)
--------------------------------
 pi_lambda(x) q_m(y | x)

  p_M(u|c) p_M(y_s|c,u) exp[-lambda phi(y)] p_M(x_s|c,u)
= ------------------------------------------------------------------
  p_M(u|c) p_M(x_s|c,u) exp[-lambda phi(x)] p_M(y_s|c,u)

= exp{-lambda [phi(y) - phi(x)]}.
```

`Z_lambda` also cancels because both states belong to the same target. The
implemented log acceptance is therefore exactly

```text
log alpha_lambda,m(x,y)
  = min(0, -lambda [phi(y) - phi(x)]).
```

The current `phi(x)` is cached in the particle state; only `phi(y)` is evaluated
for a proposal. No complete-sequence LM probability, reverse teacher-forcing
score, or local normalizer is needed.

### Conditions for the fast formula

The cancellation requires all of the following:

- forward and reverse use the same selected suffix length `m`;
- `x` and `y` share the same retained prefix;
- the proposed suffix is sampled exactly from `p_M(. | c,u)` at temperature one;
- no truncation, penalty, early stop, or asymmetric EOS handling changes it;
- suffix-length selection is state-independent, or its forward/reverse ratio is
  explicitly included.

At temperature different from one, after top-k/top-p filtering, with variable
asymmetric lengths, or with state-dependent suffix selection, the displayed
cancellation is generally invalid and the full proposal ratio is required. The
active configuration excludes these cases rather than silently approximating
them.

### Detailed balance and stationarity

Let

```text
r(x,y) = pi_lambda(y)q_m(x|y) / [pi_lambda(x)q_m(y|x)].
```

For `x != y`, the accepted transition density is

```text
M_(lambda,m)(x,y) = q_m(y|x) min(1,r(x,y)).
```

Then

```text
pi_lambda(x) M_(lambda,m)(x,y)
  = min{pi_lambda(x)q_m(y|x), pi_lambda(y)q_m(x|y)}
  = pi_lambda(y) M_(lambda,m)(y,x).
```

The rejection probability supplies the diagonal term, so detailed balance also
holds for the complete kernel. Summing over `x` proves stationarity:

```text
sum_x pi_lambda(x) M_(lambda,m)(x,y) = pi_lambda(y).
```

The state-independent random-scan mixture over `m` is stationary by the linear
argument in Section 2. This is the correctness argument for the active mutation
kernel.

### Relation to locally informed proposals

Zanella (2020) studies locally informed constructions involving proposal
weights such as balancing functions satisfying `g(t)=t g(1/t)`, with examples
including Barker, square-root, and Metropolis choices under the paper's stated
base-kernel assumptions and local normalization. Those ideas motivated earlier
experiments in this project.

The active proposal is instead the generally asymmetric conditional LM suffix
distribution. It has no target-ratio-informed local proposal normalizer and no
active balancing function. We therefore do not identify it with Zanella's
locally balanced construction and do not use that paper as its correctness
proof. The direct calculation above is sufficient.

## 4. Sequential Monte Carlo

Choose a non-increasing high-ARI schedule

```text
lambda_0 = 0, lambda_1, ..., lambda_K.
```

The particles start independently from the language model:

```text
X_0^i ~ p_M(. | c),       W_0^i = 1/N.
```

For transition `k=1,...,K`, the code performs these operations in order:

1. Apply one or more suffix direct-MH moves targeting `pi_(lambda_(k-1))`.
2. Reweight the mutated particles from `gamma_(k-1)` to `gamma_k`.
3. Compute normalized weights and ESS.
4. If ESS/N is below the threshold and `k<K`, resample systematically and reset
   weights to `1/N`.

The mutation target is deliberately the previous distribution. Since that
kernel is invariant, mutation does not introduce an extra importance ratio. The
incremental ratio is

```text
G_k(x) = gamma_(lambda_k)(x) / gamma_(lambda_(k-1))(x)
       = exp{-[lambda_k-lambda_(k-1)] phi(x)}.
```

Thus, before normalization,

```text
log w_k^i = log W_(k-1)^i
            - [lambda_k-lambda_(k-1)] phi(X_k^i).
```

The implementation normalizes with one `torch.logsumexp` and computes

```text
ESS_k = 1 / sum_i (W_k^i)^2
```

in log-space. Systematic resampling is a selection step, not a target change.
It duplicates particles in proportion to normalized weights and resets the new
weights to uniform. This can reduce weight degeneracy but introduces shared
genealogy.

The same pre-normalization calculation estimates the adjacent normalizing-
constant ratio. With normalized incoming weights, the code computes

```text
ell_k = logsumexp_i(log W_(k-1)^i + log G_k(X_k^i)),
log Z_hat_k = log Z_hat_(k-1) + ell_k,
log W_k^i = log W_(k-1)^i + log G_k(X_k^i) - ell_k.
```

The initial value is `log Z_hat_0=0` because `lambda_0=0` and `p_M` is
normalized. The increment is recorded before any resampling at level `k`;
resampling then resets particle weights but does not alter the accumulated
normalizer estimate. This is exactly the weighting/resampling order implemented
in `run_suffix_direct_mh_smc`.

There is no final resampling. Keeping `W_K` preserves the information required
for final expectations and for correction back to the base model:

```text
E_(pi_lambda_K)[f(X)] approximately sum_i W_K^i f(X_K^i).
```

For a fixed number of levels, standard SMC consistency conditions include
correct initialization, invariant mutation kernels, correct incremental
weights, and a valid resampling scheme. They imply convergence of the weighted
empirical measure as `N` grows; they do not imply that a particular finite
population mixes or has low variance. Del Moral, Doucet, and Jasra (2006)
provide the methodological SMC context for this sequence-of-targets view.

## 5. Importance sampling back to p_M

For one final tilt, suppressing the prompt in notation,

```text
pi_lambda(x) = p_M(x) exp[-lambda phi(x)] / Z_lambda.
```

Rearrange this identity:

```text
p_M(x) = pi_lambda(x) Z_lambda exp[lambda phi(x)].
```

For any integrable `f`,

```text
E_pM[f]
  = Z_lambda E_pi[f(X) exp(lambda phi(X))].
```

Putting `f=1` gives

```text
1 = Z_lambda E_pi[exp(lambda phi(X))],
```

so division eliminates the unknown normalizer:

```text
E_pM[f]
  = E_pi[f(X) exp(lambda phi(X))]
    / E_pi[exp(lambda phi(X))].
```

When the final SMC approximation has normalized weights `W_i`, define

```text
log r_i = log W_i + lambda phi_i,
R_i     = exp(log r_i - logsumexp_j(log r_j)).
```

The implemented self-normalized estimate is

```text
E_hat_pM[f] = sum_i R_i f(X_i).
```

For a tail threshold `a`, set `f_a(x)=1{phi(x)>=a}`. The estimate is then the
sum of corrected weights for particles in the event.

SNIS is consistent under the usual support and moment conditions, but the ratio
of two random estimates is generally biased at finite `N`. The relevant overlap
diagnostic is

```text
corrected ESS = 1 / sum_i R_i^2.
```

This is different from final SMC ESS, `1/sum_i W_i^2`. Healthy SMC ESS means the
population represents the tilted target reasonably evenly. Healthy corrected
ESS means those particles also represent the base model after multiplying by
`exp(lambda phi)`. A strong negative tilt can achieve the first while nearly all
corrected mass is carried by a few lower-ARI particles, causing the second to
collapse.

## 6. Mixtures and multiple tilted distributions

The repository implements single-component SNIS (**A**), independent terminal
lambda studies (**B**), and the normalized deterministic-mixture validation in
`run_tail_validation.py` (**C**). The unnormalized construction in **D** is not
used by the active validation pipeline.

### A. One tilted distribution with SNIS

Samples from one `pi_lambda` are corrected with `exp(lambda phi)`. As shown
above, `Z_lambda` cancels after self-normalization.

### B. Separate SNIS estimates at different lambdas

Each final lambda can produce its own SNIS estimate, and each such estimate does
not require its own `Z_lambda`. Comparing or summarizing independent estimates
is not automatically multiple importance sampling and is not described as a
mixture in this repository.

### C. Implemented normalized deterministic mixture

Component `j` uses its own cap and independent full SMC run:

```text
psi_j(x) = min(raw_ari(x), c_j),
q_j(x)   = p_M(x) exp[-lambda_j psi_j(x)] / Z_j.
```

Runs do not share particles, ancestry, or RNG state. An intermediate population
from a schedule ending at another lambda is not a component. For prescribed
positive coefficients `rho_j` summing to one,

For prescribed coefficients `rho_k`, a normalized mixture is

```text
q_mix(x) = sum_j rho_j q_j(x)
         = p_M(x) sum_j rho_j exp[-lambda_j psi_j(x)] / Z_j.
```

Therefore

```text
p_M(x)/q_mix(x)
  = 1 / sum_j rho_j exp[-lambda_j psi_j(x)] / Z_j.
```

The different `Z_j` occur inside a sum, so one common self-normalization cannot
remove them. The implementation inserts the independently estimated
`Z_hat_j` from each SMC run and evaluates this denominator with `logsumexp`.

The final SMC population for component `j` is itself weighted. Its empirical
mass is `W_ji`, not one unweighted count. For sample `x_ji`, the implemented
unnormalized deterministic-mixture contribution is

```text
log d_ji = log rho_j + log W_ji
           - logsumexp_l(
               log rho_l - lambda_l psi_l(x_ji) - log Z_hat_l
             ).
```

After normalizing all `d_ji` together, the tail estimate is

```text
p_hat_a = sum_(j,i) D_ji 1{raw_ari(x_ji) >= a}.
```

Thus the cap affects only component construction and the mixture denominator;
the event always uses raw ARI. When component particles are equally weighted
and `rho_j=n_j/sum_l n_l`, this reduces to the usual balance-heuristic sample
weight. With nonuniform final SMC weights, the `W_ji` factor preserves the
weighted empirical approximation instead of treating particles as independent
unweighted draws.

Because `Z_hat_j` and the component empirical measures are random, this is a
plug-in self-normalized estimator. It is not exactly unbiased at finite particle
count. Independent seed groups, rather than particles within one run, are the
replication unit. Veach and Guibas (1995) supplies multiple-importance-sampling
context; the formula above is the direct derivation used by this code.

The tracked official direct array is an empirical reference for the paper's
observable, not analytic ground truth. Its model, tokenizer, prompt, generation
length, and temperature match this project. Strict observable equivalence is
unverified because neither the paper's custom ARI edge-case code nor the source
completions/token IDs are published, and the public notebook uses a different
decode/slicing convention. `run_tail_validation.py` checks the published setup
before sampling and requires an explicit opt-in to report a cross-implementation
comparison when this observable limitation remains.

### D. An unnormalized mixture potential

One may instead define

```text
H(x)   = sum_k rho_k exp[-lambda_k phi(x)],
q_H(x) proportional to p_M(x) H(x).
```

Correction to `p_M` is then proportional to `1/H(x)`, whose one global
normalizer disappears under SNIS. But `q_H` is generally not the normalized
mixture in C with the same prescribed `rho_k`; its effective component weights
depend on the `Z_k`. This is a distinct target and remains future work here.

## 7. Small examples

### 7.1 Four-state MH stationarity

Let completions be `{00,01,10,11}`, with

```text
p_M = (0.4, 0.1, 0.2, 0.3),
phi = (0,   1,   2,   3),
lambda = -0.5.
```

Take `m=T=2`, so the suffix proposal is an independent draw from `p_M`. The
direct acceptance is `min(1,exp(0.5[phi(y)-phi(x)]))`. For every pair,

```text
pi(x)p_M(y)alpha(x,y)
  = min{pi(x)p_M(y), pi(y)p_M(x)}
  = pi(y)p_M(x)alpha(y,x).
```

For example, between `00` and `11`, the forward acceptance is `1`, while the
reverse acceptance is `exp(-1.5)`. Since

```text
pi(11)/pi(00) = [0.3 exp(1.5)] / 0.4,
```

the two probability flows are equal. Pairwise equality plus rejection on the
diagonal proves that the normalized vector proportional to
`(0.4, 0.1e^0.5, 0.2e^1, 0.3e^1.5)` is stationary.

### 7.2 Incremental SMC weights

Suppose the current normalized weights are `(0.5,0.5)`, the particle
observables are `(2,6)`, and the schedule moves from `0` to `-0.25`. The
unnormalized multipliers are

```text
exp(0.25*2) = 1.6487,
exp(0.25*6) = 4.4817.
```

After normalization, weights are approximately `(0.269,0.731)`. This is the
selection toward larger `phi` created by the target change, not by the mutation
kernel itself.

### 7.3 SNIS cancellation of Z

Suppose two weighted tilted particles have `W=(0.4,0.6)`, `phi=(2,6)`, and
`lambda=-0.25`. Their base-model correction is proportional to

```text
(0.4 exp(-0.5), 0.6 exp(-1.5)) = (0.2426, 0.1339).
```

Normalization gives `R approximately (0.6446,0.3554)`. No numerical value of
`Z_lambda` appears because the same factor would multiply both terms and cancel.

### 7.4 Corrected ESS collapse

With equal final SMC weights, `lambda=-2`, and `phi=(0,1,2,3)`, correction back
to the base model is proportional to

```text
exp(lambda phi) = (1, 0.1353, 0.0183, 0.0025).
```

The normalized corrected weights are approximately
`(0.865,0.117,0.016,0.002)`, giving corrected ESS about `1.31` out of `4`, even
though the original SMC ESS is exactly `4`. The scale
`|lambda| * (max(phi)-min(phi))` controls this instability.

### 7.5 Why a normalized mixture needs Z_k

Take two tilts with equal mixture coefficients and one state with `phi=1`:

```text
q(x)/p_M(x) = 0.5 exp(-lambda_1)/Z_1
              + 0.5 exp(-lambda_2)/Z_2.
```

Multiplying both `Z_1` and `Z_2` by the same constant would cancel under final
self-normalization, but their unknown ratio changes the sum and therefore
changes every importance weight. Separate single-tilt cancellations cannot
recover that ratio.

## 8. Guarantees and limitations

- **Mutation stationarity.** Under the conditions in Section 3, each direct-MH
  suffix kernel satisfies detailed balance for its stated `pi_lambda`.
- **Asymptotic SMC correctness.** With valid mutation, weighting, and
  resampling, the weighted particle measure is consistent as particle count
  grows under standard regularity conditions.
- **Asymptotic SNIS correctness.** The correction is consistent when proposal
  support covers the integrand and required moments exist.
- **Finite-sample bias.** SNIS is a ratio estimator and is generally biased for
  finite particle counts.
- **MCMC autocorrelation.** Repeated suffix states are correlated. Invariance
  does not imply rapid convergence from the current population.
- **Genealogical dependence.** Resampling creates duplicate ancestors, so final
  particles are not independent observations.
- **Between-run uncertainty.** Independent seeds are needed to assess estimator
  variability; one large population is not a substitute for replicated runs.
- **No a priori efficiency guarantee.** Performance depends on observable
  geometry, LM support, suffix-size law, lambda schedule, mutation count, and
  particle count.
- **Overlap is essential.** Discovery of extreme completions and reliable
  correction to `p_M` can require different terminal tilts. Report both final
  SMC ESS and corrected importance ESS.

## References and their role

1. Jake McAllister Dorman, Edward Gillman, Dominic C. Rose, Jamie F. Mair, and
   Juan P. Garrahan. "Rare Event Analysis of Large Language Models." arXiv
   2602.06791, 2026; ICML 2026 Oral Spotlight.
   [doi:10.48550/arXiv.2602.06791](https://doi.org/10.48550/arXiv.2602.06791).
   Starting point for the rare-event question, exponential tilt, and
   TPS/annealing workflow.
2. Giacomo Zanella. "Informed Proposals for Local MCMC in Discrete Spaces."
   *Journal of the American Statistical Association* 115, no. 530 (2020):
   852-865. [doi:10.1080/01621459.2019.1585255](https://doi.org/10.1080/01621459.2019.1585255).
   Inspiration for earlier local informed-proposal and balancing-function
   experiments; not the correctness proof for the active suffix kernel.
3. Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. "Sequential Monte Carlo
   Samplers." *Journal of the Royal Statistical Society: Series B* 68, no. 3
   (2006): 411-436.
   [doi:10.1111/j.1467-9868.2006.00553.x](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
   Methodological context and standard terminology for SMC over a sequence of
   unnormalized targets.
4. John Geweke. "Bayesian Inference in Econometric Models Using Monte Carlo
   Integration." *Econometrica* 57, no. 6 (1989): 1317-1339.
   [doi:10.2307/1913710](https://doi.org/10.2307/1913710). Importance-sampling
   context, including normalized weighted Monte Carlo estimators.
5. Eric Veach and Leonidas J. Guibas. "Optimally Combining Sampling Techniques
   for Monte Carlo Rendering." In *Proceedings of SIGGRAPH '95* (1995):
   419-428. [doi:10.1145/218380.218498](https://doi.org/10.1145/218380.218498).
   Primary context for multiple importance sampling and the balance heuristic;
   the active repository does not implement that mixture estimator.
6. W. K. Hastings. "Monte Carlo Sampling Methods Using Markov Chains and Their
   Applications." *Biometrika* 57, no. 1 (1970): 97-109.
   [doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97).
   Historical source for general Metropolis-Hastings terminology; the exact
   acceptance used here is derived directly in Section 3.
