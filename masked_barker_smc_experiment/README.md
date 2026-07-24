# Masked Barker full-sequence SMC

This pipeline is independent of the repository's earlier full-sequence SMC
and mixture-MCMC experiments. Its particle state is a fixed-length complete
completion \(x=x_{1:T}\), and its target at level \(k\) is

\[
\gamma_k(x)=p_M(x\mid c)\exp\{-\lambda_k\phi(x)\}.
\]

It does not evaluate the unknown normalizing constants \(Z_k\).

## Mutation kernel

For each particle and MCMC step, a mask \(S\) is drawn independently of the
completion text. Conditional on \(S\), the LLM resamples the selected tokens
from left to right and gives an exact proposal probability \(R_S(x,y)\).
The reverse probability \(R_S(y,x)\) is evaluated with the same mask and the
same procedure.

The first acceptance factor is

\[
a_{\mathrm{sym}}(x,y;S)
=1\wedge\frac{R_S(y,x)}{R_S(x,y)}.
\]

Therefore the off-diagonal proposal kernel is

\[
K_S(x,y)=R_S(x,y)a_{\mathrm{sym}}(x,y;S)
=\min\{R_S(x,y),R_S(y,x)\}=K_S(y,x).
\]

A state-independent mixture over mask sizes and masks is still symmetric,
because it is a fixed mixture of symmetric \(K_S\) kernels. Explicit and
deterministic masks are also valid: each MCMC step uses one fixed symmetric
kernel.

For the current target, define

\[
\Delta_k(x,y)=\log p_M(y\mid c)-\log p_M(x\mid c)
-\lambda_k[\phi(y)-\phi(x)].
\]

The Barker factor is \(g(e^{\Delta_k})=\sigma(\Delta_k)\), where
\(g(t)=t/(1+t)\) satisfies \(g(t)=t\,g(1/t)\). The total acceptance
probability is \(a_{\mathrm{sym}}\sigma(\Delta_k)\), evaluated with one
uniform draw. Thus, off the diagonal,

\[
M_k(x,y)=K(x,y)g\!\left(\frac{\pi_k(y)}{\pi_k(x)}\right)
\]

and

\[
\pi_k(x)M_k(x,y)=\pi_k(y)M_k(y,x).
\]

The mutation kernel leaves \(\pi_k\) invariant, and a composition of multiple
such MCMC steps remains invariant. This implementation directly uses the
symmetric \(K\) and Barker factor; it does not construct a normalized
\(Q_g\), does not evaluate \(Z_g(x)\), and does not claim the
Peskun-optimality result proved for that different normalized proposal.

## SMC update

At transition \(k\), mutation targets \(\pi_{k-1}\). The incremental update is

\[
\log w_k^{(i)}=\log w_{k-1}^{(i)}
-(\lambda_k-\lambda_{k-1})\phi(X_k^{(i)}).
\]

Weights are normalized with `torch.logsumexp`. If ESS is below the configured
fraction of \(N\), systematic resampling is applied and weights are reset to
\(1/N\). Systematic resampling is unbiased, so it preserves the particle
approximation in expectation.

These statements establish correctness, stationarity, and the usual SMC
consistency conditions. They do not imply a high acceptance rate, rapid
mixing, or guaranteed rare-event discovery.

## Run

```bash
python masked_barker_smc_experiment/run_experiment.py \
  --device cuda \
  --num-particles 64 \
  --max-new-tokens 100 \
  --lambdas 0,-0.25,-0.5,-1.0 \
  --mcmc-steps-per-level 5 \
  --ess-threshold 0.5 \
  --mask-mode stochastic \
  --mask-sizes 1,2,4,8 \
  --mask-rho-schedule 0.4,0.3,0.2,0.1 \
  --mask-policy uniform_noncontiguous \
  --proposal-temperature 1.0 \
  --batch-size 32 \
  --output-dir outputs/masked_barker_smc
```

Rows in `--mask-rho-schedule` can be separated by semicolons to define one
row per SMC transition. Deterministic schedules use the same convention in
`--deterministic-mask-size-schedule`. Explicit masks are JSON with zero-based
completion positions:

```text
--mask-mode deterministic --mask-policy explicit \
--explicit-masks '[[[0,1],[2,3]],[[4],[5]]]'
```

The exact `textstat` ARI requires decoded text. Consequently, each batched
proposal step has one explicit device-to-CPU boundary for ARI evaluation.
All model scoring, mask operations, proposal probabilities, acceptance
probabilities, weights, ESS, and resampling remain on the model device.
