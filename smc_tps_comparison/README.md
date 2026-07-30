# Raw-ARI SMC versus official annealed TPS

`run_comparison.py` is a minimal orchestrator. It calls the existing
`suffix_direct_mh_smc` implementation and the
official TPS adapter; it contains no sampler copy.

Both methods target the same final raw-ARI tilted distribution. SMC estimates
final expectations with normalized particle weights. TPS estimates them from
the correlated final-bias time series only. Weight ESS and temporal ESS are
reported separately because they measure different phenomena.

Every run completes its full schedule. Wall time is measured after completion
and is never a stopping rule. The aggregate report compares between-seed
stability and empirical estimator efficiency. Without an independent reference
summary, it does not report a false RMSE or claim ground-truth accuracy.

Maximum ARI, first threshold crossing, and distinct tail completions are
secondary discovery metrics over generated and evaluated proposals (including
rejected proposals) and the initial state. They are never substituted for the
weighted SMC population or final-bias TPS trajectory in target estimators.

```bash
python smc_tps_comparison/run_comparison.py \
  --algorithms suffix_direct_mh_smc,official_annealed_tps \
  --seeds 200,201,202 --device cuda --ari-mode raw \
  --experiment-label raw_ari_smc_vs_official_tps \
  --configuration-label selected --skip-completed
```

Run either method alone by passing only its algorithm name. Completed outputs
are reused only when their full SHA-256 configuration fingerprint matches,
including both repository commits. An incompatible run is never overwritten
without `--overwrite`.

## Correctness and convergence

TPS applies a direct MH kernel invariant for each tilted target and warm-starts
the next bias. Its samples remain autocorrelated. SMC applies invariant mutation
kernels, correct incremental weights, log-space normalization, and systematic
resampling below the ESS threshold, except at the final level. Under the usual
SMC assumptions, weighted estimates converge as particle count increases;
resampling controls weight degeneracy while introducing genealogy.

The notebook's tolerances are explicitly *operational convergence criteria*.
They assess stability across complete runs and independent seeds; they are not
a mathematical proof that either finite experiment has converged.
