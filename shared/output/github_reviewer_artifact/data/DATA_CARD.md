# Data card for the reviewer artifact

## Included

- `final_analysis/run_estimates.csv`: one aggregate estimate per independent
  method, particle budget, cohort, run, and event threshold.
- `final_analysis/summary.csv`: bias, dispersion, RMSE, and measured cost
  summaries.
- `final_analysis/bootstrap_comparisons.csv`: run-level bootstrap comparisons.
- `final_analysis/direct_reference.csv`: IID reference estimates and Wilson
  intervals from 200,000 samples.
- `final_analysis/normalizer_and_genealogy.csv`: normalizer, ESS, resampling,
  and ancestry diagnostics.
- `components/component_estimates_all_thresholds.csv`: separately normalized
  reverse-SNIS estimates for each lambda component and threshold.
- `components/component_metrics.csv`: component-level weight, genealogy,
  terminal-diversity, acceptance, and reward diagnostics.

## Provenance

The final confirmatory computation completed as job 2167995. The released
manifest records the frozen decoder/evaluator fingerprints and input study
identities. Submission and confirmatory cohorts use disjoint seeds. Randomness
comes from model categorical draws, resampling, and (for Trajectory-MH)
rollback proposals and acceptance decisions.

## Not included

Raw model completions, token arrays, checkpoints, caches, scheduler logs, and
cluster paths are omitted to keep the artifact small and anonymous. The
qualitative audit table is included because it supports inspection of the
classifier-defined event, but it is not used to recompute the estimators.

## Limitations

The toxicity observable is a frozen RoBERTa classifier score. The 0.999 event
has limited direct-reference discoveries and is exploratory. Direct sampling
provides a high-precision Monte Carlo reference, not an exact probability.
