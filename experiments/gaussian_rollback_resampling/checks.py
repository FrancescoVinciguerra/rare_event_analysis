"""Assertions embedded in the notebook; no local imports are required there."""
import numpy as np
from scipy.special import logsumexp


def run_checks():
    checks = []

    def passed(name):
        checks.append(name)
        print("PASS:", name)

    def close_mc(values, expected, tolerance=7):
        values = np.asarray(values)
        error = values.std(ddof=1)/np.sqrt(values.size)
        assert abs(values.mean()-expected) <= tolerance*error + 1e-12

    rng = np.random.default_rng(71001)
    paths = base_paths(rng, 150_000, 16)
    close_mc(paths[:, -1], 0)
    close_mc(paths[:, -1]**2, 1)
    close_mc((paths[:, -1] >= THRESHOLD).astype(float), P_EVENT)
    assert np.all(paths[:, 0] == 0)
    passed("original terminal law N(0,1)")

    proposed, depths = rollback_proposal(rng, paths[:30_000], 16)
    for d in range(1, 17):
        rows = depths == d
        np.testing.assert_array_equal(proposed[rows, :17-d], paths[:30_000][rows, :17-d])
        close_mc(rows.astype(float), 1/16)
    passed("uniform depth and complete conserved prefix")

    beta = 2.0
    # Exact tilted draws are solely an invariance TEST, not the SMC sampler.
    stationary = base_paths(rng, 100_000, 16) + beta*np.arange(17)[None, :]/16
    for _ in range(3):
        rollback_mh(rng, stationary, beta, 16)
    for j in (4, 8, 16):
        centered = stationary[:, j]-beta*j/16
        close_mc(centered, 0)
        close_mc(centered**2, j/16)
    close_mc((stationary[:, 4]-beta/4)*(stationary[:, -1]-beta), .25)
    passed("MH invariance of path moments and covariance")

    cfg = Config(tilt=2.0)
    seed = 71002
    disabled = run_smc(np.random.default_rng(seed), 2048, cfg, resampling=False)
    baseline = run_independent_baseline(np.random.default_rng(seed), 2048, cfg)
    np.testing.assert_array_equal(disabled.paths, baseline.paths)
    np.testing.assert_allclose(disabled.log_weights, baseline.log_weights, atol=1e-13)
    np.testing.assert_allclose(disabled.log_z, baseline.log_z, atol=1e-13)
    for key, value in baseline.estimates(cfg.tilt).items():
        np.testing.assert_allclose(disabled.estimates(cfg.tilt)[key], value, atol=1e-13)
    assert not any(s["resampled"] for s in disabled.levels)
    passed("disabled branch exactly reproduces independent path-weight baseline")

    zero = run_smc(np.random.default_rng(71003), 4096, Config(tilt=0))
    assert not any(s["resampled"] for s in zero.levels)
    np.testing.assert_allclose(zero.estimates(0)["z_hat"], 1)
    mc = np.mean(zero.paths[:, -1] >= THRESHOLD)
    np.testing.assert_allclose(zero.estimates(0)["ordinary"], mc)
    np.testing.assert_allclose(zero.estimates(0)["snis"], mc)
    passed("ESS policy genuinely skips resampling at zero tilt")

    # Deterministic concentrated population guarantees ESS/N < 0.5.
    initial = np.zeros((4, 3))
    initial[3] = [0, 5, 10]
    branch = run_smc(np.random.default_rng(71004), 4,
                     Config(steps=2, levels=3, tilt=1, max_depth=2, sweeps=0),
                     initial_paths=initial, trace=True)
    assert branch.levels[0]["ess_before"] < 2
    assert branch.levels[0]["resampled"]
    assert any(not row["resampled"] for row in branch.levels)
    for row, tr in zip(branch.levels, branch.trace):
        if row["resampled"]:
            np.testing.assert_array_equal(tr["after_selection"], tr["before"][tr["parents"]])
            np.testing.assert_allclose(tr["log_w"], -np.log(4))
            np.testing.assert_allclose(row["ess_after"], 4)
    passed("active ESS branch copies full paths and resets only particle weights")

    for row, tr in zip(branch.levels, branch.trace):
        expected_increment = logsumexp(tr["old_log_w"] + (1/3)*tr["before"][:, -1])
        np.testing.assert_allclose(row["log_c"], expected_increment)
        np.testing.assert_allclose(row["log_z"], tr["old_log_z"]+expected_increment)
    np.testing.assert_allclose(branch.log_z, sum(r["log_c"] for r in branch.levels))
    nonresampling_increments = [r["log_c"] for r in branch.levels if not r["resampled"]]
    assert any(abs(x) > .1 for x in nonresampling_increments)
    passed("Z_hat includes non-resampling levels and is never reset")

    final = run_smc(np.random.default_rng(71005), 4,
                   Config(steps=2, levels=1, tilt=1, max_depth=2, sweeps=0),
                   initial_paths=initial, trace=True)
    assert final.levels[0]["ess_before"] < 2
    assert not final.levels[0]["resampled"]
    assert np.exp(final.log_weights).max() > .99
    np.testing.assert_allclose(np.exp(final.log_z), np.exp(initial[:, -1]).mean())
    passed("final resampling excluded even when final ESS is below threshold")

    moving = run_smc(np.random.default_rng(71006), 128, Config(tilt=3), trace=True)
    old_beta = 0
    for row, tr in zip(moving.levels, moving.trace):
        increment = logsumexp(tr["old_log_w"]+(row["beta"]-old_beta)*tr["before"][:, -1])
        np.testing.assert_allclose(increment, row["log_c"])
        old_beta = row["beta"]
    assert not np.array_equal(moving.trace[-1]["after_selection"], moving.paths)
    assert moving.transitions == 128*16 + sum(s["transitions_stage"] for s in moving.levels)
    assert 128*8 <= moving.transitions-128*16 <= 128*8*16
    assert moving.proposals == 128*8
    passed("weight-before-move order, final mutation and exact Gaussian cost")

    weights = np.exp(moving.log_weights)
    inverse = np.exp(-3*moving.paths[:, -1])
    h = moving.paths[:, -1] >= THRESHOLD
    estimates = moving.estimates(3)
    np.testing.assert_allclose(estimates["ordinary"], np.exp(moving.log_z)*np.sum(weights*inverse*h))
    np.testing.assert_allclose(estimates["snis"], np.sum(weights*inverse*h)/np.sum(weights*inverse))
    np.testing.assert_allclose(estimates["reverse_mass"], np.exp(moving.log_z)*np.sum(weights*inverse))
    passed("both requested estimators use normalized terminal weights and Z_hat")

    class FixedOffset:
        def __init__(self, offset):
            self.offset = offset
        def random(self):
            return self.offset

    # Integrate exactly over offset intervals; counts constant within intervals.
    weights = np.array([.05, .15, .3, .5])
    n = len(weights)
    knots = np.unique(np.r_[0, np.mod(n*np.cumsum(weights), 1), 1])
    counts, probabilities = [], []
    for left, right in zip(knots[:-1], knots[1:]):
        parents = systematic_resample(FixedOffset((left+right)/2), weights)
        counts.append(np.bincount(parents, minlength=n))
        probabilities.append(right-left)
    counts, probabilities = np.array(counts), np.array(probabilities)
    np.testing.assert_allclose(probabilities@counts, n*weights, atol=1e-14)
    centered = counts-n*weights
    covariance = (centered.T*probabilities)@centered
    multinomial_covariance = n*(np.diag(weights)-np.outer(weights, weights))
    assert not np.allclose(covariance, multinomial_covariance)
    # Conditional variance formula for resampling followed by independent moves.
    means, variances = np.array([1., -2., 4., 0.]), np.array([.1, .2, .3, .4])
    total_var_direct = probabilities@((counts@variances)/n**2 + (counts@means/n-weights@means)**2)
    total_var_formula = weights@variances/n + means@covariance@means/n**2
    np.testing.assert_allclose(total_var_direct, total_var_formula)
    passed("systematic offspring unbiasedness and correct conditional covariance")

    estimates, active_runs = [], 0
    cfg = Config(steps=4, levels=4, tilt=2, max_depth=4)
    for rep in range(500):
        result = run_smc(np.random.default_rng(np.random.SeedSequence([71007, rep])), 512, cfg)
        estimates.append(result.estimates(cfg.tilt))
        active_runs += any(r["resampled"] for r in result.levels)
    assert active_runs > 100
    close_mc([r["ordinary"] for r in estimates], P_EVENT)
    close_mc([r["reverse_mass"] for r in estimates], 1)
    close_mc([r["z_hat"] for r in estimates], np.exp(2))
    passed("whole-population identities E[p_ord]=p, E[reverse_mass]=1, E[Z_hat]=Z with active resampling")
    return checks


if __name__ == "__main__":
    from smc_resampling import *
    run_checks()
