"""Targeted mathematical and implementation checks, fixed seeds."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from model import Config, P_EVENT, THRESHOLD, base_paths, rollback_proposal, rollback_mh, run_smc


def within_mc_error(values, expected, sigma=7):
    assert abs(np.mean(values) - expected) < sigma * np.std(values, ddof=1) / np.sqrt(len(values)) + 1e-12


def test_original_terminal_marginal():
    paths = base_paths(np.random.default_rng(1701), 200_000, 16)
    assert np.all(paths[:, 0] == 0)
    within_mc_error(paths[:, -1], 0)
    within_mc_error(paths[:, -1] ** 2, 1)
    within_mc_error((paths[:, -1] >= THRESHOLD).astype(float), P_EVENT)
    within_mc_error((paths[:, -1] <= norm.ppf(0.1)).astype(float), 0.1)
    within_mc_error((paths[:, -1] <= norm.ppf(0.9)).astype(float), 0.9)


def test_proposal_preserves_full_prefix_and_uniform_depth():
    rng = np.random.default_rng(1702)
    paths = base_paths(rng, 50_000, 16)
    proposed, depths = rollback_proposal(rng, paths, 16)
    for d in range(1, 17):
        rows = depths == d
        assert np.array_equal(proposed[rows, :17-d], paths[rows, :17-d])
        within_mc_error(rows.astype(float), 1 / 16)
    within_mc_error(proposed[:, -1], 0)
    within_mc_error(proposed[:, -1] ** 2, 1)


def test_fixed_depth_mh_density_cancellation():
    rng = np.random.default_rng(1703)
    paths = base_paths(rng, 1000, 16)
    prop, depth = rollback_proposal(rng, paths, 16)
    log_p = lambda p: -0.5 * 16 * np.square(np.diff(p, axis=1)).sum(axis=1)
    beta = 2.0
    forward = np.zeros(len(paths))
    reverse = np.zeros(len(paths))
    for d in range(1, 17):
        rows = depth == d
        forward[rows] = log_p(prop[rows, -d-1:])
        reverse[rows] = log_p(paths[rows, -d-1:])
    ratio = log_p(prop) + beta * prop[:, -1] - log_p(paths) - beta * paths[:, -1] + reverse - forward
    np.testing.assert_allclose(ratio, beta * (prop[:, -1] - paths[:, -1]), atol=1e-12)


def test_stationary_path_distribution_is_preserved():
    # Exact tilted initialization is used ONLY to test invariance, never in SMC.
    rng = np.random.default_rng(1704)
    n, steps, beta = 150_000, 16, 2.0
    for max_depth in (1, 4, 16):
        paths = base_paths(rng, n, steps)
        paths += beta * np.arange(steps + 1)[None, :] / steps
        for _ in range(3):
            rollback_mh(rng, paths, beta, max_depth)
        for j in (1, 4, 8, 16):
            centered = paths[:, j] - beta * j / steps
            within_mc_error(centered, 0)
            within_mc_error(centered**2, j / steps)
        within_mc_error((paths[:, 4] - beta/4) * (paths[:, 16] - beta), 0.25)


def test_weights_use_pre_mutation_states_and_normalizer_telescopes():
    cfg = Config(tilt=2.0)
    result = run_smc(np.random.default_rng(1705), 2048, cfg, trace=True)
    np.testing.assert_allclose(result.log_weight, np.sum(result.trace["increments"], axis=0))
    np.testing.assert_allclose(result.log_weight, cfg.tilt / cfg.levels * np.sum(result.trace["before"], axis=0))
    np.testing.assert_allclose(sum(result.trace["log_z_increments"]), logsumexp(result.log_weight) - np.log(2048))
    assert np.max(np.abs(result.log_weight - cfg.tilt * result.paths[:, -1])) > 0.1


def test_no_mutation_and_zero_tilt_reduce_to_crude_mc():
    for cfg in (Config(tilt=3, sweeps=0), Config(tilt=0)):
        result = run_smc(np.random.default_rng(1706), 20_000, cfg, trace=True)
        np.testing.assert_allclose(result.log_correction, 0, atol=1e-14)
        mc = np.mean(result.paths[:, -1] >= THRESHOLD)
        for estimate in result.estimates().values():
            np.testing.assert_allclose(estimate, mc, atol=1e-15)


def test_weighted_feynman_kac_and_reverse_identities():
    result = run_smc(np.random.default_rng(1707), 200_000, Config(tilt=1.0, max_depth=16))
    w = np.exp(result.log_weight)
    x = result.paths[:, -1]
    z = np.exp(0.5)
    within_mc_error(w, z)
    within_mc_error(w * x, z)
    within_mc_error(w * x*x, 2*z)
    u, c = result.contributions()
    within_mc_error(c, 1)
    within_mc_error(c*x, 0)
    within_mc_error(c*x*x, 1)
    within_mc_error(u, P_EVENT)


def test_estimator_algebra_includes_all_normalizing_factors():
    result = run_smc(np.random.default_rng(1708), 10_000, Config(tilt=2.0))
    raw_w = np.exp(result.log_weight)
    normalized_w = raw_w / raw_w.sum()
    inverse_tilt = np.exp(-2 * result.paths[:, -1])
    h = result.paths[:, -1] >= THRESHOLD
    u, c = result.contributions()
    estimates = result.estimates()
    np.testing.assert_allclose(estimates["ordinary"], u.mean())
    np.testing.assert_allclose(estimates["ordinary"], raw_w.mean() * np.sum(normalized_w * inverse_tilt * h))
    np.testing.assert_allclose(estimates["snis"], u.sum()/c.sum())
    np.testing.assert_allclose(estimates["snis"], np.sum(normalized_w * inverse_tilt * h)/np.sum(normalized_w * inverse_tilt))
