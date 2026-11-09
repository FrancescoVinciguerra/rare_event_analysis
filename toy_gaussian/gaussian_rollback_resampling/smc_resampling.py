"""Gaussian full-path RollBack SMC: paper-faithful ESS/systematic policy.

This file is also embedded verbatim in the standalone Colab notebook.
Arrays store [x_T, ..., x_0]. No language model and no direct tilted sampling.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

P_EVENT = 0.001
THRESHOLD = float(norm.isf(P_EVENT))


@dataclass(frozen=True)
class Config:
    steps: int = 16
    levels: int = 8
    tilt: float = 2.0
    max_depth: int = 16
    sweeps: int = 1
    ess_threshold: float = 0.5

    def __post_init__(self):
        if self.steps < 1 or self.levels < 1 or self.sweeps < 0:
            raise ValueError("Invalid steps, levels or sweeps")
        if not 1 <= self.max_depth <= self.steps or not np.isfinite(self.tilt) or self.tilt < 0:
            raise ValueError("Invalid rollback depth or tilt")
        if not 0 <= self.ess_threshold <= 1:
            raise ValueError("ESS threshold must be in [0,1]")


def base_paths(rng, n, steps):
    paths = np.zeros((n, steps + 1))
    paths[:, 1:] = np.cumsum(rng.normal(0, 1 / np.sqrt(steps), (n, steps)), axis=1)
    return paths


def rollback_proposal(rng, paths, max_depth):
    n, columns = paths.shape
    steps = columns - 1
    depths = rng.integers(1, max_depth + 1, n)
    proposed = paths.copy()
    for depth in range(1, max_depth + 1):
        rows = np.flatnonzero(depths == depth)
        start = steps - depth
        noise = rng.normal(0, 1 / np.sqrt(steps), (len(rows), depth))
        proposed[rows, start + 1:] = paths[rows, start, None] + np.cumsum(noise, axis=1)
    return proposed, depths


def rollback_mh(rng, paths, beta, max_depth):
    proposed, depths = rollback_proposal(rng, paths, max_depth)
    log_alpha = np.minimum(0.0, beta * (proposed[:, -1] - paths[:, -1]))
    accept = np.log(rng.random(len(paths))) < log_alpha
    paths[accept] = proposed[accept]
    return int(accept.sum()), int(depths.sum())


def systematic_resample(rng, normalized_weights):
    """One common uniform offset; preserve input order, as in paper code.

    The offspring are NOT independent categorical draws.
    """
    weights = np.asarray(normalized_weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("Invalid weights")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be normalized")
    n = len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    points = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cumulative, points, side="left")


@dataclass
class SMCResult:
    paths: np.ndarray
    log_weights: np.ndarray  # normalized terminal weights
    log_z: float
    levels: list
    transitions: int
    accepted: int
    proposals: int
    roots: np.ndarray
    trace: list | None = None

    def estimates(self, tilt):
        reverse_log_weights = self.log_weights - tilt * self.paths[:, -1]
        event = self.paths[:, -1] >= THRESHOLD
        log_num = logsumexp(reverse_log_weights[event]) if event.any() else -np.inf
        log_den = logsumexp(reverse_log_weights)
        return {"ordinary": float(np.exp(self.log_z + log_num)),
                "snis": float(np.exp(log_num - log_den)),
                "reverse_mass": float(np.exp(self.log_z + log_den)),
                "z_hat": float(np.exp(self.log_z))}


def run_smc(rng, n, config, resampling=True, trace=False, initial_paths=None):
    """Order is weighting -> conditional systematic resampling -> MH.

    The normalizer is updated at EVERY level, not only at resampling times.
    Initial paths can be supplied for deterministic branch tests only.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if initial_paths is None:
        paths = base_paths(rng, n, config.steps)
        transitions = n * config.steps
    else:
        paths = np.array(initial_paths, dtype=float, copy=True)
        if paths.shape != (n, config.steps + 1) or not np.all(paths[:, 0] == 0):
            raise ValueError("Invalid initial trajectories")
        transitions = 0
    log_w = np.full(n, -np.log(n))
    log_z = 0.0
    roots = np.arange(n)
    accepted = proposals = 0
    history, full_trace = [], [] if trace else None
    schedule = np.linspace(0.0, config.tilt, config.levels + 1)
    for k, (old_beta, beta) in enumerate(zip(schedule[:-1], schedule[1:]), start=1):
        if trace:
            before = paths.copy()
            old_log_w, old_log_z = log_w.copy(), log_z
        # PAPER STEP 1: weight BEFORE mutation. log_w is already normalized.
        unnormalized = log_w + (beta - old_beta) * paths[:, -1]
        log_c = float(logsumexp(unnormalized))
        log_z += log_c  # EVERY LEVEL, regardless of the resampling decision.
        log_w = unnormalized - log_c
        ess_before = float(np.exp(-logsumexp(2 * log_w)))
        # PAPER STEP 2: strict ESS/N < 0.5; explicitly EXCLUDE final level.
        do_resample = bool(resampling and k < config.levels and ess_before < config.ess_threshold * n)
        parents = None
        if do_resample:
            parents = systematic_resample(rng, np.exp(log_w))
            paths = paths[parents].copy()  # copy COMPLETE stored trajectories
            roots = roots[parents].copy()
            log_w.fill(-np.log(n))
            # Do NOT reset log_z here.
        ess_after = float(np.exp(-logsumexp(2 * log_w)))
        if trace:
            after_selection = paths.copy()
        # PAPER STEP 3: mutate at beta_k at EVERY level, including final.
        stage_cost = stage_accepted = 0
        for _ in range(config.sweeps):
            moves, cost = rollback_mh(rng, paths, beta, config.max_depth)
            stage_accepted += moves
            stage_cost += cost
            proposals += n
        accepted += stage_accepted
        transitions += stage_cost
        history.append({"level": k, "beta": float(beta), "ess_before": ess_before,
                        "ess_after": ess_after, "resampled": do_resample,
                        "log_c": log_c, "log_z": float(log_z), "z_hat": float(np.exp(log_z)),
                        "transitions_stage": stage_cost, "transitions_cumulative": transitions,
                        "accepted": stage_accepted, "distinct_roots": int(np.unique(roots).size)})
        if trace:
            full_trace.append({"before": before, "old_log_w": old_log_w, "old_log_z": old_log_z,
                               "parents": parents, "after_selection": after_selection,
                               "after_mutation": paths.copy(), "log_w": log_w.copy()})
    return SMCResult(paths, log_w, log_z, history, transitions, accepted, proposals, roots, full_trace)


def run_independent_baseline(rng, n, config):
    """Original no-resampling algorithm, implemented through path weights.

    Kept independent of run_smc to check the two normalizer calculations.
    Same base-path and MH random draws as the original delivered baseline.
    """
    paths = base_paths(rng, n, config.steps)
    log_raw = np.zeros(n)
    transitions = n * config.steps
    accepted = proposals = 0
    history = []
    old_log_z = 0.0
    schedule = np.linspace(0.0, config.tilt, config.levels + 1)
    for k, (old_beta, beta) in enumerate(zip(schedule[:-1], schedule[1:]), start=1):
        log_raw += (beta-old_beta) * paths[:, -1]
        log_z = float(logsumexp(log_raw)-np.log(n))
        log_w = log_raw-logsumexp(log_raw)
        ess = float(np.exp(-logsumexp(2*log_w)))
        stage_cost = stage_accepted = 0
        for _ in range(config.sweeps):
            moves, cost = rollback_mh(rng, paths, beta, config.max_depth)
            stage_accepted += moves
            stage_cost += cost
            proposals += n
        accepted += stage_accepted
        transitions += stage_cost
        history.append({"level": k, "beta": float(beta), "ess_before": ess, "ess_after": ess,
                        "resampled": False, "log_c": log_z-old_log_z, "log_z": log_z,
                        "z_hat": float(np.exp(log_z)), "transitions_stage": stage_cost,
                        "transitions_cumulative": transitions, "accepted": stage_accepted,
                        "distinct_roots": n})
        old_log_z = log_z
    return SMCResult(paths, log_w, log_z, history, transitions, accepted, proposals, np.arange(n))
