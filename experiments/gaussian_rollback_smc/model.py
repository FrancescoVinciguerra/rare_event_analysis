"""Full-trajectory RollBack SMC without resampling.

Storage order: paths[:, j] = x_{T-j}; paths[:, 0] = x_T = 0.
No draws from a tilted distribution are used by run_smc.
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

    def __post_init__(self):
        if self.steps < 1 or self.levels < 1 or self.sweeps < 0:
            raise ValueError("Invalid step, level or sweep count")
        if not 1 <= self.max_depth <= self.steps or self.tilt < 0:
            raise ValueError("Invalid depth or tilt")


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


@dataclass
class Result:
    paths: np.ndarray
    log_weight: np.ndarray
    log_correction: np.ndarray
    accepted: int
    proposals: int
    transitions: int
    trace: dict | None = None

    def estimates(self):
        event = self.paths[:, -1] >= THRESHOLD
        log_num = logsumexp(self.log_correction[event]) if event.any() else -np.inf
        return {
            "ordinary": float(np.exp(log_num - np.log(len(event)))),
            "snis": float(np.exp(log_num - logsumexp(self.log_correction))),
        }

    def contributions(self):
        c = np.exp(self.log_correction)
        return c * (self.paths[:, -1] >= THRESHOLD), c


def run_smc(rng, n, config, trace=False):
    paths = base_paths(rng, n, config.steps)
    log_w = np.zeros(n)
    transitions = n * config.steps
    accepted = proposals = 0
    schedule = np.linspace(0.0, config.tilt, config.levels + 1)
    history = {"initial": paths.copy(), "before": [], "after": [], "increments": [], "log_z_increments": []} if trace else None
    for old_beta, beta in zip(schedule[:-1], schedule[1:]):
        # Critical: evaluate the potential BEFORE this level's mutation.
        increment = (beta - old_beta) * paths[:, -1]
        if trace:
            history["before"].append(paths[:, -1].copy())
            history["increments"].append(increment.copy())
            history["log_z_increments"].append(float(logsumexp(log_w + increment) - logsumexp(log_w)))
        log_w += increment
        for _ in range(config.sweeps):
            moves, cost = rollback_mh(rng, paths, beta, config.max_depth)
            accepted += moves
            proposals += n
            transitions += cost
        if trace:
            history["after"].append(paths[:, -1].copy())
    return Result(paths, log_w, log_w - config.tilt * paths[:, -1], accepted, proposals, transitions, history)


def ideal_variances(tilt):
    """CLT coefficients ONLY for independent exact pi_lambda samples."""
    tail = norm.sf(THRESHOLD + tilt)
    scale = np.exp(tilt**2)
    return {
        "ordinary": float(scale * tail - P_EVENT**2),
        "snis": float(scale * ((1 - 2 * P_EVENT) * tail + P_EVENT**2)),
    }
