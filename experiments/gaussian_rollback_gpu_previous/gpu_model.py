"""GPU-only full-trajectory RollBack SMC estimators and two MC controls."""
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from scipy.stats import norm

P_A = 0.001
a = float(norm.isf(P_A))  # Theoretical quantile. NEVER estimated from samples.
T = 16
D = 16  # Fixed for every experiment.
DTYPE = torch.float32
DEVICE = torch.device("cuda:0")


@dataclass(frozen=True)
class GPUConfig:
    n: int
    levels: int
    tilt: float


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab select a GPU runtime; no CPU fallback is used.")
    torch.set_grad_enabled(False)
    return {"device": str(DEVICE), "gpu": torch.cuda.get_device_name(DEVICE),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "dtype": str(DTYPE), "T": T, "D": D, "threshold": a, "p_A": P_A}


def generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def gaussian_paths(n, rng):
    innovations = torch.randn((n, T), device=DEVICE, dtype=DTYPE, generator=rng) / math.sqrt(T)
    return torch.cat((torch.zeros((n, 1), device=DEVICE, dtype=DTYPE), innovations.cumsum(dim=1)), dim=1)


def rollback(paths, beta, rng):
    n = paths.shape[0]
    depths = torch.randint(1, D+1, (n,), device=DEVICE, generator=rng)
    cut = T-depths
    columns = torch.arange(T, device=DEVICE)[None, :]
    suffix = columns >= cut[:, None]
    # Padded GPU implementation: N*T normals are drawn, only sum(depths) used.
    noise = torch.randn((n, T), device=DEVICE, dtype=DTYPE, generator=rng) / math.sqrt(T)
    noise = torch.where(suffix, noise, 0.0)
    anchor = paths.gather(1, cut[:, None])
    future = anchor + noise.cumsum(dim=1)
    proposal = torch.cat((paths[:, :1], torch.where(suffix, future, paths[:, 1:])), dim=1)
    accept = torch.log(torch.rand(n, device=DEVICE, dtype=DTYPE, generator=rng)) < torch.minimum(
        torch.zeros((), device=DEVICE, dtype=DTYPE), beta*(proposal[:, -1]-paths[:, -1]))
    return torch.where(accept[:, None], proposal, paths), depths, accept


def systematic(log_weights, rng, offset=None):
    n = log_weights.numel()
    cumulative = log_weights.exp().cumsum(0)
    cumulative[-1] = 1.0
    if offset is None:
        offset = torch.rand((), device=DEVICE, generator=rng)
    points = (torch.arange(n, device=DEVICE, dtype=DTYPE) + offset)/n
    return torch.searchsorted(cumulative, points, right=False)


@torch.inference_mode()
def run_smc_population(config, seed, debug=False):
    rng = generator(seed)
    n = config.n
    paths = gaussian_paths(n, rng)
    log_w = torch.full((n,), -math.log(n), device=DEVICE, dtype=DTYPE)
    log_z = torch.zeros((), device=DEVICE, dtype=DTYPE)
    used_transitions = torch.tensor(n*T, device=DEVICE, dtype=torch.int64)
    roots = torch.arange(n, device=DEVICE)
    resampled = []
    ess_by_level = []
    max_weight_by_level = []
    history = []
    for k in range(1, config.levels+1):
        beta = config.tilt*k/config.levels
        increment = config.tilt/config.levels * paths[:, -1]
        if debug:
            old_x, old_w, old_z = paths[:, -1].clone(), log_w.clone(), log_z.clone()
        unnormalized = log_w + increment
        log_c = torch.logsumexp(unnormalized, 0)
        log_z = log_z + log_c  # Every level, including levels without resampling.
        log_w = unnormalized-log_c
        ess = torch.exp(-torch.logsumexp(2*log_w, 0))
        # Both diagnostics refer to the weighted population BEFORE selection.
        ess_by_level.append(ess/n)
        max_weight_by_level.append(log_w.max().exp())
        # The unavoidable ESS host decision and its synchronization are TIMED.
        selected = k < config.levels and bool((ess < 0.5*n).item())
        if selected:
            parents = systematic(log_w, rng)
            paths = paths.index_select(0, parents)
            roots = roots.index_select(0, parents)
            log_w = torch.full_like(log_w, -math.log(n))
            resampled.append(k)
        paths, depths, accepted = rollback(paths, beta, rng)
        used_transitions += depths.sum()
        if debug:
            history.append({"level": k, "old_x": old_x, "old_log_w": old_w,
                            "old_log_z": old_z, "log_c": log_c.clone(),
                            "log_z": log_z.clone(), "ess": ess.clone(),
                            "max_weight_before": max_weight_by_level[-1].clone(),
                            "resampled": selected, "depths": depths.clone(),
                            "accepted": accepted.clone()})
    corrected_log_weights = log_w-config.tilt*paths[:, -1]
    # Score is x_0; event uses the theoretical Gaussian quantile.
    event = paths[:, -1] >= a
    log_num = torch.logsumexp(torch.where(event, corrected_log_weights, -torch.inf), 0)
    log_den = torch.logsumexp(corrected_log_weights, 0)
    normalized_corrected_log_weights = corrected_log_weights-log_den
    terminal_ess_fraction = torch.exp(-torch.logsumexp(2*log_w, 0))/n
    reverse_ess_fraction = torch.exp(-torch.logsumexp(2*normalized_corrected_log_weights, 0))/n
    distinct_roots = torch.unique(roots).numel()
    # BOTH estimators use this SAME weighted terminal population and therefore
    # share the entire generation, weighting, resampling and mutation cost.
    estimate_smc = float(torch.exp(log_z+log_num).item())
    estimate_snis = float(torch.exp(log_num-log_den).item())
    result = {"smc": estimate_smc, "snis": estimate_snis,
              "z_hat": float(log_z.exp().item()),
              "reverse_mass": float(torch.exp(log_z+log_den).item()),
              "used_transitions": int(used_transitions.item()),
              "gaussian_variates_drawn": n*T*(config.levels+1),
              "resampling_count": len(resampled), "resampled_levels": resampled,
              "ess_by_level": torch.stack(ess_by_level).cpu().tolist(),
              "max_weight_by_level": torch.stack(max_weight_by_level).cpu().tolist(),
              "terminal_ess_fraction": float(terminal_ess_fraction.item()),
              "terminal_max_weight": float(log_w.max().exp().item()),
              "reverse_ess_fraction": float(reverse_ess_fraction.item()),
              "reverse_max_weight": float(normalized_corrected_log_weights.max().exp().item()),
              "distinct_roots": int(distinct_roots),
              "distinct_root_fraction": float(distinct_roots/n),
              "n": n, "R": config.levels, "lambda": config.tilt}
    if debug:
        result.update(paths=paths, log_weights=log_w, log_z=log_z,
                      roots=roots, history=history)
    return result


@torch.inference_mode()
def crude_paths(count, seed, chunk_size=262144):
    """Primary MC: explicitly generate complete paths from the SAME Q_t."""
    rng = generator(seed)
    hits = torch.zeros((), device=DEVICE, dtype=torch.int64)
    for first in range(0, count, chunk_size):
        paths = gaussian_paths(min(chunk_size, count-first), rng)
        hits += (paths[:, -1] >= a).sum()
    estimate = float((hits.to(torch.float64)/count).item())
    return {"estimate": estimate, "used_transitions": count*T,
            "gaussian_variates_drawn": count*T, "n": count, "R": 0, "lambda": 0,
            "resampling_count": 0, "resampled_levels": []}


@torch.inference_mode()
def crude_direct(count, seed, chunk_size=4194304):
    """Separate toy-specific control: X ~ N(0,1), without trajectories."""
    rng = generator(seed)
    hits = torch.zeros((), device=DEVICE, dtype=torch.int64)
    for first in range(0, count, chunk_size):
        x = torch.randn(min(chunk_size, count-first), device=DEVICE, dtype=DTYPE, generator=rng)
        hits += (x >= a).sum()
    estimate = float((hits.to(torch.float64)/count).item())
    return {"estimate": estimate, "used_transitions": 0,
            "gaussian_variates_drawn": count, "n": count, "R": 0, "lambda": 0,
            "resampling_count": 0, "resampled_levels": []}


def measured(function, *args):
    """Wall clock includes allocation, RNG setup, kernels, host ESS, correction.

    Warmup is separate. Synchronization before t0 drains prior work; final
    synchronization is inside the interval. No CUDA-event-only timing.
    """
    torch.cuda.synchronize(DEVICE)
    start = time.perf_counter()
    result = function(*args)
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter()-start
    return result, elapsed


def make_seed(stream, config_id, rep, root=20260929):
    return int(np.random.SeedSequence([root, stream, config_id, rep]).generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63-1))


def warmup(repetitions=30):
    start = time.perf_counter()
    for rep in range(repetitions):
        seed = make_seed(90, 0, rep)
        run_smc_population(GPUConfig(128, 4, 2), seed)
        crude_paths(32768, seed)
        crude_direct(1048576, seed)
    torch.cuda.synchronize(DEVICE)
    return time.perf_counter()-start
