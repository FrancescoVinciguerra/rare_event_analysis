"""One deep-tilt RollBack SMC run with per-level estimators and terminal copies."""
import math
import time
import numpy as np
import torch
from scipy.stats import norm

T = 16
N = 128
R = 20
DELTA_BETA = 0.25
LAMBDA = DELTA_BETA * R
P_EVENTS = (1e-4, 1e-5)
THRESHOLDS = tuple(float(norm.isf(p)) for p in P_EVENTS)
DEVICE = torch.device("cuda:0")
DTYPE = torch.float32


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime; all estimators use CUDA.")
    torch.set_grad_enabled(False)
    return {"device": str(DEVICE), "gpu": torch.cuda.get_device_name(DEVICE),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "T": T, "N": N, "R": R, "lambda": LAMBDA,
            "p_A": list(P_EVENTS), "thresholds_theoretical": list(THRESHOLDS),
            "dtype": str(DTYPE)}


def make_seed(stream, rep, level=0, root=20261001):
    return int(np.random.SeedSequence([root, stream, rep, level]).generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63-1))


def generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def gaussian_paths(count, rng):
    increments = torch.randn((count, T), device=DEVICE, dtype=DTYPE, generator=rng) / math.sqrt(T)
    return torch.cat((torch.zeros((count, 1), device=DEVICE, dtype=DTYPE), increments.cumsum(1)), dim=1)


def rollback_M_k(paths, beta, rng):
    count = paths.shape[0]
    d = torch.randint(1, T+1, (count,), device=DEVICE, generator=rng)
    cut = T-d
    column = torch.arange(T, device=DEVICE)[None, :]
    regenerate = column >= cut[:, None]
    # Padded CUDA draw: all T normals are drawn; only d are used per path.
    increments = torch.randn((count, T), device=DEVICE, dtype=DTYPE, generator=rng) / math.sqrt(T)
    increments = torch.where(regenerate, increments, 0.0)
    anchor = paths.gather(1, cut[:, None])
    future = anchor + increments.cumsum(1)
    proposal = torch.cat((paths[:, :1], torch.where(regenerate, future, paths[:, 1:])), dim=1)
    log_alpha = torch.minimum(torch.zeros((), device=DEVICE, dtype=DTYPE),
                              beta*(proposal[:, -1]-paths[:, -1]))
    accepted = torch.log(torch.rand(count, device=DEVICE, dtype=DTYPE, generator=rng)) < log_alpha
    return torch.where(accepted[:, None], proposal, paths), d, accepted


def systematic(log_weights, rng, offset=None):
    count = len(log_weights)
    cumulative = log_weights.exp().cumsum(0)
    cumulative[-1] = 1.0
    if offset is None:
        offset = torch.rand((), device=DEVICE, generator=rng)
    points = (torch.arange(count, device=DEVICE, dtype=DTYPE)+offset)/count
    return torch.searchsorted(cumulative, points, right=False)


def timed(function, *args):
    torch.cuda.synchronize(DEVICE)
    begin = time.perf_counter()
    value = function(*args)
    torch.cuda.synchronize(DEVICE)
    return value, time.perf_counter()-begin


@torch.inference_mode()
def initialize(seed, count=N):
    rng = generator(seed)
    return {"paths": gaussian_paths(count, rng),
            "log_weights": torch.full((count,), -math.log(count), device=DEVICE, dtype=DTYPE),
            "log_z": torch.zeros((), device=DEVICE, dtype=DTYPE),
            "roots": torch.arange(count, device=DEVICE), "rng": rng}


@torch.inference_mode()
def weight_before_M_k(state, beta):
    paths, log_weights = state["paths"], state["log_weights"]
    unnormalized = log_weights + DELTA_BETA*paths[:, -1]
    log_increment = torch.logsumexp(unnormalized, 0)
    state["log_z"] = state["log_z"]+log_increment
    state["log_weights"] = unnormalized-log_increment
    pre_ess = torch.exp(-torch.logsumexp(2*state["log_weights"], 0))/len(paths)
    pre_max = state["log_weights"].max().exp()
    return {"pre_ess_fraction": float(pre_ess.item()),
            "pre_max_weight": float(pre_max.item()),
            "log_increment": float(log_increment.item())}


@torch.inference_mode()
def evaluate_level(paths, log_weights, log_z, roots, beta):
    count = len(paths)
    terminal = paths[:, -1]
    corrected_log_weights = log_weights-beta*terminal
    log_denominator = torch.logsumexp(corrected_log_weights, 0)
    normalized_corrected = corrected_log_weights-log_denominator
    thresholds = torch.tensor(THRESHOLDS, device=DEVICE, dtype=DTYPE)
    event = terminal[:, None] >= thresholds[None, :]
    log_numerators = torch.logsumexp(
        torch.where(event, corrected_log_weights[:, None], -torch.inf), dim=0)
    smc_estimates = torch.exp(log_z+log_numerators)
    snis_estimates = torch.exp(log_numerators-log_denominator)
    forward_ess = torch.exp(-torch.logsumexp(2*log_weights, 0))/count
    reverse_ess = torch.exp(-torch.logsumexp(2*normalized_corrected, 0))/count
    distinct_roots = torch.unique(roots).numel()
    return {"smc": smc_estimates.cpu().tolist(),
            "snis": snis_estimates.cpu().tolist(),
            "event_particles": event.sum(0).cpu().tolist(),
            "smc_ess_fraction": float(forward_ess.item()),
            "smc_max_weight": float(log_weights.max().exp().item()),
            "reverse_ess_fraction": float(reverse_ess.item()),
            "reverse_max_weight": float(normalized_corrected.max().exp().item()),
            "distinct_root_fraction": float(distinct_roots/count),
            "z_hat": float(log_z.exp().item())}


@torch.inference_mode()
def terminal_copy_evaluation(state, beta, seed, debug=False):
    """Copy AFTER weighting, BEFORE resampling; mutate with NO terminal selection."""
    rng = generator(seed)
    paths = state["paths"].clone()
    weights = state["log_weights"].clone()
    roots = state["roots"].clone()
    paths, depths, _ = rollback_M_k(paths, beta, rng)
    result = evaluate_level(paths, weights, state["log_z"], roots, beta)
    result["extra_used_transitions"] = int(depths.sum().item())
    result["extra_gaussian_draws"] = len(paths)*T
    if debug:
        result.update(paths=paths, log_weights=weights, roots=roots,
                      log_z=state["log_z"].clone())
    return result


@torch.inference_mode()
def main_resample_mutate_evaluate(state, beta, level, pre_ess_fraction, debug=False):
    count = len(state["paths"])
    selected = level < R and pre_ess_fraction < .5
    if selected:
        parents = systematic(state["log_weights"], state["rng"])
        state["paths"] = state["paths"].index_select(0, parents)
        state["roots"] = state["roots"].index_select(0, parents)
        state["log_weights"] = torch.full_like(state["log_weights"], -math.log(count))
    state["paths"], depths, _ = rollback_M_k(state["paths"], beta, state["rng"])
    result = evaluate_level(state["paths"], state["log_weights"], state["log_z"],
                            state["roots"], beta)
    result["resampled"] = bool(selected)
    result["used_transitions_stage"] = int(depths.sum().item())
    result["gaussian_draws_stage"] = count*T
    if debug:
        result.update(paths=state["paths"].clone(),
                      log_weights=state["log_weights"].clone(),
                      roots=state["roots"].clone(), log_z=state["log_z"].clone())
    return result


def run_deep(seed, count=N, debug=False):
    """All 20 levels of ONE SMC run; no averaging of estimates across levels."""
    state, initial_seconds = timed(initialize, seed, count)
    main_seconds = initial_seconds
    extra_seconds = 0.0
    main_transitions = count*T
    extra_transitions = 0
    main_gaussian_draws = count*T
    extra_gaussian_draws = 0
    rows = []
    traces = []
    for level in range(1, R+1):
        beta = DELTA_BETA*level
        if debug:
            before = {"x": state["paths"][:, -1].clone(),
                      "log_weights": state["log_weights"].clone(),
                      "log_z": state["log_z"].clone(),
                      "roots": state["roots"].clone()}
        pre, elapsed = timed(weight_before_M_k, state, beta)
        main_seconds += elapsed
        terminal, elapsed = timed(terminal_copy_evaluation, state, beta,
                                  make_seed(2, seed, level), debug)
        terminal_elapsed = elapsed
        extra_seconds += terminal_elapsed
        extra_transitions += terminal["extra_used_transitions"]
        extra_gaussian_draws += terminal["extra_gaussian_draws"]
        main, elapsed = timed(main_resample_mutate_evaluate, state, beta, level,
                              pre["pre_ess_fraction"], debug)
        main_elapsed = elapsed
        main_seconds += elapsed
        main_transitions += main["used_transitions_stage"]
        main_gaussian_draws += main["gaussian_draws_stage"]
        common = {"level": level, "beta": beta, "seed": seed, "N": count,
                  "pre_ess_fraction": pre["pre_ess_fraction"],
                  "pre_max_weight": pre["pre_max_weight"],
                  "log_z_increment": pre["log_increment"],
                  "main_cumulative_seconds": main_seconds,
                  "main_cumulative_transitions": main_transitions,
                  "main_cumulative_gaussian_draws": main_gaussian_draws,
                  "terminal_copy_cumulative_seconds": main_seconds-main_elapsed+terminal_elapsed,
                  "terminal_copy_cumulative_transitions": main_transitions-main["used_transitions_stage"]+terminal["extra_used_transitions"],
                  "extra_cumulative_seconds": extra_seconds,
                  "extra_cumulative_transitions": extra_transitions,
                  "extra_cumulative_gaussian_draws": extra_gaussian_draws}
        for label, result in (("main", main), ("terminal_copy", terminal)):
            for index, p_true in enumerate(P_EVENTS):
                rows.append({**common, "population": label, "p_A": p_true,
                             "smc": result["smc"][index],
                             "snis": result["snis"][index],
                             "event_particles": result["event_particles"][index],
                             "smc_ess_fraction": result["smc_ess_fraction"],
                             "smc_max_weight": result["smc_max_weight"],
                             "reverse_ess_fraction": result["reverse_ess_fraction"],
                             "reverse_max_weight": result["reverse_max_weight"],
                             "distinct_root_fraction": result["distinct_root_fraction"],
                             "resampled": result.get("resampled", False),
                             "z_hat": result["z_hat"]})
        if debug:
            traces.append({"level": level, "beta": beta, "before": before,
                           "weighted_log_weights": state["log_weights"].clone() if not main["resampled"] else None,
                           "pre": pre, "main": main, "terminal": terminal})
    return rows, traces


@torch.inference_mode()
def crude_paths_two_events(count, seed, chunk_size=262144):
    rng = generator(seed)
    hits = torch.zeros(2, device=DEVICE, dtype=torch.int64)
    thresholds = torch.tensor(THRESHOLDS, device=DEVICE, dtype=DTYPE)
    for start in range(0, count, chunk_size):
        paths = gaussian_paths(min(chunk_size, count-start), rng)
        hits += (paths[:, -1, None] >= thresholds).sum(0)
    return (hits.to(torch.float64)/count).cpu().tolist()


@torch.inference_mode()
def crude_prefix_estimates(counts, seed, chunk_size=262144):
    """Use common P paths across level budgets, retaining correct marginal MC law."""
    rng = generator(seed)
    requested = np.asarray(counts, dtype=int)
    unique = np.unique(requested)
    maximum = int(unique[-1])
    thresholds = torch.tensor(THRESHOLDS, device=DEVICE, dtype=DTYPE)
    hits = torch.zeros(2, device=DEVICE, dtype=torch.int64)
    values = {}
    previous = 0
    for end in range(chunk_size, maximum+chunk_size, chunk_size):
        end = min(end, maximum)
        paths = gaussian_paths(end-previous, rng)
        event_cumulative = (paths[:, -1, None] >= thresholds).to(torch.int32).cumsum(0)
        for target in unique[(unique > previous) & (unique <= end)]:
            hit_at_target = hits+event_cumulative[int(target-previous-1)]
            values[int(target)] = (hit_at_target.to(torch.float64)/int(target)).cpu().tolist()
        hits += event_cumulative[-1].to(torch.int64)
        previous = end
        if end == maximum:
            break
    return [values[int(n)] for n in requested]


def warmup(repetitions=20):
    for rep in range(repetitions):
        seed = make_seed(99, rep)
        state = initialize(seed)
        pre = weight_before_M_k(state, .25)
        terminal_copy_evaluation(state, .25, make_seed(100, rep))
        main_resample_mutate_evaluate(state, .25, 1, pre["pre_ess_fraction"])
        crude_paths_two_events(32768, seed)
    torch.cuda.synchronize(DEVICE)
