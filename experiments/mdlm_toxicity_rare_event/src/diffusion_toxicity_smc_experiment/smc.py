"""SMC on the native MDLM reverse chain and component-wise reverse-SNIS."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .ancestral import NativeMDLMSampler


RewardFunction = Callable[[torch.Tensor], torch.Tensor]
EventFunction = Callable[[torch.Tensor], torch.Tensor]


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    log_weights = torch.as_tensor(log_weights, dtype=torch.float64)
    if log_weights.ndim != 1 or log_weights.numel() == 0:
        raise ValueError("log_weights must be a non-empty vector")
    if bool(torch.isnan(log_weights).any().item()) or bool(torch.isposinf(log_weights).any().item()):
        raise ValueError("log_weights may be finite or -inf, but not NaN/+inf")
    normalizer = torch.logsumexp(log_weights, dim=0)
    if not bool(torch.isfinite(normalizer).item()):
        raise ValueError("at least one log weight must be finite")
    return log_weights - normalizer


def ess_from_log_weights(normalized_log_weights: torch.Tensor) -> float:
    normalized = normalize_log_weights(normalized_log_weights)
    return float(torch.exp(-torch.logsumexp(2.0 * normalized, dim=0)).item())


def resample_indices(
    normalized_log_weights: torch.Tensor,
    *,
    method: str,
    generator: torch.Generator,
    num_samples: int | None = None,
) -> torch.Tensor:
    log_weights = normalize_log_weights(normalized_log_weights)
    weights = log_weights.exp()
    count = weights.numel() if num_samples is None else int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be positive")
    if method == "multinomial":
        return torch.multinomial(weights, count, replacement=True, generator=generator)
    if method == "systematic":
        start = torch.rand((), device=weights.device, generator=generator) / count
        points = start + torch.arange(count, device=weights.device) / count
    elif method == "stratified":
        points = (
            torch.arange(count, device=weights.device)
            + torch.rand(count, device=weights.device, generator=generator)
        ) / count
    else:
        raise ValueError("resampling method must be multinomial, systematic, or stratified")
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, points, right=False)


def beta_at_level(
    *, level: int, terminal_level: int, lambda_final: float, start_fraction: float
) -> float:
    if level <= 0:
        return 0.0
    if level >= terminal_level:
        return float(lambda_final)
    fraction = level / terminal_level
    if fraction <= start_fraction:
        return 0.0
    return float(lambda_final) * (fraction - start_fraction) / (1.0 - start_fraction)


def telescoping_log_increments(beta: Sequence[float], rewards: Sequence[float]) -> list[float]:
    if len(beta) != len(rewards) or len(beta) < 2:
        raise ValueError("beta and rewards must have equal length >= 2")
    potentials = [float(b) * float(r) for b, r in zip(beta, rewards)]
    return [current - previous for previous, current in zip(potentials, potentials[1:])]


def logmeanexp_sample_rewards(sample_rewards: torch.Tensor) -> torch.Tensor:
    """Aggregate FK's K log-probability rewards as log(mean(probability))."""

    rewards = torch.as_tensor(sample_rewards, dtype=torch.float64)
    if rewards.ndim != 2 or rewards.shape[1] <= 0:
        raise ValueError("sample_rewards must have shape [particles, samples]")
    if not bool(torch.isfinite(rewards).all().item()):
        raise FloatingPointError("intermediate sample rewards contain NaN or Inf")
    return torch.logsumexp(rewards, dim=1) - math.log(rewards.shape[1])


@dataclass(frozen=True)
class ReverseSNISResult:
    estimate: float
    corrected_ess: float
    corrected_ess_over_n: float
    corrected_log_weights: tuple[float, ...]
    event_discoveries: int
    tilted_event_fraction: float


def reverse_snis_component(
    *,
    final_log_weights: torch.Tensor,
    rewards: torch.Tensor,
    events: torch.Tensor,
    lambda_final: float,
) -> ReverseSNISResult:
    """Consistent component-wise SNIS correction; generally biased at finite N."""

    log_weights = normalize_log_weights(final_log_weights.double())
    rewards = torch.as_tensor(rewards, dtype=torch.float64, device=log_weights.device)
    events = torch.as_tensor(events, dtype=torch.bool, device=log_weights.device)
    if rewards.shape != log_weights.shape or events.shape != log_weights.shape:
        raise ValueError("weights, rewards, and events must have the same shape")
    corrected = normalize_log_weights(log_weights - float(lambda_final) * rewards)
    probabilities = corrected.exp()
    estimate = float(torch.sum(probabilities * events.double()).item())
    ess = ess_from_log_weights(corrected)
    discoveries = int(events.sum().item())
    return ReverseSNISResult(
        estimate=estimate,
        corrected_ess=ess,
        corrected_ess_over_n=ess / events.numel(),
        corrected_log_weights=tuple(float(value) for value in corrected.cpu().tolist()),
        event_discoveries=discoveries,
        tilted_event_fraction=discoveries / events.numel(),
    )


def aggregate_component_estimates(
    estimates: Sequence[float], coefficients: Sequence[float]
) -> float:
    if len(estimates) == 0 or len(estimates) != len(coefficients):
        raise ValueError("one coefficient is required per component")
    weights_a = torch.tensor(coefficients, dtype=torch.float64)
    if bool((weights_a < 0).any().item()) or not math.isclose(
        float(weights_a.sum()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("component coefficients must be non-negative and sum to one")
    estimate_tensor = torch.tensor(estimates, dtype=torch.float64)
    if not bool(torch.isfinite(estimate_tensor).all().item()):
        raise FloatingPointError("component estimates contain NaN or Inf")
    return float(torch.dot(weights_a, estimate_tensor).item())


@dataclass(frozen=True)
class SMCComponentConfig:
    num_particles: int
    num_steps: int
    sequence_length: int
    lambda_final: float
    ess_threshold: float = 0.5
    resampling_method: str = "multinomial"
    reward_frequency: int = 20
    beta_start_fraction: float = 0.0
    final_resampling: bool = False
    microbatch_size: int | None = None
    seed: int = 1234
    tilted_sampler_type: str = "fk_bootstrap"
    num_tempering_levels: int = 8
    mcmc_steps_per_level: int = 1
    mutation_policy: str = "after_resampling"
    rollback_steps: tuple[int, ...] = ()
    rollback_probabilities: tuple[float, ...] = ()
    waste_free_chain_length: int = 1
    num_x0_samples: int = 4
    intermediate_reward_aggregation: str = "logmeanexp"

    def __post_init__(self) -> None:
        if self.num_particles <= 0 or self.num_steps <= 0 or self.sequence_length <= 0:
            raise ValueError("particle, step, and length counts must be positive")
        if not 0.0 <= self.ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be in [0, 1]")
        if self.resampling_method not in {"multinomial", "systematic", "stratified"}:
            raise ValueError("invalid resampling method")
        if self.reward_frequency <= 0:
            raise ValueError("reward_frequency must be positive")
        if self.num_x0_samples <= 0:
            raise ValueError("num_x0_samples must be positive")
        if self.intermediate_reward_aggregation != "logmeanexp":
            raise ValueError("intermediate_reward_aggregation must be logmeanexp")
        if self.microbatch_size is not None and self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive when provided")
        if not 0.0 <= self.beta_start_fraction < 1.0:
            raise ValueError("beta_start_fraction must be in [0, 1)")
        if self.tilted_sampler_type not in {
            "fk_bootstrap",
            "trajectory_resample_move",
            "trajectory_waste_free",
        }:
            raise ValueError("invalid tilted_sampler_type")
        if self.num_tempering_levels <= 0:
            raise ValueError("num_tempering_levels must be positive")
        if self.mcmc_steps_per_level < 0:
            raise ValueError("mcmc_steps_per_level must be non-negative")
        if self.mutation_policy not in {"after_resampling", "every_level"}:
            raise ValueError("mutation_policy must be after_resampling or every_level")
        trajectory_backend = self.tilted_sampler_type in {
            "trajectory_resample_move",
            "trajectory_waste_free",
        }
        if trajectory_backend:
            if (
                self.tilted_sampler_type == "trajectory_resample_move"
                and self.mcmc_steps_per_level <= 0
            ):
                raise ValueError("trajectory resample-move requires MCMC steps")
            if not self.rollback_steps:
                raise ValueError("trajectory backends require rollback_steps")
            if any(step <= 0 or step > self.num_steps for step in self.rollback_steps):
                raise ValueError("rollback_steps must be in [1, num_steps]")
            if len(set(self.rollback_steps)) != len(self.rollback_steps):
                raise ValueError("rollback_steps must be unique")
            if self.rollback_probabilities:
                if len(self.rollback_probabilities) != len(self.rollback_steps):
                    raise ValueError("one rollback probability is required per depth")
                if any(value < 0.0 for value in self.rollback_probabilities):
                    raise ValueError("rollback probabilities must be non-negative")
                if not math.isclose(
                    sum(self.rollback_probabilities), 1.0, abs_tol=1e-9
                ):
                    raise ValueError("rollback probabilities must sum to one")
        if self.tilted_sampler_type == "trajectory_waste_free":
            if self.waste_free_chain_length < 2:
                raise ValueError("trajectory waste-free requires chain_length >= 2")
            if self.num_particles % self.waste_free_chain_length != 0:
                raise ValueError(
                    "num_particles must be divisible by waste-free chain_length"
                )
            if self.mutation_policy != "every_level":
                raise ValueError(
                    "trajectory waste-free requires mutation_policy='every_level'"
                )
            if self.final_resampling:
                raise ValueError(
                    "trajectory waste-free requires final_resampling=false; the "
                    "terminal weighted chain population is part of Algorithm 2"
                )


@dataclass
class SMCComponentResult:
    terminal_tokens: torch.Tensor
    rewards: torch.Tensor
    events: torch.Tensor
    final_log_weights: torch.Tensor
    pre_final_resampling_log_weights: torch.Tensor
    post_final_resampling_log_weights: torch.Tensor
    root_ancestors: torch.Tensor
    genealogy: list[dict[str, Any]]
    level_diagnostics: list[dict[str, Any]]
    reverse_snis: ReverseSNISResult
    log_normalizer_estimate: float
    nfe: int
    runtime_seconds: float
    tilted_sampler_type: str = "fk_bootstrap"
    mutation_diagnostics: list[dict[str, Any]] | None = None
    trajectory_storage_bytes: int = 0
    waste_free_diagnostics: list[dict[str, Any]] | None = None
    waste_free_chain_ids: torch.Tensor | None = None
    waste_free_chain_positions: torch.Tensor | None = None


def _apply_potential(
    *,
    log_weights: torch.Tensor,
    current_log_potential: torch.Tensor,
    previous_log_potential: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    unnormalized = log_weights + current_log_potential - previous_log_potential
    log_increment = torch.logsumexp(unnormalized, dim=0)
    return unnormalized - log_increment, float(log_increment.item())


@torch.inference_mode()
def run_fk_bootstrap_component(
    *,
    sampler: NativeMDLMSampler,
    fixed_prefix_token_ids: torch.Tensor | None,
    reward_fn: RewardFunction,
    event_fn: EventFunction,
    config: SMCComponentConfig,
    device: torch.device | str,
) -> SMCComponentResult:
    device = torch.device(device)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    count = config.num_particles
    states, prefix = sampler.initial_states(
        batch_size=count,
        sequence_length=config.sequence_length,
        device=device,
        fixed_prefix_token_ids=fixed_prefix_token_ids,
    )
    log_weights = torch.full(
        (count,), -math.log(count), dtype=torch.float64, device=device
    )
    previous_log_potential = torch.zeros(count, dtype=torch.float64, device=device)
    root_ancestors = torch.arange(count, device=device)
    genealogy: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    log_z = 0.0
    sampler.reset_nfe()
    started = time.perf_counter()
    dt_value = (1.0 - sampler.sampling_epsilon) / config.num_steps
    times = torch.linspace(1.0, sampler.sampling_epsilon, config.num_steps + 1).tolist()
    cache = None
    terminal_level = config.num_steps

    for step, t_scalar in enumerate(times[:-1], start=1):
        intermediate_x0_samples = None
        if config.lambda_final == 0.0:
            # Keep the lambda-zero diagnostic exactly on the ordinary base
            # sampler; no reward-only x0 predictions are needed.
            states, cache = sampler.ddpm_step(
                states,
                t_scalar=float(t_scalar),
                dt=dt_value,
                generator=generator,
                cached_p_x0=cache,
                microbatch_size=config.microbatch_size,
            )
        else:
            states, cache, intermediate_x0_samples = (
                sampler.ddpm_step_with_x0_samples(
                    states,
                    t_scalar=float(t_scalar),
                    dt=dt_value,
                    num_x0_samples=config.num_x0_samples,
                    generator=generator,
                    cached_p_x0=cache,
                    microbatch_size=config.microbatch_size,
                )
            )
        # The final checkpoint is always evaluated below on the true stochastic
        # terminal completion.  Never insert an argmax rollout at the same level.
        score_level = step % config.reward_frequency == 0 and step < config.num_steps
        resampled = False
        ess = ess_from_log_weights(log_weights)
        if score_level and config.lambda_final != 0.0:
            if intermediate_x0_samples is None:
                raise RuntimeError("FK checkpoint is missing intermediate x0 samples")
            flat_x0_samples = intermediate_x0_samples.reshape(
                count * config.num_x0_samples, config.sequence_length
            )
            sample_rewards = reward_fn(flat_x0_samples).to(
                device=device, dtype=torch.float64
            ).reshape(count, config.num_x0_samples)
            rewards = logmeanexp_sample_rewards(sample_rewards)
            beta = beta_at_level(
                level=step,
                terminal_level=terminal_level,
                lambda_final=config.lambda_final,
                start_fraction=config.beta_start_fraction,
            )
            current = beta * rewards
            log_weights, log_increment = _apply_potential(
                log_weights=log_weights,
                current_log_potential=current,
                previous_log_potential=previous_log_potential,
            )
            log_z += log_increment
            previous_log_potential = current
            ess = ess_from_log_weights(log_weights)
            max_weight = float(log_weights.max().exp().item())
            if ess < config.ess_threshold * count:
                parents = resample_indices(
                    log_weights, method=config.resampling_method, generator=generator
                )
                states = states[parents]
                root_ancestors = root_ancestors[parents]
                previous_log_potential = previous_log_potential[parents]
                log_weights.fill_(-math.log(count))
                cache = None
                resampled = True
                genealogy.append(
                    {
                        "level": step,
                        "parents": [int(value) for value in parents.cpu().tolist()],
                        "distinct_root_ancestors": len(
                            set(int(value) for value in root_ancestors.cpu().tolist())
                        ),
                    }
                )
        else:
            max_weight = float(log_weights.max().exp().item())
        diagnostics.append(
            {
                "level": step,
                "ess": ess,
                "checkpoint": bool(score_level and config.lambda_final != 0.0),
                "beta": (
                    beta_at_level(
                        level=step,
                        terminal_level=terminal_level,
                        lambda_final=config.lambda_final,
                        start_fraction=config.beta_start_fraction,
                    )
                    if score_level and config.lambda_final != 0.0
                    else None
                ),
                "ess_before_resampling": (
                    ess if score_level and config.lambda_final != 0.0 else None
                ),
                "max_normalized_weight_before_resampling": (
                    max_weight if score_level and config.lambda_final != 0.0 else None
                ),
                "resampled": resampled,
                "distinct_root_ancestors": len(
                    set(int(value) for value in root_ancestors.cpu().tolist())
                ),
                "intermediate_x0_sampling": (
                    "mdlm_multinomial" if score_level and config.lambda_final != 0.0 else None
                ),
                "intermediate_x0_samples_per_particle": (
                    config.num_x0_samples
                    if score_level and config.lambda_final != 0.0
                    else None
                ),
                "intermediate_reward_aggregation": (
                    config.intermediate_reward_aggregation
                    if score_level and config.lambda_final != 0.0
                    else None
                ),
            }
        )

    if sampler.final_noise_removal:
        states = sampler.terminal_completion(
            states,
            t_scalar=sampler.sampling_epsilon,
            generator=generator,
            microbatch_size=config.microbatch_size,
        )
    if bool((states == sampler.mask_token_id).any().item()):
        raise RuntimeError("terminal particles still contain masks")
    terminal_rewards = reward_fn(states).to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(terminal_rewards).all().item()):
        raise FloatingPointError("terminal rewards contain NaN or Inf")
    terminal_events = event_fn(states).to(device=device, dtype=torch.bool)
    if prefix.numel() and not torch.equal(
        states[:, : prefix.numel()], prefix.expand(count, -1)
    ):
        raise RuntimeError("fixed prefix changed during SMC propagation")
    current = float(config.lambda_final) * terminal_rewards
    log_weights, log_increment = _apply_potential(
        log_weights=log_weights,
        current_log_potential=current,
        previous_log_potential=previous_log_potential,
    )
    log_z += log_increment
    terminal_ess = ess_from_log_weights(log_weights)
    pre_final_resampling_log_weights = log_weights.clone()
    if config.final_resampling:
        parents = resample_indices(
            log_weights, method=config.resampling_method, generator=generator
        )
        states = states[parents]
        terminal_rewards = terminal_rewards[parents]
        terminal_events = terminal_events[parents]
        root_ancestors = root_ancestors[parents]
        log_weights.fill_(-math.log(count))
        genealogy.append(
            {
                "level": terminal_level,
                "parents": [int(value) for value in parents.cpu().tolist()],
                "final_resampling": True,
            }
        )
    diagnostics.append(
        {
            "level": terminal_level,
            "ess": terminal_ess,
            "checkpoint": True,
            "beta": float(config.lambda_final),
            "ess_before_resampling": terminal_ess,
            "max_normalized_weight_before_resampling": float(
                pre_final_resampling_log_weights.max().exp().item()
            ),
            "ess_after_resampling": ess_from_log_weights(log_weights),
            "max_normalized_weight_after_resampling": float(
                log_weights.max().exp().item()
            ),
            "resampled": bool(config.final_resampling),
            "terminal": True,
            "distinct_root_ancestors": len(
                set(int(value) for value in root_ancestors.cpu().tolist())
            ),
        }
    )
    post_final_resampling_log_weights = log_weights.clone()
    reverse = reverse_snis_component(
        final_log_weights=log_weights,
        rewards=terminal_rewards,
        events=terminal_events,
        lambda_final=config.lambda_final,
    )
    return SMCComponentResult(
        terminal_tokens=states,
        rewards=terminal_rewards,
        events=terminal_events,
        final_log_weights=log_weights,
        pre_final_resampling_log_weights=pre_final_resampling_log_weights,
        post_final_resampling_log_weights=post_final_resampling_log_weights,
        root_ancestors=root_ancestors,
        genealogy=genealogy,
        level_diagnostics=diagnostics,
        reverse_snis=reverse,
        log_normalizer_estimate=log_z,
        nfe=sampler.nfe,
        runtime_seconds=time.perf_counter() - started,
        tilted_sampler_type="fk_bootstrap",
        mutation_diagnostics=[],
    )


def trajectory_mh_log_acceptance(
    current_rewards: torch.Tensor,
    proposed_rewards: torch.Tensor,
    *,
    target_lambda: float,
) -> torch.Tensor:
    """Exact path-space MH ratio after cancellation of native MDLM kernels."""

    current = torch.as_tensor(current_rewards, dtype=torch.float64)
    proposed = torch.as_tensor(
        proposed_rewards, dtype=torch.float64, device=current.device
    )
    if current.ndim != 1 or proposed.shape != current.shape:
        raise ValueError("current and proposed rewards must be equal-length vectors")
    log_ratio = float(target_lambda) * (proposed - current)
    if not bool(torch.isfinite(log_ratio).all().item()):
        raise FloatingPointError("trajectory MH ratio contains NaN or Inf")
    return torch.minimum(torch.zeros_like(log_ratio), log_ratio)


def _validate_terminal_reward_recheck(
    cached: torch.Tensor,
    recomputed: torch.Tensor,
    *,
    backend: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Fail with actionable diagnostics if a deterministic reward drifts."""

    cached = torch.as_tensor(cached, dtype=torch.float64)
    recomputed = torch.as_tensor(
        recomputed, dtype=torch.float64, device=cached.device
    )
    if cached.shape != recomputed.shape:
        raise RuntimeError(
            f"{backend} terminal reward shape changed: "
            f"cached={tuple(cached.shape)} recomputed={tuple(recomputed.shape)}"
        )
    close = torch.isclose(cached, recomputed, rtol=rtol, atol=atol)
    if bool(close.all().item()):
        return
    absolute = (cached - recomputed).abs()
    scale = torch.maximum(cached.abs(), recomputed.abs()).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    relative = absolute / scale
    max_index = int(absolute.argmax().item())
    raise RuntimeError(
        f"{backend} terminal reward changed when deterministically re-evaluated: "
        f"mismatch_count={int((~close).sum().item())}/{cached.numel()} "
        f"max_abs={float(absolute.max().item()):.17g} "
        f"max_rel={float(relative.max().item()):.17g} "
        f"max_abs_index={max_index} "
        f"cached={float(cached[max_index].item()):.17g} "
        f"recomputed={float(recomputed[max_index].item()):.17g} "
        f"rtol={rtol:.17g} atol={atol:.17g}"
    )


def _rollback_probabilities(config: SMCComponentConfig, device: torch.device) -> torch.Tensor:
    if config.rollback_probabilities:
        values = config.rollback_probabilities
    else:
        values = (1.0 / len(config.rollback_steps),) * len(config.rollback_steps)
    return torch.tensor(values, dtype=torch.float64, device=device)


def _trajectory_mutation_sweep(
    *,
    trajectories: torch.Tensor,
    rewards: torch.Tensor,
    sampler: NativeMDLMSampler,
    reward_fn: RewardFunction,
    target_lambda: float,
    config: SMCComponentConfig,
    generator: torch.Generator,
    level: int,
    sweep: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """One GPU-batched mixture of exact temporal-rollback MH kernels."""

    count = trajectories.shape[0]
    device = trajectories.device
    probabilities = _rollback_probabilities(config, device)
    choices = torch.multinomial(
        probabilities, count, replacement=True, generator=generator
    )
    depth_values = torch.tensor(config.rollback_steps, device=device, dtype=torch.long)
    selected_depths = depth_values.index_select(0, choices)
    proposals = trajectories.clone()
    nfe_before = sampler.nfe
    rollback_counts: dict[str, int] = {}
    # Particles choosing the same rollback depth form one large GPU workload.
    for depth in config.rollback_steps:
        indices = torch.nonzero(selected_depths == depth, as_tuple=False).flatten()
        if indices.numel() == 0:
            rollback_counts[str(depth)] = 0
            continue
        start_step = config.num_steps - depth
        checkpoint_states = trajectories[indices, start_step]
        suffix = sampler.rollout_trajectory_suffix(
            checkpoint_states,
            start_step=start_step,
            num_steps=config.num_steps,
            generator=generator,
            microbatch_size=config.microbatch_size,
        )
        expected_suffix_length = depth + 2
        if suffix.shape[1] != expected_suffix_length:
            raise RuntimeError(
                f"rollback depth {depth} produced suffix length {suffix.shape[1]}, "
                f"expected {expected_suffix_length}"
            )
        proposals[indices, start_step:] = suffix
        rollback_counts[str(depth)] = int(indices.numel())

    proposed_rewards = reward_fn(proposals[:, -1]).to(
        device=device, dtype=torch.float64
    )
    log_acceptance = trajectory_mh_log_acceptance(
        rewards, proposed_rewards, target_lambda=target_lambda
    )
    uniforms = torch.rand(
        count, dtype=torch.float64, device=device, generator=generator
    )
    accepted = torch.log(uniforms) < log_acceptance
    proposed_change_fraction = (
        proposals[:, -1].ne(trajectories[:, -1]).to(torch.float64).mean(dim=1)
    )
    if bool(accepted.any().item()):
        trajectories[accepted] = proposals[accepted]
    rewards = torch.where(accepted, proposed_rewards, rewards)
    accepted_count = int(accepted.sum().item())
    accepted_by_depth = {
        str(depth): int((accepted & (selected_depths == depth)).sum().item())
        for depth in config.rollback_steps
    }
    acceptance_rate_by_depth = {
        key: (accepted_by_depth[key] / value if value else None)
        for key, value in rollback_counts.items()
    }
    diagnostics = {
        "level": int(level),
        "target_lambda": float(target_lambda),
        "sweep": int(sweep),
        "proposals": int(count),
        "accepted": accepted_count,
        "acceptance_rate": accepted_count / count,
        "mean_acceptance_probability": float(log_acceptance.exp().mean().item()),
        "rollback_counts": rollback_counts,
        "rollback_particle_steps": sum(
            int(depth) * rollback_counts[str(depth)]
            for depth in config.rollback_steps
        ),
        "accepted_by_rollback_depth": accepted_by_depth,
        "acceptance_rate_by_rollback_depth": acceptance_rate_by_depth,
        "mean_proposed_terminal_token_change_fraction": float(
            proposed_change_fraction.mean().item()
        ),
        "mean_accepted_terminal_token_change_fraction": (
            float(proposed_change_fraction[accepted].mean().item())
            if accepted_count
            else 0.0
        ),
        "model_nfe": int(sampler.nfe - nfe_before),
    }
    return trajectories, rewards, diagnostics


@torch.inference_mode()
def run_trajectory_resample_move_component(
    *,
    sampler: NativeMDLMSampler,
    fixed_prefix_token_ids: torch.Tensor | None,
    reward_fn: RewardFunction,
    event_fn: EventFunction,
    config: SMCComponentConfig,
    device: torch.device | str,
) -> SMCComponentResult:
    """Annealed path-space SMC with exact native-decoder rollback MH moves.

    The state is the full MDLM trajectory.  Adjacent SMC targets are
    ``P0(path) * exp(eta_k * reward(x0))``.  A mutation returns to a checkpoint
    actually visited by that path and regenerates only its future with the
    unchanged base decoder.  Forward/reverse suffix probabilities therefore
    cancel, leaving the exact acceptance ratio ``exp(eta_k * delta_reward)``.
    """

    if config.tilted_sampler_type != "trajectory_resample_move":
        raise ValueError("trajectory backend requires its matching sampler type")
    if getattr(sampler, "predictor", "ddpm") != "ddpm":
        raise ValueError("trajectory resample-move currently requires predictor='ddpm'")
    device = torch.device(device)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    count = config.num_particles
    sampler.reset_nfe()
    started = time.perf_counter()
    trajectories, prefix = sampler.sample_trajectory(
        batch_size=count,
        sequence_length=config.sequence_length,
        num_steps=config.num_steps,
        fixed_prefix_token_ids=fixed_prefix_token_ids,
        device=device,
        generator=generator,
        microbatch_size=config.microbatch_size,
    )
    expected_shape = (count, config.num_steps + 2, config.sequence_length)
    if tuple(trajectories.shape) != expected_shape:
        raise RuntimeError(
            f"complete trajectory shape {tuple(trajectories.shape)} != {expected_shape}"
        )
    rewards = reward_fn(trajectories[:, -1]).to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(rewards).all().item()):
        raise FloatingPointError("initial trajectory rewards contain NaN or Inf")
    log_weights = torch.full(
        (count,), -math.log(count), dtype=torch.float64, device=device
    )
    root_ancestors = torch.arange(count, device=device)
    genealogy: list[dict[str, Any]] = []
    level_diagnostics: list[dict[str, Any]] = []
    mutation_diagnostics: list[dict[str, Any]] = []
    log_z = 0.0
    pre_final_log_weights = log_weights.clone()

    previous_eta = 0.0
    for level in range(1, config.num_tempering_levels + 1):
        eta = float(config.lambda_final) * level / config.num_tempering_levels
        unnormalized = log_weights + (eta - previous_eta) * rewards
        log_increment = torch.logsumexp(unnormalized, dim=0)
        if not bool(torch.isfinite(log_increment).item()):
            raise FloatingPointError("tempering weights have no finite normalizer")
        log_weights = unnormalized - log_increment
        log_z += float(log_increment.item())
        ess_before = ess_from_log_weights(log_weights)
        max_weight_before = float(log_weights.max().exp().item())
        is_final = level == config.num_tempering_levels
        if is_final:
            pre_final_log_weights = log_weights.clone()
        should_resample = bool(
            (not is_final and ess_before < config.ess_threshold * count)
            or (is_final and config.final_resampling)
        )
        if should_resample:
            parents = resample_indices(
                log_weights, method=config.resampling_method, generator=generator
            )
            trajectories = trajectories.index_select(0, parents)
            rewards = rewards.index_select(0, parents)
            root_ancestors = root_ancestors.index_select(0, parents)
            log_weights.fill_(-math.log(count))
            genealogy.append(
                {
                    "level": int(level),
                    "target_lambda": eta,
                    "parents": [int(value) for value in parents.cpu().tolist()],
                    "final_resampling": bool(is_final),
                    "distinct_root_ancestors": len(
                        set(int(value) for value in root_ancestors.cpu().tolist())
                    ),
                }
            )

        apply_mutation = bool(
            config.mutation_policy == "every_level" or should_resample
        )
        if apply_mutation:
            for sweep in range(config.mcmc_steps_per_level):
                trajectories, rewards, mutation_row = _trajectory_mutation_sweep(
                    trajectories=trajectories,
                    rewards=rewards,
                    sampler=sampler,
                    reward_fn=reward_fn,
                    target_lambda=eta,
                    config=config,
                    generator=generator,
                    level=level,
                    sweep=sweep,
                )
                mutation_diagnostics.append(mutation_row)

        level_diagnostics.append(
            {
                "level": int(level),
                "checkpoint": True,
                "eta": eta,
                "ess": ess_before,
                "ess_before_resampling": ess_before,
                "ess_after_resampling": ess_from_log_weights(log_weights),
                "max_normalized_weight_before_resampling": max_weight_before,
                "max_normalized_weight_after_resampling": float(
                    log_weights.max().exp().item()
                ),
                "resampled": should_resample,
                "mutated": apply_mutation,
                "mutation_sweeps": (
                    config.mcmc_steps_per_level if apply_mutation else 0
                ),
                "terminal": is_final,
                "distinct_root_ancestors": len(
                    set(int(value) for value in root_ancestors.cpu().tolist())
                ),
            }
        )
        previous_eta = eta

    terminal_tokens = trajectories[:, -1]
    if bool((terminal_tokens == sampler.mask_token_id).any().item()):
        raise RuntimeError("terminal trajectory particles still contain masks")
    # Re-score the complete population once so terminal reporting rows and the
    # cached reward vector have identical ordering after grouped MH proposals.
    terminal_rewards = reward_fn(terminal_tokens).to(device=device, dtype=torch.float64)
    _validate_terminal_reward_recheck(
        rewards,
        terminal_rewards,
        backend="trajectory_resample_move",
    )
    rewards = terminal_rewards
    events = event_fn(terminal_tokens).to(device=device, dtype=torch.bool)
    if prefix.numel() and not torch.equal(
        terminal_tokens[:, : prefix.numel()], prefix.expand(count, -1)
    ):
        raise RuntimeError("fixed prefix changed during trajectory SMC")
    post_final_log_weights = log_weights.clone()
    reverse = reverse_snis_component(
        final_log_weights=log_weights,
        rewards=rewards,
        events=events,
        lambda_final=config.lambda_final,
    )
    return SMCComponentResult(
        terminal_tokens=terminal_tokens,
        rewards=rewards,
        events=events,
        final_log_weights=log_weights,
        pre_final_resampling_log_weights=pre_final_log_weights,
        post_final_resampling_log_weights=post_final_log_weights,
        root_ancestors=root_ancestors,
        genealogy=genealogy,
        level_diagnostics=level_diagnostics,
        reverse_snis=reverse,
        log_normalizer_estimate=log_z,
        nfe=sampler.nfe,
        runtime_seconds=time.perf_counter() - started,
        tilted_sampler_type="trajectory_resample_move",
        mutation_diagnostics=mutation_diagnostics,
        trajectory_storage_bytes=(
            trajectories.numel() * trajectories.element_size()
        ),
    )


@torch.inference_mode()
def run_trajectory_waste_free_component(
    *,
    sampler: NativeMDLMSampler,
    fixed_prefix_token_ids: torch.Tensor | None,
    reward_fn: RewardFunction,
    event_fn: EventFunction,
    config: SMCComponentConfig,
    device: torch.device | str,
) -> SMCComponentResult:
    """Waste-free annealed path SMC using the exact rollback-MH kernel.

    At every tempering level, ``M = N / P`` seed paths are resampled from the
    current weighted population.  Each seed starts a length-``P`` Markov chain
    invariant for that current target.  The seed and every subsequent MH state
    are retained, yielding exactly ``M * P = N`` states, which are then
    reweighted by the next incremental potential.  Rejected proposals appear
    as repeated *current* states; the rejected proposal itself is not promoted
    to a particle.
    """

    if config.tilted_sampler_type != "trajectory_waste_free":
        raise ValueError("waste-free backend requires its matching sampler type")
    if getattr(sampler, "predictor", "ddpm") != "ddpm":
        raise ValueError("trajectory waste-free currently requires predictor='ddpm'")

    device = torch.device(device)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    count = config.num_particles
    chain_length = config.waste_free_chain_length
    seed_count = count // chain_length
    sampler.reset_nfe()
    started = time.perf_counter()

    trajectories, prefix = sampler.sample_trajectory(
        batch_size=count,
        sequence_length=config.sequence_length,
        num_steps=config.num_steps,
        fixed_prefix_token_ids=fixed_prefix_token_ids,
        device=device,
        generator=generator,
        microbatch_size=config.microbatch_size,
    )
    expected_shape = (count, config.num_steps + 2, config.sequence_length)
    if tuple(trajectories.shape) != expected_shape:
        raise RuntimeError(
            f"complete trajectory shape {tuple(trajectories.shape)} != {expected_shape}"
        )
    rewards = reward_fn(trajectories[:, -1]).to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(rewards).all().item()):
        raise FloatingPointError("initial trajectory rewards contain NaN or Inf")

    log_weights = torch.full(
        (count,), -math.log(count), dtype=torch.float64, device=device
    )
    root_ancestors = torch.arange(count, device=device)
    genealogy: list[dict[str, Any]] = []
    level_diagnostics: list[dict[str, Any]] = []
    mutation_diagnostics: list[dict[str, Any]] = []
    waste_free_diagnostics: list[dict[str, Any]] = []
    log_z = 0.0
    pre_final_log_weights = log_weights.clone()
    final_chain_ids = torch.empty(0, dtype=torch.long, device=device)
    final_chain_positions = torch.empty(0, dtype=torch.long, device=device)

    previous_eta = 0.0
    for level in range(1, config.num_tempering_levels + 1):
        eta = float(config.lambda_final) * level / config.num_tempering_levels
        is_final = level == config.num_tempering_levels
        seed_resampling_ess = ess_from_log_weights(log_weights)
        seed_resampling_max_weight = float(log_weights.max().exp().item())

        # Algorithm 2 of Dau--Chopin: select M ancestors from the weighted
        # population representing Pi_{r-1}, then retain all P states of M
        # rollback-MH chains invariant for Pi_{r-1}.
        parents = resample_indices(
            log_weights,
            method=config.resampling_method,
            generator=generator,
            num_samples=seed_count,
        )
        current_trajectories = trajectories.index_select(0, parents)
        current_rewards = rewards.index_select(0, parents)
        seed_roots = root_ancestors.index_select(0, parents)

        chain_trajectories = torch.empty(
            (seed_count, chain_length, *current_trajectories.shape[1:]),
            dtype=current_trajectories.dtype,
            device=device,
        )
        chain_rewards = torch.empty(
            (seed_count, chain_length), dtype=torch.float64, device=device
        )
        chain_trajectories[:, 0].copy_(current_trajectories)
        chain_rewards[:, 0].copy_(current_rewards)

        level_rows: list[dict[str, Any]] = []
        for position in range(1, chain_length):
            current_trajectories, current_rewards, mutation_row = (
                _trajectory_mutation_sweep(
                    trajectories=current_trajectories,
                    rewards=current_rewards,
                    sampler=sampler,
                    reward_fn=reward_fn,
                    target_lambda=previous_eta,
                    config=config,
                    generator=generator,
                    level=level,
                    sweep=position - 1,
                )
            )
            mutation_row.update(
                {
                    "waste_free": True,
                    "chain_position": int(position),
                    "chain_length": int(chain_length),
                    "seed_count": int(seed_count),
                }
            )
            mutation_diagnostics.append(mutation_row)
            level_rows.append(mutation_row)
            chain_trajectories[:, position].copy_(current_trajectories)
            chain_rewards[:, position].copy_(current_rewards)

        trajectories = chain_trajectories.reshape(
            count, config.num_steps + 2, config.sequence_length
        )
        rewards = chain_rewards.reshape(count)
        root_ancestors = seed_roots.repeat_interleave(chain_length)
        final_chain_ids = torch.arange(seed_count, device=device).repeat_interleave(
            chain_length
        )
        final_chain_positions = torch.arange(
            chain_length, device=device
        ).repeat(seed_count)
        log_weights = torch.full(
            (count,), -math.log(count), dtype=torch.float64, device=device
        )

        # The retained, unweighted chain population targets Pi_{r-1}.  Apply
        # the next Feynman--Kac potential to obtain the weighted Pi_r sample.
        unnormalized = log_weights + (eta - previous_eta) * rewards
        log_increment = torch.logsumexp(unnormalized, dim=0)
        if not bool(torch.isfinite(log_increment).item()):
            raise FloatingPointError("tempering weights have no finite normalizer")
        log_weights = unnormalized - log_increment
        log_z += float(log_increment.item())
        output_ess = ess_from_log_weights(log_weights)
        output_max_weight = float(log_weights.max().exp().item())
        if is_final:
            pre_final_log_weights = log_weights.clone()

        expanded_parents = parents.repeat_interleave(chain_length)
        distinct_roots = len(
            set(int(value) for value in root_ancestors.cpu().tolist())
        )
        genealogy.append(
            {
                "level": int(level),
                "seed_target_lambda": previous_eta,
                "output_target_lambda": eta,
                "waste_free": True,
                "seed_count": int(seed_count),
                "chain_length": int(chain_length),
                "parents": [int(value) for value in parents.cpu().tolist()],
                "expanded_seed_parents": [
                    int(value) for value in expanded_parents.cpu().tolist()
                ],
                "final_resampling": False,
                "distinct_root_ancestors": distinct_roots,
            }
        )
        proposals = sum(int(row["proposals"]) for row in level_rows)
        accepted = sum(int(row["accepted"]) for row in level_rows)
        waste_free_row = {
            "level": int(level),
            "mh_target_lambda": previous_eta,
            "output_target_lambda": eta,
            "population_size": int(count),
            "seed_count": int(seed_count),
            "chain_length": int(chain_length),
            "mh_transitions_per_chain": int(chain_length - 1),
            "proposals": proposals,
            "accepted": accepted,
            "rejected_self_transitions": proposals - accepted,
            "acceptance_rate": accepted / proposals if proposals else None,
            "log_normalizer_increment": float(log_increment.item()),
            "distinct_seed_particles": len(
                set(int(value) for value in parents.cpu().tolist())
            ),
            "distinct_root_ancestors": distinct_roots,
        }
        waste_free_diagnostics.append(waste_free_row)
        level_diagnostics.append(
            {
                "level": int(level),
                "checkpoint": True,
                "eta": eta,
                "mh_target_lambda": previous_eta,
                "log_normalizer_increment": float(log_increment.item()),
                "ess": output_ess,
                "ess_before_resampling": seed_resampling_ess,
                "ess_after_resampling": float(count),
                "ess_after_incremental_weighting": output_ess,
                "max_normalized_weight_before_resampling": (
                    seed_resampling_max_weight
                ),
                "max_normalized_weight_after_resampling": 1.0 / count,
                "max_normalized_weight_after_incremental_weighting": (
                    output_max_weight
                ),
                "resampled": True,
                "resampling_reason": "waste_free_population_refresh",
                "mutated": True,
                "mutation_sweeps": int(chain_length - 1),
                "waste_free_seed_count": int(seed_count),
                "waste_free_chain_length": int(chain_length),
                "terminal": is_final,
                "distinct_root_ancestors": distinct_roots,
            }
        )
        previous_eta = eta

    terminal_tokens = trajectories[:, -1]
    if bool((terminal_tokens == sampler.mask_token_id).any().item()):
        raise RuntimeError("terminal trajectory particles still contain masks")
    terminal_rewards = reward_fn(terminal_tokens).to(device=device, dtype=torch.float64)
    _validate_terminal_reward_recheck(
        rewards,
        terminal_rewards,
        backend="trajectory_waste_free",
    )
    rewards = terminal_rewards
    events = event_fn(terminal_tokens).to(device=device, dtype=torch.bool)
    if prefix.numel() and not torch.equal(
        terminal_tokens[:, : prefix.numel()], prefix.expand(count, -1)
    ):
        raise RuntimeError("fixed prefix changed during trajectory waste-free SMC")

    post_final_log_weights = log_weights.clone()
    reverse = reverse_snis_component(
        final_log_weights=log_weights,
        rewards=rewards,
        events=events,
        lambda_final=config.lambda_final,
    )
    return SMCComponentResult(
        terminal_tokens=terminal_tokens,
        rewards=rewards,
        events=events,
        final_log_weights=log_weights,
        pre_final_resampling_log_weights=pre_final_log_weights,
        post_final_resampling_log_weights=post_final_log_weights,
        root_ancestors=root_ancestors,
        genealogy=genealogy,
        level_diagnostics=level_diagnostics,
        reverse_snis=reverse,
        log_normalizer_estimate=log_z,
        nfe=sampler.nfe,
        runtime_seconds=time.perf_counter() - started,
        tilted_sampler_type="trajectory_waste_free",
        mutation_diagnostics=mutation_diagnostics,
        trajectory_storage_bytes=trajectories.numel() * trajectories.element_size(),
        waste_free_diagnostics=waste_free_diagnostics,
        waste_free_chain_ids=final_chain_ids,
        waste_free_chain_positions=final_chain_positions,
    )


def run_smc_component(
    *,
    sampler: NativeMDLMSampler,
    fixed_prefix_token_ids: torch.Tensor | None,
    reward_fn: RewardFunction,
    event_fn: EventFunction,
    config: SMCComponentConfig,
    device: torch.device | str,
) -> SMCComponentResult:
    """Dispatch one tilted component to the configured interchangeable backend."""

    if config.tilted_sampler_type == "fk_bootstrap":
        return run_fk_bootstrap_component(
            sampler=sampler,
            fixed_prefix_token_ids=fixed_prefix_token_ids,
            reward_fn=reward_fn,
            event_fn=event_fn,
            config=config,
            device=device,
        )
    if config.tilted_sampler_type == "trajectory_resample_move":
        return run_trajectory_resample_move_component(
            sampler=sampler,
            fixed_prefix_token_ids=fixed_prefix_token_ids,
            reward_fn=reward_fn,
            event_fn=event_fn,
            config=config,
            device=device,
        )
    if config.tilted_sampler_type == "trajectory_waste_free":
        return run_trajectory_waste_free_component(
            sampler=sampler,
            fixed_prefix_token_ids=fixed_prefix_token_ids,
            reward_fn=reward_fn,
            event_fn=event_fn,
            config=config,
            device=device,
        )
    raise ValueError(f"unsupported tilted sampler: {config.tilted_sampler_type}")
