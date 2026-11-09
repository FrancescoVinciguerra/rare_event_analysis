"""Explicit fixed tempering grids for the existing full-trajectory algorithm.

Materialized from smc.run_trajectory_resample_move_component by
tmp/build_exploration_sources.py. Changes: explicit validated grid and saved
normalizer-update inputs. Same proposal, resampler, MH rule and operation order.
"""
from __future__ import annotations
import math
import time
from typing import Callable, Sequence
import torch
from .ancestral import NativeMDLMSampler
from .smc import (SMCComponentConfig, SMCComponentResult, RewardFunction,
    EventFunction, ess_from_log_weights, resample_indices, reverse_snis_component,
    _trajectory_mutation_sweep, _validate_terminal_reward_recheck)

def validate_schedule(schedule, config):
    values=tuple(float(x) for x in schedule)
    if len(values)!=config.num_tempering_levels:
        raise ValueError('Schedule length differs from num_tempering_levels')
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('Schedule must contain finite levels')
    if not all(a<b for a,b in zip((0.,)+values[:-1],values)):
        raise ValueError('Schedule must increase strictly from zero')
    if values[-1]!=config.lambda_final:
        raise ValueError('Schedule endpoint differs from lambda_final')
    if config.final_resampling:
        raise ValueError('This exploration excludes final resampling')
    if config.mutation_policy!='every_level':
        raise ValueError('This exploration mutates at every level')
    return values

@torch.inference_mode()
def run_scheduled_trajectory_component(
    *,
    sampler: NativeMDLMSampler,
    fixed_prefix_token_ids: torch.Tensor | None,
    reward_fn: RewardFunction,
    event_fn: EventFunction,
    config: SMCComponentConfig,
    device: torch.device | str,
    schedule: Sequence[float],
    population_observer: Callable[..., None] | None = None,
) -> SMCComponentResult:
    """Annealed path-space SMC with exact native-decoder rollback MH moves.

    The state is the full MDLM trajectory.  Adjacent SMC targets are
    ``P0(path) * exp(eta_k * reward(x0))``.  A mutation returns to a checkpoint
    actually visited by that path and regenerates only its future with the
    unchanged base decoder.  Forward/reverse suffix probabilities therefore
    cancel, leaving the exact acceptance ratio ``exp(eta_k * delta_reward)``.
    """

    schedule = validate_schedule(schedule, config)
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
    native_transitions = count * (config.num_steps + 1)
    if population_observer is not None:
        population_observer(level=0, sweep=0, target_lambda=0.,
            terminal_tokens=trajectories[:, -1], rewards=rewards,
            normalized_log_weights=log_weights, log_normalizer_estimate=log_z,
            native_transitions=native_transitions, resampled=False,
            level_is_final=False, mutation_row=None)

    previous_eta = 0.0
    for level, eta in enumerate(schedule, 1):
        trace_old_log_weights = log_weights.detach().cpu().tolist()
        trace_rewards = rewards.detach().cpu().tolist()
        trace_log_z_before = log_z
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
                native_transitions += int(mutation_row['rollback_particle_steps']) + int(mutation_row['proposals'])
                if population_observer is not None:
                    population_observer(level=level, sweep=sweep + 1,
                        target_lambda=eta, terminal_tokens=trajectories[:, -1],
                        rewards=rewards, normalized_log_weights=log_weights,
                        log_normalizer_estimate=log_z,
                        native_transitions=native_transitions,
                        resampled=should_resample, level_is_final=is_final,
                        mutation_row=mutation_row)

        level_diagnostics.append(
            {
                "level": int(level),
                "checkpoint": True,
                "eta": eta,
                "log_normalizer_increment": float(log_increment.item()),
                "log_normalizer_estimate": log_z,
                "log_z_before_weight": trace_log_z_before,
                "previous_normalized_log_weights": trace_old_log_weights,
                "rewards_before_mutation": trace_rewards,
                "delta_lambda": eta - previous_eta,
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
