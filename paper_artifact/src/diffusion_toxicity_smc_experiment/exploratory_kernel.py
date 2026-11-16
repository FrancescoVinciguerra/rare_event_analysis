"""Explicit fixed schedules for the shared RollBack–MH tempering engine."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from .ancestral import NativeMDLMSampler
from .smc import (
    EventFunction,
    RewardFunction,
    SMCComponentConfig,
    SMCComponentResult,
    _run_trajectory_tempering,
)


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
    """Run a validated explicit ladder and retain its weight-update ledger."""

    schedule = validate_schedule(schedule, config)
    return _run_trajectory_tempering(
        sampler=sampler,
        fixed_prefix_token_ids=fixed_prefix_token_ids,
        reward_fn=reward_fn,
        event_fn=event_fn,
        config=config,
        device=device,
        schedule=schedule,
        record_weight_updates=True,
        population_observer=population_observer,
    )
