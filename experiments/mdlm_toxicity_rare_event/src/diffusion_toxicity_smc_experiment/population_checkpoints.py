"""Record estimates after each MH move of ONE continuing SMC run.

Observation never draws random numbers, resamples, mutates, or changes weights.
An event count of at least one marks an eligible population. This records
event discovery; it does not stop the ladder or guarantee estimation accuracy.
"""
from __future__ import annotations

import math
import time

import torch

from .pipeline import estimates
from .smc import ess_from_log_weights
from .potentials import ToxicityPotential


def first_event_checkpoint(checkpoints, threshold, minimum_event_particles=1):
    """First eligible population; never select using its estimation error."""
    if int(minimum_event_particles)!=minimum_event_particles or minimum_event_particles<1:
        raise ValueError('Minimum event count must be a positive integer')
    for checkpoint in checkpoints:
        correction = checkpoint['corrections'][str(threshold)]
        if correction['event_particles'] >= minimum_event_particles:
            return checkpoint
    return None


def make_observer(context, study, started, *, num_particles):
    """Return a callback and its recorded populations (no file I/O in timing).

    Runtime includes every preceding observation of this actual continuing
    run, GPU synchronization, and the two corrections. Arrays are copied to
    host memory so later mutations cannot overwrite earlier observations.
    """
    checkpoints = []
    potential = ToxicityPotential(study.get('phi', 'log_probability'))
    actual = getattr(context.functions, 'potential', potential)
    if actual.name != potential.name:
        raise ValueError('Recorder Phi differs from the sampler Phi')

    def observe(**state):
        context.synchronize()
        rewards = state['rewards'].detach().cpu().tolist()
        log_weights = state['normalized_log_weights'].detach().cpu().tolist()
        scores = [potential.probability(phi) for phi in rewards]
        score_source = ('exp(saved float32 log-softmax); checked distance to thresholds'
                        if potential.name == 'log_probability' else
                        'sigmoid(saved classifier logit); checked distance to thresholds')
        tolerance = 2 * float(torch.finfo(torch.float32).eps)
        near_threshold = (
            any(abs(phi - math.log(a)) <= tolerance
                for a in study['thresholds'] for phi in rewards)
            if potential.name == 'log_probability' else
            any(abs(q - a) <= tolerance for a in study['thresholds'] for q in scores))
        if near_threshold:
            # Resolve close decisions using the original classifier output;
            # any extra scoring is part of the measured estimate time.
            context.functions.reward(state['terminal_tokens'])
            context.functions.event(state['terminal_tokens'])
            scores = [float(row['toxic_probability']) for row in context.functions.last_event_rows]
            score_source = 'original classifier probabilities after threshold-near recheck'
        corrections = {
            str(a): estimates(log_weights, rewards, state['log_normalizer_estimate'],
                              scores, a, state['target_lambda'])
            for a in study['thresholds']}
        context.synchronize()
        checkpoints.append({
            'level': state['level'], 'sweep': state['sweep'],
            'lambda': state['target_lambda'], 'num_particles': num_particles,
            'log_z_hat': state['log_normalizer_estimate'],
            'normalized_log_weights': log_weights, 'rewards': rewards,
            'phi': potential.name,
            'scores': scores, 'score_source': score_source,
            'resampled_at_level': state['resampled'],
            'level_is_final': state['level_is_final'],
            'native_transitions_cumulative': state['native_transitions'],
            'smc_ess_fraction': ess_from_log_weights(state['normalized_log_weights']) / num_particles,
            'smc_max_weight': math.exp(max(log_weights)),
            'acceptance_fraction': None if state['mutation_row'] is None else state['mutation_row']['acceptance_rate'],
            'corrections': corrections,
            'runtime_seconds_cumulative': time.perf_counter() - started,
            'population_policy': 'actual continuing run, including intermediate resampling',
        })

    return observe, checkpoints
