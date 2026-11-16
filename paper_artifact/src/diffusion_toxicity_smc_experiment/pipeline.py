"""Shared model setup, reverse-tilt estimates, and SMC weight checks."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .potentials import potential_from_config
from .smc import SMCComponentConfig, ess_from_log_weights


def component_config(config,seed):
    s=config['smc']; m=s['tilted_sampler']; d=config['decoder']
    return SMCComponentConfig(num_particles=s['num_particles'],num_steps=d['denoising_steps'],
        sequence_length=d['sequence_length'],lambda_final=s['tilted_lambdas'][0],
        ess_threshold=s['ess_threshold'],resampling_method=s['resampling_method'],
        reward_frequency=s['reward_interval'],beta_start_fraction=0.,final_resampling=False,
        microbatch_size=s['microbatch_size'],seed=int(seed),tilted_sampler_type=m['type'],
        num_tempering_levels=m['num_tempering_levels'],mcmc_steps_per_level=m['mcmc_steps_per_level'],
        mutation_policy=m['mutation_policy'],rollback_steps=tuple(m['rollback_steps']),
        rollback_probabilities=tuple(m['rollback_probabilities']))


class Context:
    def __init__(self,config,*,local_files_only=True):
        from .ancestral import NativeMDLMSampler
        from .evaluation import ToxicityEvaluator, UnconditionalToxicityFunctions
        from .runtime import load_generator_runtime,load_evaluator_runtime
        self.generator=load_generator_runtime(config,local_files_only=local_files_only)
        self.evaluator_runtime=load_evaluator_runtime(config,local_files_only=local_files_only)
        e=config['evaluation']; numerical=e['numerical_batching']; d=config['decoder']
        self.device=self.generator.device
        self.evaluator=ToxicityEvaluator(self.evaluator_runtime.classifier,
            self.evaluator_runtime.classifier_tokenizer,self.device,positive_label_index=1,
            batch_size=e['batch_size'],padding=numerical['padding'],
            pad_final_batch_to_batch_size=numerical['pad_final_batch_to_batch_size'])
        self.functions=UnconditionalToxicityFunctions(generator_tokenizer=self.generator.tokenizer,
            evaluator=self.evaluator,reward_completion_tokens=e['reward']['completion_tokens'],
            event_completion_tokens=e['event']['completion_tokens'],
            reward_max_length=e['reward']['classifier_max_length'],event_max_length=e['event']['classifier_max_length'],
            event_text_scope=e['event']['text_scope'],
            completion_trimming_tokenizer=self.evaluator_runtime.completion_trimming_tokenizer,
            phi=potential_from_config(config).name)
        self.sampler=NativeMDLMSampler(self.generator.model,mask_token_id=d['mask_token_id'],
            predictor=d['predictor'],noise_epsilon=d['noise_schedule']['epsilon'],
            sampling_epsilon=d['sampling_epsilon'],final_noise_removal=d['final_noise_removal'],
            final_noise_removal_method=d['final_noise_removal_method'])
        self.config=config

    def synchronize(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)


def estimates(log_weights, rewards, log_z, scores, threshold, lambda_final):
    """Both LaTeX estimators, in float64 log space on one population."""
    log_weights = torch.as_tensor(log_weights,dtype=torch.float64)
    rewards = torch.as_tensor(rewards,dtype=torch.float64)
    scores = torch.as_tensor(scores,dtype=torch.float64)
    if not torch.isclose(torch.logsumexp(log_weights,0),torch.tensor(0.,dtype=torch.float64),atol=1e-9):
        raise ValueError('Terminal SMC weights must already be normalized')
    reverse = log_weights - lambda_final * rewards
    denominator = torch.logsumexp(reverse,0)
    event = scores >= threshold
    numerator = torch.logsumexp(torch.where(event,reverse,-torch.inf),0)
    return {'smc': float(torch.exp(log_z+numerator)),
        'snis': float(torch.exp(numerator-denominator)),
        'event_particles': int(event.sum()),
        'reverse_ess_fraction': float(torch.exp(-torch.logsumexp(2*(reverse-denominator),0))/len(scores)),
        'reverse_max_weight': float(torch.exp(reverse-denominator).max()),
        'reverse_mass': float(torch.exp(log_z+denominator))}


def finalize_population(context, component, thresholds, result, started):
    """Apply the reverse correction and assemble the common population record."""
    log_weights = result.final_log_weights.detach().cpu().tolist()
    rewards = result.rewards.detach().cpu().tolist()
    scores = [float(row["toxic_probability"]) for row in context.functions.last_event_rows]
    corrections = {
        str(threshold): estimates(
            log_weights, rewards, result.log_normalizer_estimate, scores, threshold, component.lambda_final
        )
        for threshold in thresholds
    }
    context.synchronize()
    elapsed = time.perf_counter() - started

    roots = result.root_ancestors.cpu().tolist()
    tokens = result.terminal_tokens.cpu().tolist()
    proposals = sum(int(move["proposals"]) for move in result.mutation_diagnostics)
    accepted = sum(int(move["accepted"]) for move in result.mutation_diagnostics)
    rollback_steps = sum(int(move["rollback_particle_steps"]) for move in result.mutation_diagnostics)
    summary = {
        "num_particles": component.num_particles,
        "seed": component.seed,
        "lambda": component.lambda_final,
        "log_normalizer_estimate": result.log_normalizer_estimate,
        "total_decoder_particle_steps": component.num_particles * component.num_steps + rollback_steps,
        "mcmc_proposals": proposals,
        "mcmc_acceptance_rate": accepted / proposals if proposals else 0.,
        "distinct_root_ancestors": len(set(roots)),
        "distinct_terminal_sequences": len({tuple(sequence) for sequence in tokens}),
        "final_smc_ess": ess_from_log_weights(result.final_log_weights),
        "runtime_seconds": elapsed,
        "nfe": result.nfe,
        "resampling_count": len(result.genealogy),
    }
    return {
        "summary": summary,
        "corrections": corrections,
        "particles": [
            {"root_ancestor": int(root), "token_ids": sequence, "reward": reward,
             "event_toxic_probability": score, "final_log_weight": log_weight}
            for root, sequence, reward, score, log_weight in zip(roots, tokens, rewards, scores, log_weights, strict=True)
        ],
        "genealogy": result.genealogy,
        "level_diagnostics": result.level_diagnostics,
        "mutation_diagnostics": result.mutation_diagnostics,
        "timing_includes": "initial generation, score, weights, resampling, rollback, MH, both corrections; "
                           "excludes model load, warmup and file IO",
    }


def audit_population(p,config):
    """Reconstruct every normalizer increment from its saved pre-mutation inputs."""
    c=config['_exploration'];n=config['smc']['num_particles'];previous=0.;previous_z=0.
    expected_old=np.full(n,-math.log(n))
    for level,grid in zip(p['level_diagnostics'],c['schedule']):
        old=np.array(level['previous_normalized_log_weights']);rewards=np.array(level['rewards_before_mutation'])
        if not np.allclose(old,expected_old,rtol=0,atol=1e-10):raise AssertionError('Mutation changed weights or resampling reset is wrong')
        x=old+(grid-previous)*rewards;peak=x.max();increment=float(peak+np.log(np.exp(x-peak).sum()))
        if not math.isclose(increment,level['log_normalizer_increment'],abs_tol=1e-11): raise AssertionError('G evaluated on wrong state or wrong normalizer')
        if not math.isclose(previous_z+increment,level['log_normalizer_estimate'],abs_tol=1e-10): raise AssertionError('Z not updated at every level')
        if not math.isclose(previous_z,level['log_z_before_weight'],abs_tol=1e-10): raise AssertionError('Wrong incoming Z')
        expected=not level['terminal'] and level['ess_before_resampling']<.5*n
        if bool(level['resampled'])!=expected: raise AssertionError('Wrong ESS or terminal resampling policy')
        if level['mutation_sweeps']!=c['moves_per_level']: raise AssertionError('Wrong move count')
        expected_old=np.full(n,-math.log(n)) if expected else x-increment
        previous=grid;previous_z=level['log_normalizer_estimate']
    if len(p['level_diagnostics'])!=len(c['schedule']): raise AssertionError('Missing levels')
    if not math.isclose(previous_z,p['summary']['log_normalizer_estimate'],abs_tol=1e-10): raise AssertionError('Terminal Z mismatch')
    if sum(x['proposals'] for x in p['mutation_diagnostics'])!=n*len(c['schedule'])*c['moves_per_level']: raise AssertionError('Missing proposals')
    if 'particles' in p:
        if not np.allclose([x['final_log_weight'] for x in p['particles']],expected_old,rtol=0,atol=1e-10):raise AssertionError('Terminal mutation changed weights')
    steps=config.get('decoder',{}).get('denoising_steps')
    if steps is not None:
        expected_steps=n*steps+sum(x['rollback_particle_steps'] for x in p['mutation_diagnostics'])
        if expected_steps!=p['summary']['total_decoder_particle_steps']:raise AssertionError('Rejected proposal cost omitted')
    return True
