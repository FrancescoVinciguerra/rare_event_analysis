"""GPU population measurement using the scheduled version of the SAME kernel.

Population recording is materialized from camera_ready_gpu.population; only
the kernel call receives the explicit fixed schedule. No models load on import.
"""
import torch
import time
from .camera_ready_gpu import component_config
from .exploratory_kernel import run_scheduled_trajectory_component
from .low_budget import estimates
from .smc import ess_from_log_weights
from .potentials import potential_from_config

@torch.inference_mode()
def population(context,config,study,seed):
    c=component_config(config,seed); f=context.functions
    potential = potential_from_config(config)
    if study.get('phi', 'log_probability') != potential.name:
        raise ValueError('Study Phi differs from the population configuration')
    if getattr(f, 'potential', potential).name != potential.name:
        raise ValueError('Context Phi differs from the population configuration')
    context.synchronize(); started=time.perf_counter()
    observer = None
    checkpoints = None
    if study.get('record_population_checkpoints', False):
        from .population_checkpoints import make_observer
        observer, checkpoints = make_observer(context, study, started, num_particles=c.num_particles)
    result=run_scheduled_trajectory_component(schedule=config['_exploration']['schedule'],sampler=context.sampler,fixed_prefix_token_ids=None,
        reward_fn=f.reward,event_fn=f.event,config=c,device=context.device,
        population_observer=observer)
    # Include final transfers and BOTH threshold corrections in the measured estimate.
    log_weights=result.final_log_weights.detach().cpu().tolist()
    rewards=result.rewards.detach().cpu().tolist()
    scores=[float(r['toxic_probability']) for r in f.last_event_rows]
    corrections={str(a):estimates(log_weights,rewards,result.log_normalizer_estimate,
        scores,a,c.lambda_final) for a in study['thresholds']}
    context.synchronize(); elapsed=time.perf_counter()-started
    if checkpoints is not None:
        # The final time also includes the terminal score recheck and corrections.
        checkpoints[-1]['runtime_seconds_cumulative'] = elapsed
        checkpoints[-1]['corrections'] = corrections
    roots=result.root_ancestors.cpu().tolist(); tokens=result.terminal_tokens.cpu().tolist()
    proposals=sum(int(m['proposals']) for m in result.mutation_diagnostics)
    accepted=sum(int(m['accepted']) for m in result.mutation_diagnostics)
    ddpm=c.num_particles*c.num_steps+sum(int(m['rollback_particle_steps']) for m in result.mutation_diagnostics)
    summary={'num_particles':c.num_particles,'seed':int(seed),'lambda':c.lambda_final,
        'log_normalizer_estimate':result.log_normalizer_estimate,
        'total_decoder_particle_steps':ddpm,'mcmc_proposals':proposals,
        'mcmc_acceptance_rate':accepted/proposals if proposals else 0.,
        'distinct_root_ancestors':len(set(roots)),
        'distinct_terminal_sequences':len({tuple(t) for t in tokens}),
        'final_smc_ess':ess_from_log_weights(result.final_log_weights),
        'runtime_seconds':elapsed,'nfe':result.nfe,
        'resampling_count':len(result.genealogy)}
    output = {'summary':summary,'phi':potential.manifest(),'corrections':corrections,
        'particles':[{'root_ancestor':int(roots[i]),'token_ids':tokens[i],
            'reward':rewards[i],'event_toxic_probability':scores[i],
            'final_log_weight':log_weights[i]} for i in range(c.num_particles)],
        'genealogy':result.genealogy,'level_diagnostics':result.level_diagnostics,
        'mutation_diagnostics':result.mutation_diagnostics,
        'timing_includes':'initial generation, score, weights, resampling, rollback, MH, both corrections; excludes model load, warmup and file IO'}
    if checkpoints is not None:
        from .population_checkpoints import first_event_checkpoint
        output['population_checkpoints'] = checkpoints
        output['first_event'] = {}
        for a in study['thresholds']:
            cap = study.get('report_max_lambda', {}).get(a, c.lambda_final)
            eligible = [cp for cp in checkpoints if cp['lambda'] <= cap]
            first = first_event_checkpoint(eligible, a, study.get('minimum_event_particles', 1))
            output['first_event'][str(a)] = {
                'minimum_event_particles': study.get('minimum_event_particles', 1),
                'lambda_cap': cap,
                'first_checkpoint_index': next((i for i, cp in enumerate(checkpoints) if cp is first), None),
                'detected': first is not None}
        output['checkpoint_policy'] = ('One continuing run; record every MH move; '
            'no cross-level averaging; no terminal forks; at least one event marks an eligible population; '
            'record every population and continue the ladder, without stopping at first discovery')
    return output
