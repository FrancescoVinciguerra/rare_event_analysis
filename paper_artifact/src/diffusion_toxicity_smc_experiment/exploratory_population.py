"""GPU population measurement using the scheduled version of the SAME kernel.

The explicit fixed schedule can record each mutation as an optional checkpoint.
"""
import torch
import time
from .exploratory_kernel import run_scheduled_trajectory_component
from .pipeline import component_config, finalize_population
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
    output = finalize_population(context, c, study['thresholds'], result, started)
    output['phi'] = potential.manifest()
    if checkpoints is not None:
        from .population_checkpoints import first_event_checkpoint
        # Include terminal score validation and reverse correction in the last checkpoint.
        checkpoints[-1]['runtime_seconds_cumulative'] = output['summary']['runtime_seconds']
        checkpoints[-1]['corrections'] = output['corrections']
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
