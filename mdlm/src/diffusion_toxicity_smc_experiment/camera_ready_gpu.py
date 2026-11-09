"""Paired native-transition and wall-clock comparisons on one allocated GPU."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import time

import torch

from .configuration import fingerprint
from .low_budget import crude_seed, study_configs, transition_budget
from .pipeline import Context, component_config, finalize_population
from .smc import run_smc_component
from .storage import atomic_write_json, initialize_manifest


@torch.inference_mode()
def population(context,config,study,seed):
    c=component_config(config,seed); f=context.functions
    context.synchronize(); started=time.perf_counter()
    result=run_smc_component(sampler=context.sampler,fixed_prefix_token_ids=None,
        reward_fn=f.reward,event_fn=f.event,config=c,device=context.device)
    return finalize_population(context, c, study['thresholds'], result, started)


@torch.inference_mode()
def crude(context,seed,thresholds,*,count,policy='matched_transitions',target_seconds=None):
    """Fixed sample count, chosen before observing ANY baseline sample.

    Wall-clock counts use independent timing calibration, not a stopping rule
    depending on the runtime/content of the baseline's own observations.
    """
    if count<1: raise ValueError('Empty crude budget')
    g=torch.Generator(device=context.device.type).manual_seed(int(seed))
    f=context.functions; d=context.config['decoder']; s=context.config['smc']
    context.sampler.reset_nfe()
    context.synchronize(); started=time.perf_counter()
    paths,_=context.sampler.sample_trajectory(batch_size=count,sequence_length=d['sequence_length'],
        num_steps=d['denoising_steps'],fixed_prefix_token_ids=None,device=context.device,
        generator=g,microbatch_size=s['microbatch_size'])
    f.reward(paths[:,-1]); f.event(paths[:,-1])
    scores=[float(r['toxic_probability']) for r in f.last_event_rows]
    estimates_out={str(a):sum(q>=a for q in scores)/len(scores) for a in thresholds}
    context.synchronize(); elapsed=time.perf_counter()-started
    return {'scores':scores,'seed':int(seed),'n':len(scores),'native_transitions':33*len(scores),
        'runtime_seconds':elapsed,'nfe':context.sampler.nfe,'estimates':estimates_out,
        'target_seconds':target_seconds,
        'overshoot_seconds':max(0.,elapsed-target_seconds) if target_seconds is not None else None,
        'policy':policy,'sample_count_chosen_before_generation':True}


def wall_clock_count(smc_seconds,seconds_per_crude_sample):
    if smc_seconds<=0 or seconds_per_crude_sample<=0: raise ValueError('Invalid timing calibration')
    return max(1,math.floor(smc_seconds/seconds_per_crude_sample))


def rows_for(population_data,baseline,study):
    summary=population_data['summary']; cost=transition_budget(summary); n=summary['num_particles']
    rows=[]
    for a,hits in zip(study['thresholds'],study['reference']['discoveries']):
        event_roots={p['root_ancestor'] for p in population_data['particles'] if p['event_toxic_probability']>=a}
        rows.append({'N':n,'seed':summary['seed'],'crude_seed':baseline['seed'],'threshold':a,
            'comparison':baseline['policy'],'lambda_final':study['lambda_final'],
            'p_reference':hits/study['reference']['samples'],
            **cost,'crude_n':baseline['n'],'crude_native_transitions':baseline['native_transitions'],
            'crude':baseline['estimates'][str(a)],
            'crude_event_particles':sum(q>=a for q in baseline['scores']),
            'smc_runtime_seconds':summary['runtime_seconds'],'crude_runtime_seconds':baseline['runtime_seconds'],
            'wall_clock_ratio':baseline['runtime_seconds']/summary['runtime_seconds'],
            'clock_overshoot_seconds':baseline['overshoot_seconds'],
            'log_z_hat':summary['log_normalizer_estimate'],'smc_ess_fraction':summary['final_smc_ess']/n,
            'distinct_root_ancestors':summary['distinct_root_ancestors'],
            'distinct_terminal_sequences':summary['distinct_terminal_sequences'],
            'acceptance_fraction':summary['mcmc_acceptance_rate'],
            'event_distinct_roots':len(event_roots),**population_data['corrections'][str(a)]})
    return rows


def run(study_path:Path,output_root:Path):
    from .low_budget import report
    study,configs=study_configs(study_path); root=output_root/study['output_subdirectory']
    initialize_manifest(root,{'study_id':fingerprint(study),'study':study,
        'resolved_configs':configs,'runner_version':'paired_transition_clock_v2',
        'runtime_source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['smc.py','ancestral.py','evaluation.py','run_smc.py','runtime.py',
                         'camera_ready_gpu.py','low_budget.py','pipeline.py']},
        'budget_definition':study['budget_definition']})
    schedule=[(config,int(config['smc']['base_seed'])+1000000*i)
        for config in configs for i in range(study['independent_runs'])]
    def final_path(config,seed): return root/'runs'/f'N{config["smc"]["num_particles"]}_seed{seed}.json'
    if all(final_path(c,s).exists() for c,s in schedule): report(study_path,output_root); return
    context=Context(configs[0])
    warm_started=time.perf_counter()
    population(context,configs[0],study,9824711)
    crude(context,9824712,study['thresholds'],count=16)
    calibrations=[crude(context,seed,study['thresholds'],count=128) for seed in [9824713,9824714,9824715]]
    seconds_per_sample=sorted(b['runtime_seconds']/b['n'] for b in calibrations)[1]
    atomic_write_json(root/'warmup.json',{'seconds':time.perf_counter()-warm_started,
        'seeds':[9824711,9824712,9824713,9824714,9824715],
        'crude_calibration_seconds':[b['runtime_seconds'] for b in calibrations],
        'crude_calibration_n':128,'seconds_per_crude_sample':seconds_per_sample,
        'excluded_from_estimates_and_measured_run_times':True})
    for config,seed in schedule:
        dest=final_path(config,seed)
        if dest.exists(): continue
        context.config=config
        p_path=root/'populations'/dest.name
        if p_path.exists(): p=json.loads(p_path.read_text(encoding='utf-8'))
        else:
            p=population(context,config,study,seed)
            atomic_write_json(p_path,p)
        budget=transition_budget(p['summary'])
        baselines={}
        for policy,stream in [('matched_transitions',924071),('matched_wall_clock',924072)]:
            b_path=root/'baselines'/policy/dest.name
            if b_path.exists(): b=json.loads(b_path.read_text(encoding='utf-8'))
            else:
                # Independent streams; the same two thresholds share each baseline.
                b=crude(context,crude_seed(seed,stream),study['thresholds'],
                    count=budget['crude_n'] if policy=='matched_transitions' else wall_clock_count(p['summary']['runtime_seconds'],seconds_per_sample),
                    policy=policy,target_seconds=p['summary']['runtime_seconds'] if policy=='matched_wall_clock' else None)
                atomic_write_json(b_path,b)
            baselines[policy]=b
        atomic_write_json(dest,{'summary':p['summary'],'rows':[
            r for b in baselines.values() for r in rows_for(p,b,study)],
            'population_file':str(p_path),'no_cross_tilt_aggregation':True})
        # Rebuild only the small CSVs after every completed population. SVG/HTML
        # are regenerated at job end, so checkpoint I/O is bounded.
        report(study_path,output_root,render=False)
        print(f'COMPLETED N={config["smc"]["num_particles"]} seed={seed} SMC_seconds={p["summary"]["runtime_seconds"]:.3f}',flush=True)
    report(study_path,output_root)
