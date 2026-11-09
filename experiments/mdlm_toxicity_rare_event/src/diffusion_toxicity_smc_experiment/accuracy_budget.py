"""Measured-cost screening for useful probability estimates, not just events.

Reuse the tested complete-trajectory RollBack kernel and the two estimators.
Each configuration is a genuine terminal run; it never resamples at its final
weight update. All thresholds share the SMC population and crude population.
No model load or GPU computation occurs on import or during the plan command.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np

from .configuration import decoder_id, evaluator_id, fingerprint, load_yaml
from .exploratory import audit_population
from .low_budget import crude_seed
from .storage import atomic_write_json, initialize_manifest
from .potentials import ToxicityPotential


def study_configs(path):
    path=Path(path);study=load_yaml(path);base=load_yaml(path.parent/study['base_config'])
    if (decoder_id(base),evaluator_id(base))!=(study['reference']['decoder_id'],study['reference']['evaluator_id']):
        raise ValueError('Frozen P/evaluator differ from the reference')
    scaling = study.get('particle_scaling', False)
    ultra_low = study.get('ultra_low_particle_budget', False)
    deeper_tilt = study.get('deeper_tilt_budget', False)
    extreme_tail = study.get('extreme_tail_scaling', False)
    if scaling:
        if extreme_tail:
            expected_thresholds = [.9995]
        else:
            expected_thresholds = [.99, .995] if ultra_low or deeper_tilt else [.99, .995, .999]
    else:
        expected_thresholds = [.99, .999] if study.get('same_run_ladder') else [.9, .99, .999]
    if study['thresholds']!=expected_thresholds or study['comparison']!='measured_matched_wall_clock':
        raise ValueError('Thresholds differ from this frozen study or measured clock comparison is missing')
    if len(study['reference']['discoveries'])!=len(study['thresholds']):
        raise ValueError('Every threshold needs its own independent reference count')
    if scaling:
        checkpoint_tilts = study.get('checkpoint_tilts')
        if extreme_tail:
            if study.get('terminal_particle_scaling', False):
                expected_cases = {(n, 6.5) for n in (320, 384, 448, 512)}
                expected_checkpoints = {.9995: [6.5]}
            else:
                expected_cases = {(256, 9.)}
                expected_checkpoints = {.9995: [k / 2 for k in range(1, 19)]}
        elif deeper_tilt:
            expected_cases = {(24, 2.), (32, 2.)}
            expected_checkpoints = {.99: [.5, 1.], .995: [1., 1.5, 2.]}
        elif ultra_low:
            expected_cases = {(32, 1.5), (48, 1.5)}
            expected_checkpoints = {.99: [.5], .995: [1.]}
        elif checkpoint_tilts:
            expected_cases = {(64, 3.), (96, 3.)}
            expected_checkpoints = {.99: [.5], .995: [1.], .999: [.5, 3.]}
        else:
            expected_cases = {(n, tilt) for n in (128, 160, 192, 256) for tilt in (.5, 2.5)}
            expected_checkpoints = None
        cases = study['configurations']
        if len(cases) != len(expected_cases) or {(c['N'], c['schedule'][-1]) for c in cases} != expected_cases:
            raise ValueError('Particle-scaling configurations differ from the frozen design')
        if study['plot_thresholds'] != expected_thresholds or study['phi'] != 'logit':
            raise ValueError('Particle scaling requires the logit potential and its event thresholds')
        if checkpoint_tilts != expected_checkpoints or (expected_checkpoints
                and not study.get('record_population_checkpoints')):
            raise ValueError('The low-budget study requires its paired checkpoints')
    elif study['plot_thresholds']!=[.99,.999] or (not study.get('same_run_ladder') and study['control_threshold']!=.9):
        raise ValueError('Keep 0.9 as control; focus rare-event plots on 0.99 and 0.999')
    if int(study['independent_runs'])<1:raise ValueError('No independent runs')
    configs=[];ids=set()
    for i,case in enumerate(study['configurations']):
        grid=[float(x) for x in case['schedule']]
        if case['id'] in ids or not grid or any(not math.isfinite(x) for x in grid):
            raise ValueError('Duplicate configuration or invalid grid')
        if scaling:
            if grid != [.5*k for k in range(1, len(grid)+1)] or case['moves_per_level'] != 4:
                raise ValueError('Particle scaling fixes half-step schedules and four MH moves')
        elif study.get('same_run_ladder'):
            if grid != [.5*k for k in range(1,17)] or case['moves_per_level'] != 4:
                raise ValueError('Same-run study fixes 0.5 increments to 8 and four MH moves')
            if case['N'] != 128 or not study.get('record_population_checkpoints'):
                raise ValueError('Same-run study records every MH population for N=128')
        elif study.get('gradual_budget_screen'):
            if grid != [.5*k for k in range(1,len(grid)+1)] or case['moves_per_level'] not in (1,2,4):
                raise ValueError('Gradual screening requires increments of 0.5 and 1, 2, or 4 MH moves')
        elif study.get('strong_tilt_screen'):
            levels = {16: 5, 256: 6}.get(case['N'])
            if case['moves_per_level'] != 4 or len(grid) != levels or not 0 < grid[-1] <= 8 or not np.allclose(
                grid, np.linspace(0, grid[-1], len(grid) + 1)[1:]
            ):
                raise ValueError('Strong-tilt screen requires its fixed-size linear schedules')
        else:
            if case['moves_per_level'] != 4 or grid != [.5*k for k in range(1,len(grid)+1)]:
                raise ValueError('The original screening fixes half-step schedules')
            if case['N'] not in [16,32,64] or not .5<=grid[-1]<=3.:
                raise ValueError('This pilot uses N=16/32/64 and final tilts up to 3')
        ids.add(case['id']);c=copy.deepcopy(base);s=c['smc'];m=s['tilted_sampler']
        s.update(num_particles=int(case['N']),independent_runs=int(study['independent_runs']),
            num_independent_runs=int(study['independent_runs']),tilted_lambdas=[grid[-1]],
            tilted_component_weights_a=[1.],diagnostic_lambdas=[],final_resampling=False,
            base_seed=int(study['base_seed'])+i*10000)
        m.update(num_tempering_levels=len(grid),mcmc_steps_per_level=case['moves_per_level'],mutation_policy='every_level')
        if s['resampling_method']!='systematic' or s['ess_threshold']!=.5:
            raise ValueError('The ESS/systematic policy changed')
        if m['rollback_steps']!=[4,8,16,24,32] or m['rollback_probabilities']!=[.2]*5:
            raise ValueError('Native MDLM rollback depths must remain unchanged')
        if 'phi' in study:
            s['phi'] = ToxicityPotential(study['phi']).name
        c['_exploration']={**case,'schedule':grid,'label':f"{case['id']}, tilt={grid[-1]:g}, MH={case['moves_per_level']}"}
        configs.append(c)
    return study,configs


def assigned_runs(study,configs,worker_id,workers):
    if not 1<=workers<=3 or not 0<=worker_id<workers:raise ValueError('Invalid GPU worker')
    # Endpoints use distinct seeds, including when N is the same.
    return [(c,c['smc']['base_seed']+1000000*i)
        for i in range(study['independent_runs']) if i%workers==worker_id for c in configs]


def plan(path):
    study,configs=study_configs(path);cases=[]
    for c in configs:
        s=c['smc'];m=s['tilted_sampler'];case=c['_exploration'];n=s['num_particles']
        expected=n*(33+len(case['schedule'])*case['moves_per_level']*17.8)
        cases.append({'id':case['id'],'N':n,'schedule':[0.]+case['schedule'],
            'moves_per_level':case['moves_per_level'],'independent_runs':study['independent_runs'],
            'base_seed':s['base_seed'],'expected_native_transitions':expected,
            'expected_equivalent_crude_paths':expected/33,
            'time_budget':'Complete synchronized SMC duration; independently calibrated MC count, actual MC duration recorded'})
    return {'status':'prepared_not_submitted','study':study,'configurations':cases,
        'total_SMC_runs':len(configs)*study['independent_runs'],
        'new_algorithms':False,'no_final_resampling':True,'averaging_across_tilts':False}


def summarize(rows,study):
    out=[]
    for case,a in sorted({(r['configuration'],r['threshold']) for r in rows}):
        group=[r for r in rows if (r['configuration'],r['threshold'])==(case,a)];first=group[0]
        for method in ['smc','snis','crude']:
            x=np.array([r[method]/r['p_reference'] for r in group]);metric=np.log10(1+abs(x-1))
            # All estimates, including zeros; no arbitrary accuracy cutoff.
            out.append({k:first[k] for k in ['configuration','N','levels','moves_per_level','lambda_final','threshold','p_reference']}|
                {'method':method,'runs':len(x),'zero_count':int((x==0).sum()),'zero_percent':float(100*(x==0).mean()),
                 'mean_estimate':float(x.mean()*first['p_reference']),
                 'median_estimate':float(np.median(x)*first['p_reference']),
                 'mean_ratio':float(x.mean()),'median_ratio':float(np.median(x)),
                 'mean_log_relative_error':float(metric.mean()),'median_log_relative_error':float(np.median(metric)),
                 'mean_SMC_seconds':float(np.mean([r['smc_runtime_seconds'] for r in group])),
                 'mean_MC_seconds':float(np.mean([r['crude_runtime_seconds'] for r in group])),
                 'mean_crude_paths':float(np.mean([r['crude_n'] for r in group])),
                 'clock_outside_tolerance_runs':sum(not r['clock_within_tolerance'] for r in group)})
            if study.get('strong_tilt_screen') or study.get('gradual_budget_screen'):
                positive = x > 0
                out[-1].update(
                    positive_count=int(positive.sum()),
                    positive_mean_log_relative_error=float(metric[positive].mean()) if positive.any() else None,
                    positive_median_log_relative_error=float(np.median(metric[positive])) if positive.any() else None,
                )
            if study.get('gradual_budget_screen'):
                out[-1].update(
                    N_min=min(r['N'] for r in group), N_max=max(r['N'] for r in group),
                    target_MC_paths=first['target_MC_paths'],
                    target_threshold=first['target_threshold'],
                    budget_fraction=first['budget_fraction'],
                    budget_outside_tolerance_runs=sum(not r['budget_within_tolerance'] for r in group),
                    positive_9999_runs=sum(r[f'{method}_event_9999'] for r in group),
                    positive_99999_runs=sum(r[f'{method}_event_99999'] for r in group),
                )
    return out


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def checkpoint_comparisons(context,c,study,seed,p,terminal_baseline,*,root,seconds_per_path,timing_device,deadline=None):
    """Independent, actually timed MC at level ends and first event discovery.

    Every MH population is recorded. To bound pilot cost, additional MC is
    measured at level ends and the first population with an event, never chosen by
    estimation accuracy. There is no division of the terminal wall time.
    """
    from .camera_ready_gpu import crude,wall_clock_count
    snapshots=p['population_checkpoints'];last=len(snapshots)-1
    selected={i for i,cp in enumerate(snapshots) if cp['sweep']==c['_exploration']['moves_per_level']}
    selected.update(x['first_checkpoint_index'] for x in p['first_event'].values()
                    if x['first_checkpoint_index'] is not None)
    baselines={last:terminal_baseline}
    for index in sorted(selected-{last}):
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Soft job deadline: population and completed MC checkpoints are saved')
        cp=snapshots[index]
        destination=root/'checkpoint_baselines'/f"{c['_exploration']['id']}_seed{seed}_level{cp['level']}_sweep{cp['sweep']}.json"
        if destination.exists():baseline=json.loads(destination.read_text(encoding='utf-8'))
        else:
            baseline=crude(context,crude_seed(seed,study['crude_stream']+1000000+100*cp['level']+cp['sweep']),
                study['thresholds'],count=wall_clock_count(cp['runtime_seconds_cumulative'],seconds_per_path),
                policy='matched_wall_clock_checkpoint',target_seconds=cp['runtime_seconds_cumulative'])
            baseline.update(timing_device=timing_device,count_chosen_from_independent_calibration=True,
                calibration_seconds_per_path_used=seconds_per_path)
            atomic_write_json(destination,baseline)
            print(f"CHECKPOINT_MC seed={seed} level={cp['level']} sweep={cp['sweep']} "
                  f"n={baseline['n']} SMC={cp['runtime_seconds_cumulative']:.3f}s MC={baseline['runtime_seconds']:.3f}s",flush=True)
        if baseline.get('timing_device')!=timing_device:raise RuntimeError('Checkpoint MC belongs to another GPU')
        baselines[index]=baseline
    rows=[];first_events=[]
    for index,cp in enumerate(snapshots):
        for a,hits in zip(study['thresholds'],study['reference']['discoveries']):
            cap=study.get('report_max_lambda',{}).get(a,c['_exploration']['schedule'][-1])
            value=cp['corrections'][str(a)];baseline=baselines.get(index)
            ratio=None if baseline is None else baseline['runtime_seconds']/cp['runtime_seconds_cumulative']
            rows.append(dict(configuration=c['_exploration']['id'],seed=seed,N=cp['num_particles'],
                level=cp['level'],sweep=cp['sweep'],**{'lambda':cp['lambda']},threshold=a,
                p_reference=hits/study['reference']['samples'],
                event_fraction=value['event_particles']/cp['num_particles'],
                eligible_population=value['event_particles']>=study.get('minimum_event_particles',1),
                within_event_lambda_cap=cp['lambda']<=cap,
                native_transitions_cumulative=cp['native_transitions_cumulative'],
                smc_runtime_seconds_cumulative=cp['runtime_seconds_cumulative'],
                log_z_hat=cp['log_z_hat'],smc_ess_fraction=cp['smc_ess_fraction'],
                resampled_at_level=cp['resampled_at_level'],
                **value,MC_comparison_measured=baseline is not None,
                crude=None if baseline is None else baseline['estimates'][str(a)],
                crude_n=None if baseline is None else baseline['n'],
                crude_zero=None if baseline is None else baseline['estimates'][str(a)]==0,
                crude_runtime_seconds=None if baseline is None else baseline['runtime_seconds'],
                wall_clock_ratio=ratio,
                clock_within_tolerance=None if ratio is None else abs(ratio-1)<=study['timing_tolerance_fraction']))
            rule=p['first_event'][str(a)]
            eligible=[i for i,x in enumerate(snapshots) if x['lambda']<=cap]
            chosen=rule['first_checkpoint_index'] if rule['detected'] else eligible[-1]
            if index==chosen:
                first_events.append({**rows[-1],'first_event_detected':rule['detected'],
                    'selection':'first_event' if rule['detected'] else 'no_event_at_lambda_cap'})
    return rows,first_events


def report(path,output_root):
    study,configs=study_configs(path);root=Path(output_root)/study['output_subdirectory']
    files=sorted((root/'workers').glob('worker_*/runs/*.json'))
    if not files:return
    rows=[r for f in files for r in json.loads(f.read_text(encoding='utf-8'))['rows']]
    keys=[(r['configuration'],r['seed'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run or mismatched merge')
    manifests=[json.loads(f.read_text(encoding='utf-8')) for f in (root/'workers').glob('worker_*/manifest.json')]
    if any(m['study_id']!=fingerprint(study) for m in manifests):raise ValueError('Mixed study outputs')
    if any(m['source_SHA256']!=manifests[0]['source_SHA256'] for m in manifests):raise ValueError('Different runtime code across workers')
    atomic_write_json(root/'manifest.json',{'study':study,'study_id':fingerprint(study),'source_SHA256':manifests[0]['source_SHA256']})
    summary=summarize(rows,study);write_csv(root/'runs.csv',rows);write_csv(root/'summary.csv',summary)
    if study.get('record_population_checkpoints'):
        completed=[json.loads(f.read_text(encoding='utf-8')) for f in files]
        checkpoints=[r for result in completed for r in result['checkpoint_rows']]
        first_events=[r for result in completed for r in result['first_event_rows']]
        write_csv(root/'checkpoint_runs.csv',checkpoints)
        write_csv(root/'first_event_runs.csv',first_events)
    headers=['configuration','threshold','method','runs','median_log_relative_error','mean_log_relative_error',
             'median_ratio','mean_ratio','zero_count','zero_percent','mean_crude_paths','mean_SMC_seconds','mean_MC_seconds']
    if study.get('strong_tilt_screen') or study.get('gradual_budget_screen'):
        headers += ['positive_count','positive_mean_log_relative_error','positive_median_log_relative_error']
    if study.get('gradual_budget_screen'):
        headers += ['target_threshold','budget_fraction','target_MC_paths','N_min','N_max',
                    'budget_outside_tolerance_runs','positive_9999_runs','positive_99999_runs']
    body=f'<h1>Accuracy versus measured cost: exploratory screening</h1><p>{len(files)}/{len(configs)*study["independent_runs"]} complete runs.</p>'
    body+='<p>All requested events and both estimators share each population. Independent seeds; no final resampling. '
    body+='Complete times measured on the same GPU after warm-up. MC count fixed from independent calibration; timing mismatches retained, never filtered by accuracy. '
    body+='Reference probabilities are finite estimates from 200,000 IID trajectories. This pilot selects candidates; confirmation requires new seeds.</p>'
    body+='<p>No 50% accuracy cutoff or pass/fail label. Boxplots show medians and means across all independent runs, including zeros and tails. '
    body+='Plots focus on score 0.99 and 0.999. The last reference has only 26/200000 discoveries (p_ref=0.00013), with finite-reference uncertainty.</p>'
    body+='<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'
    for r in summary:body+='<tr>'+''.join('<td>'+escape(f'{r[h]:.5g}' if isinstance(r[h],float) else str(r[h]))+'</td>' for h in headers)+'</tr>'
    body+='</table><p><a href="runs.csv">Individual runs</a> · <a href="summary.csv">Summary</a></p>'
    if study.get('record_population_checkpoints'):
        body+='<h2>Same-run populations</h2><p>Every MH population of the same run is saved; weights and Z change only at weighting. '
        body+='MC is independently generated and timed at every level end and first event discovery. Timing mismatches are recorded. '
        body+='No MC comparison is fabricated for other individual MH moves. Eligibility means at least one particle in the event; '
        body+='the ladder continues after discovery. Zero estimates and runs with no discoveries remain in the complete data.</p>'
        body+='<p><a href="checkpoint_runs.csv">All populations, with eligibility flags</a> · <a href="first_event_runs.csv">First discovery or no discovery, all runs</a></p>'
    (root/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px Arial}table{border-collapse:collapse}th,td{padding:6px;border:1px solid #ddd}</style>'+body,encoding='utf-8')
    atomic_write_json(root/'report_status.json',{'status':'PILOT_COMPLETE' if len(files)==len(configs)*study['independent_runs'] else 'PARTIAL',
        'completed_runs':len(files),'expected_runs':len(configs)*study['independent_runs'],'confirmatory':False})


def run(path,output_root,worker_id=0,workers=3,max_seconds=20700):
    from .camera_ready_gpu import Context, crude, rows_for, wall_clock_count
    from .exploratory_population import population
    study,configs=study_configs(path);tasks=assigned_runs(study,configs,worker_id,workers)
    root=Path(output_root)/study['output_subdirectory']/'workers'/f'worker_{worker_id}'
    modules=['accuracy_budget.py','exploratory_population.py','exploratory_kernel.py','exploratory.py','camera_ready_gpu.py',
             'low_budget.py','smc.py','ancestral.py','evaluation.py','run_smc.py','runtime.py','configuration.py','pipeline.py']
    if study.get('record_population_checkpoints'):
        modules.append('population_checkpoints.py')
    initialize_manifest(root,{'study':study,'study_id':fingerprint(study),'worker_id':worker_id,'workers':workers,
        'resolved_configs':configs,'source_SHA256':{x:hashlib.sha256(Path(__file__).with_name(x).read_bytes()).hexdigest() for x in modules}})
    def dest(c,seed):return root/'runs'/f'{c["_exploration"]["id"]}_seed{seed}.json'
    if all(dest(c,s).exists() for c,s in tasks):return
    if os.environ.get('SLURM_JOB_ID'):
        import torch
        if torch.cuda.device_count()!=1:raise RuntimeError('Each worker must see exactly one allocated GPU')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
        torch.set_num_interop_threads(1)
    started=time.monotonic();context=Context(configs[0]);warm_started=time.monotonic()
    timing_device={'hostname':socket.gethostname(),'SLURM_STEP_GPUS':os.environ.get('SLURM_STEP_GPUS'),
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'SLURM_CPUS_PER_TASK':os.environ.get('SLURM_CPUS_PER_TASK'),
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
        'GPUs_per_run':1,'workers_in_allocation':workers}
    warm_seed=study['warmup_base_seed']+worker_id*100
    warm=copy.deepcopy(configs[-1]);warm['smc']['num_particles']=study['warmup_paths'];context.config=warm
    population(context,warm,study,warm_seed)
    crude(context,warm_seed+1,study['thresholds'],count=study['warmup_paths'])
    pilots=[crude(context,warm_seed+2+i,study['thresholds'],count=study['calibration_paths'])
            for i in range(study['calibration_repetitions'])]
    seconds_per_path=float(np.median([p['runtime_seconds']/p['n'] for p in pilots]))
    calibration_path=root/f'calibration_{time.time_ns()}.json'
    atomic_write_json(calibration_path,{'seconds_per_path':seconds_per_path,
        'calibration_seconds':[p['runtime_seconds'] for p in pilots],'calibration_paths':study['calibration_paths'],
        'seeds':[warm_seed+i for i in range(2+len(pilots))],'warmup_and_calibration_seconds':time.monotonic()-warm_started,
        'excluded_from_estimates_and_measured_run_times':True,'timing_device':timing_device})
    for c,seed in tasks:
        final=dest(c,seed)
        if final.exists():continue
        if time.monotonic()-started>max_seconds-900:
            print('SOFT TIME LIMIT: checkpoints saved; no automatic resubmission.',flush=True);return
        if study.get('gradual_budget_screen'):
            case = c['_exploration']
            calibration = root/'budget_calibration'/f"{case['id']}.json"
            if not calibration.exists():
                target_seconds = case['target_MC_paths'] * seconds_per_path
                calibration_seed = warm_seed + 1000 + 10 * configs.index(c) + worker_id * 1000
                trials = []
                n = c['smc']['num_particles']
                for attempt in range(2):
                    trial = copy.deepcopy(c)
                    trial['smc']['num_particles'] = n
                    context.config = trial
                    seconds = population(context, trial, study, calibration_seed + attempt)['summary']['runtime_seconds']
                    trials.append({'N': n, 'seconds': seconds, 'seed': calibration_seed + attempt})
                    if abs(seconds / target_seconds - 1) <= .05:
                        break
                    n = max(8, min(384, 8 * round(n * target_seconds / seconds / 8)))
                    if n == trials[-1]['N']:
                        break
                best = min(trials, key=lambda x: abs(x['seconds'] - target_seconds))
                atomic_write_json(calibration, {'N': best['N'], 'target_seconds': target_seconds,
                                               'trials': trials, 'timing_device': timing_device})
            selected = json.loads(calibration.read_text(encoding='utf-8'))
            if selected['timing_device'] != timing_device:
                raise RuntimeError('Budget calibration belongs to a different GPU')
            c = copy.deepcopy(c)
            c['smc']['num_particles'] = selected['N']
        context.config=c;pp=root/'populations'/final.name;bp=root/'baselines'/final.name
        if pp.exists():p=json.loads(pp.read_text(encoding='utf-8'))
        else:
            p=population(context,c,study,seed);p['timing_device']=timing_device
            audit_population(p,c);atomic_write_json(pp,p)
            print(f"SMC_LADDER_SAVED seed={seed} N={c['smc']['num_particles']} "
                  f"lambda={c['_exploration']['schedule'][-1]:g} seconds={p['summary']['runtime_seconds']:.3f}",flush=True)
        audit_population(p,c);target=p['summary']['runtime_seconds']
        if bp.exists():b=json.loads(bp.read_text(encoding='utf-8'))
        else:
            if p.get('timing_device')!=timing_device:
                raise RuntimeError('Saved SMC time belongs to another physical GPU; do not pair it with a new GPU baseline')
            b=crude(context,crude_seed(seed,study['crude_stream']),study['thresholds'],
                count=wall_clock_count(target,seconds_per_path),policy='matched_wall_clock',target_seconds=target)
            b['count_chosen_from_independent_calibration']=True
            b['calibration_seconds_per_path_used']=seconds_per_path
            b['calibration_file']=calibration_path.name
            b['timing_device']=timing_device
            atomic_write_json(bp,b)
        if b.get('timing_device')!=p.get('timing_device'):
            raise RuntimeError('SMC/MC physical timing devices differ')
        case=c['_exploration'];rows=rows_for(p,b,{**study,'lambda_final':case['schedule'][-1]})
        for r in rows:r.update(configuration=case['id'],configuration_label=case['label'],
            levels=len(case['schedule']),moves_per_level=case['moves_per_level'],lambda_final=case['schedule'][-1],
            analysis_role='control' if r['threshold']==study.get('control_threshold') else 'rare_event',
            crude_expected_events_from_reference=b['n']*r['p_reference'],
            timing_basis='measured_complete_terminal_run',
            clock_within_tolerance=abs(r['wall_clock_ratio']-1)<=study['timing_tolerance_fraction'])
        if study.get('gradual_budget_screen'):
            extreme = {}
            for suffix, threshold in [('9999', .9999), ('99999', .99999)]:
                found = any(x['event_toxic_probability'] >= threshold for x in p['particles'])
                extreme.update({f'smc_event_{suffix}':found, f'snis_event_{suffix}':found,
                                f'crude_event_{suffix}':any(x >= threshold for x in b['scores'])})
            for r in rows:
                r.update(target_threshold=case['target_threshold'], budget_fraction=case['budget_fraction'],
                         target_MC_paths=case['target_MC_paths'], target_seconds=selected['target_seconds'],
                         budget_within_tolerance=abs(target/selected['target_seconds']-1)<=.10, **extreme)
        completed={'rows':rows,'schedule':[0.]+case['schedule'],'normalizer_and_policy_audit_passed':True,
            'sample_count_fixed_before_MC_observations':True,'no_final_resampling':True}
        if study.get('record_population_checkpoints'):
            try:
                checkpoint_rows,first_events=checkpoint_comparisons(context,c,study,seed,p,b,root=root,
                    seconds_per_path=seconds_per_path,timing_device=timing_device,deadline=started+max_seconds)
            except TimeoutError as exc:
                print(str(exc),flush=True)
                return
            completed.update(checkpoint_rows=checkpoint_rows,first_event_rows=first_events)
        atomic_write_json(final,completed)
        print(f'COMPLETED {case["id"]} seed={seed} SMC={target:.3f}s MC={b["runtime_seconds"]:.3f}s n={b["n"]}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['plan','run','report'])
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('experiments/mdlm_toxicity_rare_event/artifacts/current'))
    p.add_argument('--worker-id',type=int,default=0);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-seconds',type=int,default=20700);a=p.parse_args()
    if a.stage=='plan':print(json.dumps(plan(a.study),indent=2))
    elif a.stage=='report':report(a.study,a.output_root)
    else:run(a.study,a.output_root,a.worker_id,a.workers,a.max_seconds)


if __name__=='__main__':main()
