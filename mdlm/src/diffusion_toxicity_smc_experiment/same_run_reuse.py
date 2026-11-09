"""Continue saved SMC ladders; reuse completed IID MC data, with no new MC draws.

SMC and its estimators are unchanged. MC budgets use independently measured
seconds per trajectory. Their costs are calibrated equivalents, not newly
measured runtimes. Disjoint fixed MC blocks distinguish independent seeds;
prefixes within a seed are correlated across checkpoints, as intended.
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

from .accuracy_budget import assigned_runs, study_configs
from .exploratory import audit_population
from .storage import atomic_write_json, initialize_manifest
from .configuration import fingerprint
from .potentials import potential_from_config


def native_clock_count(seconds, seconds_per_path):
    if seconds<=0 or seconds_per_path<=0:raise ValueError('Invalid measured cost')
    return max(1,math.floor(seconds/seconds_per_path))


def fixed_mc_block(scores, *, seed_rank, total_seeds, required):
    """Blocks fixed by planned seed rank, never by successes or available SMC runs."""
    size=len(scores)//total_seeds
    if not 0<=seed_rank<total_seeds or not 1<=required<=size:
        raise ValueError('Saved independent MC pool does not cover the fixed seed block')
    start=seed_rank*size
    return scores[start:start+required],start,size


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def calibration_for(root,population):
    device=population['timing_device']
    matches=[]
    for path in sorted(root.glob('workers/worker_*/calibration_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        physical=('hostname','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK')
        if all(data['timing_device'].get(k)==device.get(k) for k in physical):
            if not data['excluded_from_estimates_and_measured_run_times']:
                raise ValueError('Timing calibration was not independent')
            matches.append((path,data))
    if not matches:raise ValueError('No independent calibration for this GPU and CPU allocation')
    return matches[0]


def mc_pool(pool_root,study):
    manifest=json.loads((pool_root/'manifest.json').read_text(encoding='utf-8'))
    reference=manifest['study']['reference']
    if any(reference[k]!=study['reference'][k] for k in ['decoder_id','evaluator_id']):
        raise ValueError('Saved MC belongs to another native model or classifier')
    status=json.loads((pool_root/'report_status.json').read_text(encoding='utf-8'))
    if status['completed_runs']!=status['expected_runs']:
        raise ValueError('Reuse a completed MC study, not an outcome-dependent subset')
    files=sorted(pool_root.glob('workers/worker_*/baselines/*.json'))
    values=[];seeds=[];sources=[]
    for path in files:
        raw=path.read_bytes();data=json.loads(raw)
        if data['n']!=len(data['scores']):raise ValueError('Incomplete MC file')
        if data['native_transitions']!=33*data['n']:raise ValueError('MC baseline did not use complete native trajectories')
        if not data['sample_count_chosen_before_generation']:
            raise ValueError('MC sample count was chosen from its own outcomes')
        seeds.append(data['seed']);values.extend(data['scores'])
        sources.append(dict(file=str(path.relative_to(pool_root)),sha256=hashlib.sha256(raw).hexdigest(),
                            seed=data['seed'],samples=data['n']))
    if len(seeds)!=len(set(seeds)):raise ValueError('Repeated MC random stream')
    return np.asarray(values,dtype=float),sources


def report(study_path,output_root,pool_root):
    study,configs=study_configs(study_path);root=Path(output_root)/study['output_subdirectory']
    tasks=sorted(assigned_runs(study,configs,0,1),key=lambda x:x[1])
    seed_ranks={seed:i for i,(_,seed) in enumerate(tasks)}
    scores,sources=mc_pool(Path(pool_root),study)
    rows=[];first_events=[];blocks={};population_sources=[]
    references=dict(zip(study['thresholds'],[n/study['reference']['samples'] for n in study['reference']['discoveries']]))
    for file in sorted(root.glob('workers/worker_*/populations/*.json')):
        raw=file.read_bytes();population=json.loads(raw)
        seed=int(population['summary']['seed'])
        if seed not in seed_ranks:raise ValueError('Unplanned SMC seed')
        config=next(c for c,s in tasks if s==seed)
        expected_phi=potential_from_config(config).name
        if population.get('phi',{}).get('name','log_probability')!=expected_phi:
            raise ValueError('Saved population uses another Phi')
        audit_population(population,config)
        checkpoints=population['population_checkpoints']
        if len(checkpoints)!=65:raise ValueError('Expected initial population plus 64 MH moves')
        cal_path,cal=calibration_for(root,population);cost=float(cal['seconds_per_path'])
        counts=[native_clock_count(cp['runtime_seconds_cumulative'],cost) for cp in checkpoints]
        block,start,size=fixed_mc_block(scores,seed_rank=seed_ranks[seed],total_seeds=len(tasks),required=max(counts))
        blocks[str(seed)]=dict(pool_start=start,fixed_block_size=size,scores=block.tolist())
        population_sources.append(dict(file=str(file.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),seed=seed))
        for index,(cp,n) in enumerate(zip(checkpoints,counts)):
            if n>len(block):raise ValueError('MC prefix exceeds assigned seed block')
            prefix=block[:n]
            for threshold in study['thresholds']:
                value=cp['corrections'][str(threshold)]
                hits=int((prefix>=threshold).sum())
                row=dict(configuration=config['_exploration']['id'],seed=seed,N=128,phi=expected_phi,
                    level=cp['level'],sweep=cp['sweep'],**{'lambda':cp['lambda']},threshold=threshold,
                    p_reference=references[threshold],
                    eligible_population=value['event_particles']>=1,event_fraction=value['event_particles']/128,
                    **value,log_z_hat=cp['log_z_hat'],smc_ess_fraction=cp['smc_ess_fraction'],
                    native_transitions_cumulative=cp['native_transitions_cumulative'],
                    smc_runtime_seconds_cumulative=cp['runtime_seconds_cumulative'],
                    crude=hits/n,crude_event_particles=hits,crude_zero=hits==0,crude_n=n,
                    crude_native_transitions=33*n,crude_runtime_seconds=None,
                    crude_runtime_seconds_estimated=n*cost,MC_comparison_available=True,MC_comparison_measured=False,
                    timing_basis='measured SMC time / independent seconds-per-MC-trajectory calibration',
                    calibration_seconds_per_path=cost,calibration_file=str(cal_path.relative_to(root)),
                    MC_pool_start=start,GPUs_per_estimate=1,CPUs_per_estimate=4)
                rows.append(row)
                discovery=population['first_event'][str(threshold)]
                chosen=discovery['first_checkpoint_index'] if discovery['detected'] else len(checkpoints)-1
                if index==chosen:first_events.append({**row,'first_event_detected':discovery['detected']})
    if not rows:return
    keys=[(r['seed'],r['level'],r['sweep'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Repeated SMC population')
    write_csv(root/'checkpoint_runs.csv',rows);write_csv(root/'first_event_runs.csv',first_events)
    terminal=[r for r in rows if r['level']==16 and r['sweep']==4]
    write_csv(root/'runs.csv',terminal)
    summary=[]
    for threshold in study['thresholds']:
        for level in range(1,17):
            group=[r for r in rows if r['threshold']==threshold and r['level']==level and r['sweep']==4]
            for method in ['smc','snis','crude']:
                estimates=np.array([r[method] for r in group]);reference=references[threshold]
                summary.append(dict(threshold=threshold,**{'lambda':level*.5},method=method,runs=len(group),
                    zero_count=int((estimates==0).sum()),positive_count=int((estimates>0).sum()),
                    mean_estimate=float(estimates.mean()),median_estimate=float(np.median(estimates)),
                    median_log_relative_error=float(np.median(np.log10(1+abs(estimates/reference-1))))))
    write_csv(root/'summary.csv',summary)
    atomic_write_json(root/'saved_MC_blocks.json',blocks)
    atomic_write_json(root/'comparison_policy.json',dict(
        policy='saved_independent_MC_prefixes_at_calibrated_wall_clock_budget',new_MC_samples_generated=0,
        MC_pool_root=str(pool_root),MC_pool_samples=len(scores),MC_sources=sources,
        selection='All baseline files in deterministic path order; fixed disjoint block per PLANNED seed rank; no outcome-based selection',
        reference_data_reused_as_MC=False,MC_timing_newly_measured=False,
        levels_within_seed_are_correlated=True,independent_seed_blocks=True,
        fixed_block_size=len(scores)//len(tasks),SMC_sources=population_sources,
        resources=dict(GPUs_per_estimate=1,CPUs_per_estimate=4,parallel_seeds=3),
        scientific_SMC_parameters_unchanged=study.get('phi','log_probability')=='log_probability',
        phi=potential_from_config(configs[0]).manifest(),
        original_log_probability_results_preserved=True))
    complete=len(population_sources)
    atomic_write_json(root/'manifest.json',dict(study=study,comparison_policy='saved independent MC prefixes; calibrated equivalent cost',
        original_job=study.get('previous_job','2079596'),continuation_job=os.environ.get('SLURM_JOB_ID'),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        SMC_sources=population_sources,MC_source_files=len(sources),MC_pool_samples=len(scores),new_MC_generation=False))
    atomic_write_json(root/'report_status.json',dict(status='PILOT_COMPLETE' if complete==len(tasks) else 'PARTIAL',
        completed_runs=complete,expected_runs=len(tasks),new_MC_generation=False,
        comparison='calibrated equivalent budget using existing MC samples'))
    body=f'<h1>Same-run SMC ladders: {complete}/{len(tasks)} seeds</h1>'
    body+='<p>Phi: '+escape(potential_from_config(configs[0]).manifest()['definition'])+'. '
    body+='N=128; 16 levels from 0.5 to 8; four MH moves per level. The same population estimates score ≥0.99 and ≥0.999. '
    body+='All populations and zeros are retained; positive-only plots are conditional distributions. No averaging across levels.</p>'
    body+='<p>MC reuses completed independent samples in fixed disjoint blocks across seeds and nested prefixes within each seed. '
    body+='Budget = floor(measured cumulative SMC time / independently calibrated MC time per trajectory). '
    body+='MC cost is a calibrated equivalent, not a newly measured runtime. Each estimate uses 1 GPU and 4 CPU; 3 seeds in parallel. '
    body+='The finite independent reference is 444/200000 and 26/200000; reference samples are not used as MC baseline data.</p>'
    for name in ['checkpoint_runs.csv','first_event_runs.csv','summary.csv','comparison_policy.json']:
        body+=f'<p><a href="{name}">{escape(name)}</a></p>'
    (root/'REPORT.html').write_text('<!doctype html><html lang="it"><meta charset="utf-8"><style>body{font:16px Arial;max-width:1600px;margin:25px}</style>'+body+'</html>',encoding='utf-8')
    print(f'REUSE_REPORT {complete}/{len(tasks)} SMC ladders; {len(scores)} existing MC samples; no new MC draws',flush=True)


def run(study_path,output_root,worker_id,workers,max_seconds):
    from .camera_ready_gpu import Context
    from .exploratory_population import population
    import torch
    study,configs=study_configs(study_path)
    root=Path(output_root)/study['output_subdirectory'];worker=root/'workers'/f'worker_{worker_id}'
    tasks=assigned_runs(study,configs,worker_id,workers)
    pending=[(c,seed) for c,seed in tasks if not (worker/'populations'/f"{c['_exploration']['id']}_seed{seed}.json").exists()]
    if not pending:return
    if os.environ.get('SLURM_JOB_ID'):
        if torch.cuda.device_count()!=1:raise RuntimeError('One allocated GPU per seed worker required')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')));torch.set_num_interop_threads(1)
    if socket.gethostname().split('.')[0]!='gpu007' and os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Resume on the original physical node for timing-calibration reuse')
    original=json.loads((worker/'manifest.json').read_text(encoding='utf-8'))
    for module,expected in original['source_SHA256'].items():
        if hashlib.sha256(Path(__file__).with_name(module).read_bytes()).hexdigest()!=expected:
            raise ValueError('Original scientific source changed: '+module)
    started=time.monotonic();context=Context(configs[0])
    device=dict(hostname=socket.gethostname(),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),SLURM_CPUS_PER_TASK=os.environ.get('SLURM_CPUS_PER_TASK'),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),GPUs_per_run=1,workers_in_allocation=workers)
    calibration_for(root,{'timing_device':device})
    job=os.environ.get('SLURM_JOB_ID','local')
    resume_manifest=dict(original_job='2079596',continuation_job=job,
        pending_seeds=[seed for _,seed in pending],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_SMC_source_unchanged=True,new_MC_generation=False,timing_device=device)
    atomic_write_json(worker/f'resume_manifest_{job}.json',resume_manifest)
    warm=copy.deepcopy(configs[0]);warm['smc']['num_particles']=study['warmup_paths'];context.config=warm
    warmed=population(context,warm,{**study,'record_population_checkpoints':False},study['warmup_base_seed']+worker_id*100)
    resume_manifest.update(warmup_summary=warmed['summary'],warmup_excluded_from_estimates_and_timing=True)
    atomic_write_json(worker/f'resume_manifest_{job}.json',resume_manifest)
    for config,seed in pending:
        if time.monotonic()-started>=max_seconds-1000:
            print('SMC continuation soft deadline; saved ladders retained',flush=True);return
        context.config=config
        result=population(context,config,study,seed);result['timing_device']=device
        audit_population(result,config)
        atomic_write_json(worker/'populations'/f"{config['_exploration']['id']}_seed{seed}.json",result)
        print(f"SMC_ONLY_SAVED seed={seed} N=128 lambda=8 seconds={result['summary']['runtime_seconds']:.3f}",flush=True)


def run_fresh(study_path,output_root,worker_id,workers,max_seconds,calibration_root):
    """New configurable-Phi SMC runs; no crude generation, including warm-up.

    Reuse an independent calibration only on the SAME physical GPU and CPU
    allocation. Worker manifests protect seed/config/source identity on resume.
    """
    from .camera_ready_gpu import Context
    from .exploratory_population import population
    import torch

    study,configs=study_configs(study_path)
    if not study.get('reuse_only_MC'):
        raise ValueError('Fresh same-run driver requires explicit existing-MC reuse')
    root=Path(output_root)/study['output_subdirectory'];worker=root/'workers'/f'worker_{worker_id}'
    if root.resolve()==Path(calibration_root).resolve():
        raise ValueError('New Phi study must not overwrite the previous experiment')
    tasks=assigned_runs(study,configs,worker_id,workers)
    if os.environ.get('SLURM_JOB_ID'):
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Exactly one allocated GPU per seed worker required')
        if socket.gethostname().split('.')[0]!='gpu007':
            raise RuntimeError('Timing calibration reuse requires the original physical gpu007 node')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
        torch.set_num_interop_threads(1)
    device=dict(hostname=socket.gethostname(),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),SLURM_CPUS_PER_TASK=os.environ.get('SLURM_CPUS_PER_TASK'),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),GPUs_per_run=1,workers_in_allocation=workers)
    cal_path,cal=calibration_for(Path(calibration_root),{'timing_device':device})
    source_names=['smc.py','ancestral.py','evaluation.py','run_smc.py','runtime.py',
                  'camera_ready_gpu.py','low_budget.py','exploratory_kernel.py',
                  'exploratory_population.py','population_checkpoints.py','potentials.py',
                  'accuracy_budget.py','same_run_reuse.py','configuration.py','pipeline.py']
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names}
    manifest=dict(study_id=fingerprint(study),study=study,resolved_configs=configs,
        source_SHA256=sources,phi=potential_from_config(configs[0]).manifest(),
        planned_seeds=[seed for _,seed in tasks],new_MC_generation=False,
        calibration_source=str(cal_path),calibration_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        timing_device=device)
    initialize_manifest(worker,manifest)
    dest_cal=worker/cal_path.name
    if dest_cal.exists() and json.loads(dest_cal.read_text(encoding='utf-8'))!=cal:
        raise ValueError('Existing timing calibration differs')
    if not dest_cal.exists():atomic_write_json(dest_cal,cal)

    def destination(config,seed):
        return worker/'populations'/f"{config['_exploration']['id']}_seed{seed}.json"

    pending=[]
    for config,seed in tasks:
        if destination(config,seed).exists():
            saved=json.loads(destination(config,seed).read_text(encoding='utf-8'))
            if saved['summary']['seed']!=seed or saved.get('phi',{}).get('name')!=potential_from_config(config).name:
                raise ValueError('Saved seed or Phi differs from the requested experiment')
            audit_population(saved,config)
        else:pending.append((config,seed))
    if not pending:
        print('All assigned SMC ladders already saved; no computation repeated',flush=True)
        return
    started=time.monotonic();context=Context(configs[0])
    warm=copy.deepcopy(configs[0]);warm['smc']['num_particles']=study['warmup_paths']
    context.config=warm
    warmed=population(context,warm,{**study,'record_population_checkpoints':False},study['warmup_base_seed']+worker_id*100)
    atomic_write_json(worker/f"warmup_{os.environ.get('SLURM_JOB_ID','local')}.json",dict(
        summary=warmed['summary'],phi=potential_from_config(warm).manifest(),
        excluded_from_estimates_and_measured_run_times=True,new_MC_generation=False))
    for config,seed in pending:
        if time.monotonic()-started>=max_seconds-1000:
            print('Soft deadline: complete saved ladders retained for resume',flush=True)
            return
        context.config=config
        result=population(context,config,study,seed);result['timing_device']=device
        audit_population(result,config)
        atomic_write_json(destination(config,seed),result)
        summary = result['summary']
        print(f"SMC_PHI_SAVED phi={result['phi']['name']} seed={seed} N={summary['num_particles']} "
              f"lambda={summary['lambda']:g} seconds={summary['runtime_seconds']:.3f}", flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['run','run-fresh','report'])
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--pool-root',type=Path);parser.add_argument('--worker-id',type=int,default=0)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--max-seconds',type=int,default=6300)
    parser.add_argument('--calibration-root',type=Path)
    args=parser.parse_args()
    if args.stage=='run':run(args.study,args.output_root,args.worker_id,args.workers,args.max_seconds)
    elif args.stage=='run-fresh':
        if args.calibration_root is None:parser.error('run-fresh requires --calibration-root')
        run_fresh(args.study,args.output_root,args.worker_id,args.workers,args.max_seconds,args.calibration_root)
    elif args.pool_root is None:parser.error('report requires --pool-root')
    else:report(args.study,args.output_root,args.pool_root)
