"""Bounded, resumable exploration; no change to P, scores or MH proposals.

Three configurations, interleaved independent runs. Both estimators and both
events use the same terminal population; one independent crude stream per run.
All new files live in a new study directory, preserving the previous pilot.
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
import time
import numpy as np

from .configuration import decoder_id, evaluator_id, fingerprint, load_yaml
from .low_budget import crude_seed, transition_budget
from .pipeline import audit_population as audit_population
from .storage import atomic_write_json, initialize_manifest


def study_configs(path):
    study=load_yaml(Path(path));base=load_yaml(Path(path).parent/study['base_config'])
    if (decoder_id(base),evaluator_id(base))!=(study['reference']['decoder_id'],study['reference']['evaluator_id']):
        raise ValueError('Frozen decoder/evaluator mismatch')
    if study['thresholds']!=[.9,.99] or study['comparison']!='matched_transitions':
        raise ValueError('This exploration fixes events and transition matching')
    configs=[];identifiers=set()
    for index,case in enumerate(study['configurations']):
        if case['id'] in identifiers: raise ValueError('Duplicate configuration')
        identifiers.add(case['id']);grid=study['schedules'][case['schedule']]
        if not grid or any(not math.isfinite(x) for x in grid) or not all(a<b for a,b in zip([0]+grid[:-1],grid)) or grid[-1]!=study['lambda_final']:
            raise ValueError('Invalid fixed schedule')
        config=copy.deepcopy(base);s=config['smc'];m=s['tilted_sampler']
        s.update(num_particles=int(case['N']),num_independent_runs=study['independent_runs'],
            independent_runs=study['independent_runs'],tilted_lambdas=[study['lambda_final']],
            tilted_component_weights_a=[1.],diagnostic_lambdas=[],final_resampling=False,
            base_seed=study['base_seed']+index*10000)
        m.update(num_tempering_levels=len(grid),mcmc_steps_per_level=case['moves_per_level'],mutation_policy='every_level')
        config['_exploration']={**case,'schedule':grid,
            'label':f"{case['id']}: {case['N']} / {len(grid)} / {case['moves_per_level']}"}
        configs.append(config)
    return study,configs


def plan(path):
    study,configs=study_configs(path);cases=[]
    for config in configs:
        c=config['_exploration'];m=config['smc']['tilted_sampler'];n=config['smc']['num_particles']
        average=sum(d*p for d,p in zip(m['rollback_steps'],m['rollback_probabilities']))
        expected=n*(33+len(c['schedule'])*c['moves_per_level']*(average+1))
        cases.append({'configuration':c['id'],'N':n,'schedule':[0]+c['schedule'],
            'levels':len(c['schedule']),'moves_per_level':c['moves_per_level'],
            'expected_transitions':expected,'expected_crude_paths':expected/33,
            'independent_runs':study['independent_runs'],'base_seed':config['smc']['base_seed']})
    return {'status':'prepared_not_submitted','study':study,'configurations':cases,
        'comparison':'Each run receives floor(actual native transitions / 33) complete crude paths',
        'selection':'Exploratory results only; any final choice requires independent validation runs'}


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def assigned_runs(study,configs,worker_id=0,workers=1):
    if not 1<=workers<=3 or not 0<=worker_id<workers:
        raise ValueError('Require 1..3 workers and a valid worker_id')
    return [(c,c['smc']['base_seed']+1000000*i)
        for i in range(study['independent_runs']) if i%workers==worker_id for c in configs]


def report(path,output_root,*,root_override=None):
    study,configs=study_configs(path);root=Path(root_override) if root_override is not None else Path(output_root)/study['output_subdirectory']
    files=sorted((root/'runs').glob('*.json'))+sorted((root/'workers').glob('worker_*/runs/*.json'))
    if not files: return
    rows=[r for f in files for r in json.loads(f.read_text())['rows']]
    keys=[(r['configuration'],r['seed'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate independent run')
    worker_manifests=sorted((root/'workers').glob('worker_*/manifest.json'))
    if worker_manifests:
        manifests=[json.loads(p.read_text()) for p in worker_manifests]
        if any(m['study_id']!=fingerprint(study) for m in manifests):raise ValueError('Different studies in worker outputs')
        if any(m['runtime_source_sha256']!=manifests[0]['runtime_source_sha256'] for m in manifests):raise ValueError('Different source code across GPU workers')
        atomic_write_json(root/'manifest.json',{'study_id':fingerprint(study),'study':study,
            'resolved_configs':configs,'parallel_execution':{'workers':len(worker_manifests),'assignment':'run_index modulo workers'},
            'runtime_source_sha256':manifests[0]['runtime_source_sha256'],
            'worker_manifest_files':[str(p.relative_to(root)) for p in worker_manifests]})
    write_csv(root/'runs.csv',rows);summary=[]
    for case,a in sorted({(r['configuration'],r['threshold']) for r in rows}):
        group=[r for r in rows if (r['configuration'],r['threshold'])==(case,a)]
        first=group[0]
        for method in ['smc','snis','crude']:
            x=np.array([r[method] for r in group]);ratio=x/first['p_reference'];metric=np.log10(1+abs(ratio-1))
            summary.append({k:first[k] for k in ['configuration','configuration_label','N','levels','moves_per_level','threshold','p_reference']}|
                {'method':method,'runs':len(group),'zero_percent':float(100*(x==0).mean()),
                 'median_log_relative_error':float(np.median(metric)),
                 'mean_transitions':float(np.mean([r['native_transitions'] for r in group])),
                 'mean_crude_paths':float(np.mean([r['crude_n'] for r in group])),
                 **{f'ratio_q{int(100*q):02d}':float(np.quantile(ratio,q)) for q in [.1,.25,.5,.75,.9,.99]}})
    write_csv(root/'summary.csv',summary)
    from .exploration_figures import svg_figure
    completed=len(files);expected=len(configs)*study['independent_runs']
    counts={c['_exploration']['id']:sum(r['configuration']==c['_exploration']['id'] for r in rows)//2 for c in configs}
    def table(headers,records):
        return '<table><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in record)+'</tr>' for record in records)+'</table>'
    configurations=[[c['_exploration']['label'],'0 → '+' → '.join(map(str,c['_exploration']['schedule'])),counts[c['_exploration']['id']]] for c in configs]
    body=f'<h1>RollBack SMC: esplorazione delle schedule e delle mutazioni</h1><p><b>{completed}/{expected} run complete.</b> Otto repliche per configurazione: studio esplorativo, senza conclusioni definitive sui vincitori.</p>'
    body+='<p>Stesso P, decoder DDPM a 32 passi più completamento categorico, score log(q), eventi score ≥0,9 e ≥0,99, d uniforme in {4,8,16,24,32}. Pesatura → eventuale systematic ESS/N &lt;0,5 → tutte le mosse MH. Nessun resampling finale. Entrambi gli stimatori dalla stessa popolazione finale a tilt 3.</p>'
    body+='<p>Ogni crude usa floor(transizioni effettive/33) traiettorie native indipendenti. Sono contate anche le proposte rifiutate. Tempi registrati, senza assegnare budget a wall clock. Nessuna media fra configurazioni o livelli.</p>'
    body+='<p>Reference finite da 200.000 campioni: 0,00758 e 0,00222, con IC 95% mostrati nei grafici. Non sono probabilità esatte. Una stima zero è un errore relativo del 100%, non una stima precisa.</p>'
    body+=table(['Configurazione: N / livelli / MH','Schedule completa','Run complete'],configurations)
    for heading,kind in [('Errore relativo logaritmico','log_error'),('Rapporto stima/reference','ratio')]:
        body+='<h2>'+heading+'</h2>'+svg_figure(rows,kind)
    body+='<p><a href="runs.csv">Singole run</a> · <a href="summary.csv">Tabella completa</a> · <a href="manifest.json">Parametri e versioni</a>. Diagnostiche complete, pesi terminali, normalizzatori e genealogie nei checkpoint delle popolazioni.</p>'
    body+='<p>La scelta di configurazioni promettenti su questi dati richiede successiva validazione con nuovi seed. Conservare anche risultati negativi e run con stima nulla.</p>'
    html='<!doctype html><html lang="it"><meta charset="utf-8"><title>RollBack SMC: esplorazione</title><style>body{font:16px Arial;max-width:1500px;margin:30px auto;padding:0 20px}svg{width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}</style>'+body+'</html>'
    (root/'REPORT.html').write_text(html,encoding='utf-8')
    (root/'REPORT.md').write_text(f'# Esplorazione RollBack SMC\n\n{completed}/{expected} run complete.\n\nRisultati esplorativi. Due boxplot, zeri inclusi, code complete in REPORT.html.\n',encoding='utf-8')
    atomic_write_json(root/'report_status.json',{'completed_runs':completed,'expected_runs':expected,'completed_by_configuration':counts,'status':'COMPLETE' if completed==expected else 'PARTIAL','reference_is_exact':False})


def run(path,output_root,*,max_seconds=19800,worker_id=0,workers=1):
    from .camera_ready_gpu import Context, crude, rows_for
    from .exploratory_population import population
    study,configs=study_configs(path);root=Path(output_root)/study['output_subdirectory']
    schedule=assigned_runs(study,configs,worker_id,workers)
    if workers>1:root=root/'workers'/f'worker_{worker_id}'
    modules=['exploratory.py','exploratory_kernel.py','exploratory_population.py','exploration_figures.py',
             'smc.py','ancestral.py','evaluation.py','run_smc.py','runtime.py','camera_ready_gpu.py',
             'low_budget.py','pipeline.py']
    initialize_manifest(root,{'study_id':fingerprint(study),'study':study,'resolved_configs':configs,
        'runner_version':'explicit_fixed_grids_v2','budget':'All DDPM and terminal categorical transitions, rejected proposals included',
        'parallel_execution':{'workers':workers,'worker_id':worker_id,'assignment':'run_index modulo workers'},
        'runtime_source_sha256':{x:hashlib.sha256(Path(__file__).with_name(x).read_bytes()).hexdigest() for x in modules}})
    def dest(config,seed):return root/'runs'/f"{config['_exploration']['id']}_N{config['smc']['num_particles']}_seed{seed}.json"
    # Interleave settings: a time limit must not leave the largest case untested.
    if all(dest(c,s).exists() for c,s in schedule):report(path,output_root,root_override=root);return
    if workers>1 and os.environ.get('SLURM_JOB_ID'):
        import torch
        if torch.cuda.device_count()!=1:raise RuntimeError('Each parallel SLURM step must see exactly one GPU')
    started=time.monotonic();context=Context(configs[0]);warm=time.monotonic()
    warm_config=copy.deepcopy(configs[0]);warm_config['smc']['num_particles']=16
    warm_seed=9834811+worker_id*100
    population(context,warm_config,study,warm_seed)
    crude(context,warm_seed+1,study['thresholds'],count=16)
    atomic_write_json(root/'warmup.json',{'seeds':[warm_seed,warm_seed+1],'seconds':time.monotonic()-warm,
        'excluded_from_results_and_per_run_times':True,'worker_id':worker_id,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'SLURM_STEP_GPUS':os.environ.get('SLURM_STEP_GPUS')})
    for config,seed in schedule:
        final=dest(config,seed)
        if final.exists():continue
        # Reserve 15 minutes for a possibly long final component and reporting.
        if time.monotonic()-started>max_seconds-900:
            report(path,output_root,root_override=root);print('SOFT TIME LIMIT: checkpoints retained, no resubmission.',flush=True);return
        context.config=config;p_path=root/'populations'/final.name;b_path=root/'baselines'/final.name
        if p_path.exists():p=json.loads(p_path.read_text())
        else:
            p=population(context,config,study,seed);audit_population(p,config);atomic_write_json(p_path,p)
        audit_population(p,config);cost=transition_budget(p['summary'])
        if b_path.exists():b=json.loads(b_path.read_text())
        else:
            b=crude(context,crude_seed(seed,925081),study['thresholds'],count=cost['crude_n'])
            atomic_write_json(b_path,b)
        if b['n']!=cost['native_transitions']//33:raise AssertionError('Crude budget mismatch')
        rows=rows_for(p,b,study);case=config['_exploration']
        for row in rows:row.update(configuration=case['id'],configuration_label=case['label'],levels=len(case['schedule']),moves_per_level=case['moves_per_level'],schedule_name=next(c['schedule'] for c in study['configurations'] if c['id']==case['id']))
        atomic_write_json(final,{'rows':rows,'summary':p['summary'],'schedule':[0]+case['schedule'],'normalizer_audit_passed':True})
        report(path,output_root,root_override=root)
        print(f"COMPLETED {case['label']} seed={seed} SMC_seconds={p['summary']['runtime_seconds']:.3f} crude_n={b['n']}",flush=True)
    report(path,output_root,root_override=root)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['plan','run','report'])
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output-root',type=Path,default=Path('experiments/mdlm_toxicity_rare_event/artifacts/current'));parser.add_argument('--max-seconds',type=int,default=19800)
    parser.add_argument('--worker-id',type=int,default=0);parser.add_argument('--workers',type=int,choices=[1,2,3],default=1)
    a=parser.parse_args()
    if a.stage=='plan':print(json.dumps(plan(a.study),indent=2))
    elif a.stage=='run':run(a.study,a.output_root,max_seconds=a.max_seconds,worker_id=a.worker_id,workers=a.workers)
    else:report(a.study,a.output_root)

if __name__=='__main__':main()
