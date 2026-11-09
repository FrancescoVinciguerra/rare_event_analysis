"""New independent crude baselines at the measured time of EXISTING SMC runs.

No SMC run is repeated. Sample counts are fixed from independent calibration
before observing any final MC sample. Actual timing mismatch is retained.
"""
from pathlib import Path
import argparse
import csv
import hashlib
from html import escape
import json
import os
import socket
import time
import numpy as np

from .configuration import fingerprint,load_yaml
from .exploratory import study_configs,write_csv
from .low_budget import crude_seed
from .storage import atomic_write_json,initialize_manifest

def load_design(path):
    followup=load_yaml(Path(path));source_path=Path(path).parent/followup['source_study']
    study,configs=study_configs(source_path)
    return followup,study,configs

def source_worker(source,rank):
    if not os.environ.get('SLURM_JOB_ID'):return rank
    if socket.gethostname().split('.')[0]!='gpu007':raise RuntimeError('Require the original node gpu007')
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('Each worker must see exactly one GPU')
    physical=os.environ.get('SLURM_STEP_GPUS');matches=[]
    for index in range(3):
        old=json.loads((source/'workers'/f'worker_{index}'/'warmup.json').read_text())
        if old['SLURM_STEP_GPUS']==physical:matches.append(index)
    if len(matches)!=1:raise RuntimeError('Cannot map this GPU to the original SMC timing device')
    return matches[0]

def run(path,output_root,worker_id):
    from .camera_ready_gpu import Context,crude,rows_for,wall_clock_count
    f,study,configs=load_design(path);parent=Path(output_root);source=parent/f['source_subdirectory']
    source_rank=source_worker(source,worker_id);root=parent/f['output_subdirectory']/'workers'/f'worker_{source_rank}'
    files=sorted((source/'workers'/f'worker_{source_rank}'/'runs').glob('*.json'))
    if not files:raise ValueError('No saved SMC runs to match')
    lookup={c['_exploration']['id']:c for c in configs}
    modules=['wall_clock_followup.py','camera_ready_gpu.py','low_budget.py','smc.py','ancestral.py',
             'evaluation.py','run_smc.py','runtime.py','pipeline.py']
    initialize_manifest(root,{'followup_id':fingerprint(f),'source_study_id':fingerprint(study),
        'followup':f,'source_worker':source_rank,'SMC_repeated':False,
        'source_run_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'runtime_source_sha256':{p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in modules}})
    if all((root/'runs'/p.name).exists() for p in files):return
    context=Context(configs[0]);cal_path=root/'calibration.json'
    if cal_path.exists():cal=json.loads(cal_path.read_text())
    else:
        seed=f['warmup_base_seed']+source_rank*100
        crude(context,seed,study['thresholds'],count=f['warmup_paths'])
        pilots=[crude(context,seed+i+1,study['thresholds'],count=f['calibration_paths']) for i in range(f['calibration_repetitions'])]
        per_path=[b['runtime_seconds']/b['n'] for b in pilots]
        cal={'seconds_per_path':float(np.median(per_path)),
            'calibration_seconds':[b['runtime_seconds'] for b in pilots],
            'calibration_seeds':[b['seed'] for b in pilots],'paths_per_calibration':f['calibration_paths'],
            'warmup_seed':seed,'excluded_from_final_estimates':True,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'SLURM_STEP_GPUS':os.environ.get('SLURM_STEP_GPUS')}
        atomic_write_json(cal_path,cal)
    for file in files:
        dest=root/'runs'/file.name
        if dest.exists():continue
        old=json.loads(file.read_text());case=old['rows'][0]['configuration'];config=lookup[case];context.config=config
        p=json.loads((file.parent.parent/'populations'/file.name).read_text());target=p['summary']['runtime_seconds'];seed=p['summary']['seed']
        count=wall_clock_count(target,cal['seconds_per_path'])
        b=crude(context,crude_seed(seed,f['crude_stream']),study['thresholds'],count=count,
            policy='matched_wall_clock',target_seconds=target)
        atomic_write_json(root/'baselines'/file.name,b)
        rows=rows_for(p,b,study)
        metadata={k:old['rows'][0][k] for k in ['configuration','configuration_label','levels','moves_per_level','schedule_name']}
        for row in rows:row.update(**metadata,clock_target_seconds=target,
            calibration_seconds_per_path=cal['seconds_per_path'],
            clock_within_tolerance=abs(row['wall_clock_ratio']-1)<=f['timing_tolerance_fraction'])
        atomic_write_json(dest,{'rows':rows,'source_population_file':str(file.parent.parent/'populations'/file.name),
            'source_SMC_runtime_seconds':target,'sample_count_fixed_before_final_samples':True})
        print(f"COMPLETED {case} seed={seed} target_s={target:.3f} crude_s={b['runtime_seconds']:.3f} n={count}",flush=True)

def report(path,output_root):
    f,study,configs=load_design(path);root=Path(output_root)/f['output_subdirectory']
    files=sorted((root/'workers').glob('worker_*/runs/*.json'))
    if not files:return
    rows=[row for file in files for row in json.loads(file.read_text())['rows']]
    keys=[(r['configuration'],r['seed'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Repeated run')
    write_csv(root/'runs.csv',rows);summary=[]
    for case,a in sorted({(r['configuration'],r['threshold']) for r in rows}):
        group=[r for r in rows if (r['configuration'],r['threshold'])==(case,a)];first=group[0]
        for method in ['smc','snis','crude']:
            x=np.array([r[method] for r in group]);ratio=x/first['p_reference']
            summary.append({k:first[k] for k in ['configuration','configuration_label','N','threshold','p_reference']}|
                {'method':method,'runs':len(x),'zero_percent':float(100*(x==0).mean()),
                 'median_log_relative_error':float(np.median(np.log10(1+abs(ratio-1)))),
                 'mean_SMC_seconds':float(np.mean([r['smc_runtime_seconds'] for r in group])),
                 'mean_crude_seconds':float(np.mean([r['crude_runtime_seconds'] for r in group])),
                 'mean_crude_paths':float(np.mean([r['crude_n'] for r in group])),
                 **{f'ratio_q{int(q*100):02d}':float(np.quantile(ratio,q)) for q in [.1,.25,.5,.75,.9,.99]}})
    write_csv(root/'summary.csv',summary)
    from .exploration_figures import svg_figure
    unique=[r for r in rows if r['threshold']==.9];counts={c['_exploration']['id']:sum(r['configuration']==c['_exploration']['id'] for r in unique) for c in configs}
    ratios=[r['wall_clock_ratio'] for r in unique]
    body=f'<h1>RollBack SMC: confronto a wall clock abbinato</h1><p><b>{len(files)}/24 run.</b> Tempi e popolazioni SMC riutilizzati dal job 2076203; nuovi campioni crude indipendenti. Nessuna run SMC ripetuta.</p>'
    body+='<p>Stesso nodo gpu007 e GPU fisiche delle misure SMC precedenti. Warm-up e tre calibrazioni indipendenti per GPU; sincronizzazione prima e dopo ogni misura. Il numero di campioni crude è fissato prima di osservarli. Il matching temporale è approssimato: nessuna stima viene filtrata o ripetuta in base al risultato.</p>'
    body+=f'<p>Rapporto tempo crude/SMC: mediana {np.median(ratios):.4f}, minimo {min(ratios):.4f}, massimo {max(ratios):.4f}. Scarti per run nel CSV. Caricamento modelli, warm-up, calibrazione e salvataggio file esclusi dalle misure per stima e documentati separatamente.</p>'
    body+='<p>SMC e SNIS condividono popolazione e tempo. Tutte le soglie e configurazioni restano separate. Reference da 200.000 IID: probabilità stimate, non esatte. Otto run per configurazione: risultati esplorativi.</p>'
    for kind in ['log_error','ratio']:body+=svg_figure(rows,kind).replace('crude a pari transizioni','crude a pari tempo')
    body+='<p><a href="runs.csv">Run e tempi</a> · <a href="summary.csv">Riepilogo</a>. Il precedente confronto a pari transizioni è conservato nella directory camera_ready_exploration_v2.</p>'
    (root/'REPORT.html').write_text('<!doctype html><html lang="it"><meta charset="utf-8"><style>body{font:16px Arial;max-width:1500px;margin:30px auto;padding:20px}svg{width:100%;height:auto}</style>'+body+'</html>',encoding='utf-8')
    atomic_write_json(root/'report_status.json',{'completed_runs':len(files),'expected_runs':24,'completed_by_configuration':counts,
        'status':'COMPLETE' if len(files)==24 else 'PARTIAL','comparison':'matched_wall_clock',
        'clock_ratio_median':float(np.median(ratios)),'clock_ratio_min':min(ratios),'clock_ratio_max':max(ratios),
        'SMC_repeated':False})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['run','report']);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--worker-id',type=int,choices=[0,1,2],default=0);a=p.parse_args()
    if a.stage=='run':run(a.study,a.output_root,a.worker_id)
    else:report(a.study,a.output_root)

if __name__=='__main__':main()
