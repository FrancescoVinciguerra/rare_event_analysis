from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import math
import pytest
import torch

from diffusion_toxicity_smc_experiment.wall_clock_followup import run,report,load_design,source_worker
from diffusion_toxicity_smc_experiment.low_budget import crude_seed

STUDY=Path(__file__).resolve().parents[1]/'configs/mdlm_owt_wall_clock_followup.yaml'

def test_clock_count_selected_before_final_samples_without_repeating_SMC(tmp_path,monkeypatch):
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    f,study,configs=load_design(STUDY)
    source=tmp_path/f['source_subdirectory']/'workers/worker_0'
    (source/'runs').mkdir(parents=True);(source/'populations').mkdir()
    name='A_N64_seed1400000000.json'
    metadata={'configuration':'A','configuration_label':'A: 64 / 6 / 4','levels':6,'moves_per_level':4,'schedule_name':'uniform_6'}
    (source/'runs'/name).write_text(json.dumps({'rows':[metadata]}))
    p={'summary':{'num_particles':64,'seed':1400000000,'runtime_seconds':10.,'total_decoder_particle_steps':30000,
        'mcmc_proposals':1536,'log_normalizer_estimate':-3.,'final_smc_ess':60.,'distinct_root_ancestors':10,
        'distinct_terminal_sequences':64,'mcmc_acceptance_rate':.5},
        'particles':[{'root_ancestor':i,'event_toxic_probability':.95} for i in range(64)],
        'corrections':{str(a):{'smc':.001,'snis':.002,'reverse_mass':.5,'event_particles':64 if a==.9 else 0} for a in [.9,.99]}}
    population=source/'populations'/name;population.write_text(json.dumps(p));before=hashlib.sha256(population.read_bytes()).hexdigest()
    calls=[]
    monkeypatch.setattr(gpu,'Context',lambda config:SimpleNamespace(config=config))
    def crude(context,seed,thresholds,*,count,policy='matched_transitions',target_seconds=None):
        calls.append((seed,count,target_seconds))
        return {'seed':seed,'n':count,'scores':[.95]+[0.]*(count-1),'runtime_seconds':count*.2,
            'native_transitions':33*count,'policy':policy,'overshoot_seconds':0.,
            'estimates':{str(a):(1/count if a==.9 else 0.) for a in thresholds}}
    monkeypatch.setattr(gpu,'crude',crude)
    run(STUDY,tmp_path,0)
    assert [x[1] for x in calls]==[16,128,128,128,50]
    assert calls[-1][0]==crude_seed(1400000000,925082) and calls[-1][2]==10.
    assert len({x[0] for x in calls})==5
    out=tmp_path/f['output_subdirectory'];saved=json.loads((out/'workers/worker_0/runs'/name).read_text())
    assert saved['sample_count_fixed_before_final_samples']
    assert all(r['smc']==.001 and r['snis']==.002 and r['wall_clock_ratio']==1. for r in saved['rows'])
    assert hashlib.sha256(population.read_bytes()).hexdigest()==before
    run(STUDY,tmp_path,0);assert len(calls)==5
    report(STUDY,tmp_path)
    rows=list(csv.DictReader((out/'runs.csv').open()))
    assert len(rows)==2 and all(r['comparison']=='matched_wall_clock' for r in rows)
    html=(out/'REPORT.html').read_text();assert html.count('<svg ')==2 and 'wall clock' in html

def test_worker_maps_to_original_physical_GPU(tmp_path,monkeypatch):
    import diffusion_toxicity_smc_experiment.wall_clock_followup as module
    monkeypatch.setenv('SLURM_JOB_ID','test');monkeypatch.setenv('SLURM_STEP_GPUS','2')
    monkeypatch.setattr(module.socket,'gethostname',lambda:'gpu007')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    for rank in range(3):
        p=tmp_path/'workers'/f'worker_{rank}';p.mkdir(parents=True)
        (p/'warmup.json').write_text(json.dumps({'SLURM_STEP_GPUS':str(rank)}))
    assert source_worker(tmp_path,0)==2
    monkeypatch.setattr(module.socket,'gethostname',lambda:'gpu008')
    with pytest.raises(RuntimeError,match='gpu007'):source_worker(tmp_path,0)
