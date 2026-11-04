import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_toxicity_smc_experiment import accuracy_budget as module
from diffusion_toxicity_smc_experiment.low_budget import crude_seed,estimates


STUDY=Path(__file__).resolve().parents[1]/'configs/mdlm_owt_accuracy_budget_pilot.yaml'


def test_frozen_model_terminal_schedules_and_all_independent_streams():
    study,configs=module.study_configs(STUDY)
    assert {(c['smc']['num_particles'],tuple(c['_exploration']['schedule'])) for c in configs}=={
        (n,tuple(.5*k for k in range(1,r+1))) for n in [16,32,64] for r in range(1,7)}
    assert study['thresholds']==[.9,.99,.999] and study['plot_thresholds']==[.99,.999]
    assert study['reference']['discoveries']==[1516,444,26] and 'adequacy' not in study
    tasks=[x for rank in range(3) for x in module.assigned_runs(study,configs,rank,3)]
    seeds=[s for c,s in tasks];mc=[crude_seed(s,study['crude_stream']) for s in seeds]
    assert len(tasks)==360 and len(set(seeds+mc))==720
    for c in configs:
        assert c['smc']['final_resampling'] is False
        assert c['smc']['tilted_lambdas']==[c['_exploration']['schedule'][-1]]
    design=module.plan(STUDY)
    assert design['total_SMC_runs']==360 and design['configurations'][0]['expected_native_transitions']==pytest.approx(1667.2)


def test_means_medians_and_zero_counts_include_all_runs_without_accuracy_cutoffs():
    study,_=module.study_configs(STUDY);rows=[]
    for setting,lam in [('low',.5),('high',1.)]:
        for x in [0.,.5,1.,1.5,2.]:
            rows.append(dict(configuration=setting,N=128,levels=int(2*lam),moves_per_level=4,lambda_final=lam,
                threshold=.99,p_reference=.00222,smc=x*.00222,snis=x*.00222,crude=x*.00222,
                smc_runtime_seconds=10.,crude_runtime_seconds=10.,crude_n=50,clock_within_tolerance=True))
    out=module.summarize(rows,study)
    assert len(out)==6
    assert all(r['runs']==5 and r['zero_count']==1 and r['zero_percent']==20 for r in out)
    assert all(r['mean_ratio']==1. and r['median_ratio']==1. for r in out)
    assert not any('adequate' in key for r in out for key in r)
    assert out[0]['mean_log_relative_error']==pytest.approx(np.log10(1+abs(np.array([0,.5,1,1.5,2])-1)).mean())


def test_driver_measures_real_terminal_runs_uses_independent_calibration_and_resumes(tmp_path,monkeypatch):
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    import diffusion_toxicity_smc_experiment.exploratory_population as recorder
    study,configs=module.study_configs(STUDY);study=copy.deepcopy(study);study['independent_runs']=1
    monkeypatch.setattr(module,'study_configs',lambda _: (study,configs))
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(gpu,'Context',lambda config:SimpleNamespace(config=config))
    calls=[]
    def population(context,config,study,seed):
        s=config['smc'];n=s['num_particles'];lam=s['tilted_lambdas'][0]
        assert not s['final_resampling'] and config['_exploration']['schedule'][-1]==lam
        calls.append(('SMC',seed,lam))
        return {'summary':dict(num_particles=n,seed=seed,lambda_final=lam,log_normalizer_estimate=lam*math.log(.95),
            runtime_seconds=10.,total_decoder_particle_steps=n*32,mcmc_proposals=n*4*len(config['_exploration']['schedule']),
            mcmc_acceptance_rate=.5,distinct_root_ancestors=n,distinct_terminal_sequences=n,final_smc_ess=n),
            'particles':[dict(root_ancestor=i,event_toxic_probability=.95) for i in range(n)],
            'corrections':{str(a):estimates([-math.log(n)]*n,[math.log(.95)]*n,lam*math.log(.95),[.95]*n,a,lam) for a in study['thresholds']}}
    def crude(context,seed,thresholds,*,count,policy='matched_transitions',target_seconds=None):
        calls.append(('MC',seed,count,target_seconds))
        return dict(seed=seed,n=count,scores=[.95]*count,native_transitions=33*count,runtime_seconds=count*.2,
            policy=policy,target_seconds=target_seconds,overshoot_seconds=0.,estimates={str(a):float(a<=.95) for a in thresholds})
    monkeypatch.setattr(recorder,'population',population);monkeypatch.setattr(gpu,'crude',crude)
    monkeypatch.setattr(module,'audit_population',lambda *args:True)
    module.run(STUDY,tmp_path,worker_id=0,workers=1)
    finals=[c for c in calls if c[0]=='MC' and c[3] is not None]
    assert len(finals)==18 and all(c[2]==50 and c[3]==10. for c in finals)
    assert len({c[1] for c in calls})==len(calls)
    before=len(calls);module.run(STUDY,tmp_path,worker_id=0,workers=1);assert len(calls)==before
    module.report(STUDY,tmp_path)
    root=tmp_path/study['output_subdirectory'];rows=list(csv.DictReader((root/'runs.csv').open()))
    assert len(rows)==54 and len({r['seed'] for r in rows})==18
    assert {r['lambda_final'] for r in rows}=={'0.5','1.0','1.5','2.0','2.5','3.0'}
    assert {r['threshold'] for r in rows}=={'0.9','0.99','0.999'}
    assert all(r['analysis_role']==('control' if r['threshold']=='0.9' else 'rare_event') for r in rows)
    assert all(r['timing_basis']=='measured_complete_terminal_run' and r['clock_within_tolerance']=='True' for r in rows)
    assert json.loads((root/'report_status.json').read_text())['status']=='PILOT_COMPLETE'
    # An incomplete run may only pair its SMC timing with the same physical GPU.
    final=next((root/'workers/worker_0/runs').glob('*.json'));final.unlink()
    (root/'workers/worker_0/baselines'/final.name).unlink()
    pp=root/'workers/worker_0/populations'/final.name;p=json.loads(pp.read_text());p['timing_device']['hostname']='another-node';pp.write_text(json.dumps(p))
    with pytest.raises(RuntimeError,match='another physical GPU'):
        module.run(STUDY,tmp_path,worker_id=0,workers=1)
