import csv
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_toxicity_smc_experiment.exploratory import study_configs,plan,audit_population,report,assigned_runs
from diffusion_toxicity_smc_experiment.exploratory_kernel import run_scheduled_trajectory_component,validate_schedule
from diffusion_toxicity_smc_experiment.smc import SMCComponentConfig,run_trajectory_resample_move_component
from diffusion_toxicity_smc_experiment.low_budget import crude_seed,estimates

BASE=Path(__file__).resolve().parents[1]
STUDY=BASE/'configs/mdlm_owt_exploration_v2.yaml'

class NativeBinaryPaths:
    predictor='ddpm';mask_token_id=9
    def reset_nfe(self):self.nfe=0
    def sample_trajectory(self,*,batch_size,sequence_length,num_steps,device,generator,**kwargs):
        result=torch.zeros((batch_size,num_steps+2,sequence_length),dtype=torch.long,device=device)
        result[:,-1]=(torch.rand((batch_size,sequence_length),generator=generator,device=device)<.1).long()
        self.nfe+=num_steps+1
        return result,torch.empty(0,dtype=torch.long,device=device)
    def rollout_trajectory_suffix(self,checkpoint_states,*,start_step,num_steps,generator,**kwargs):
        depth=num_steps-start_step
        result=checkpoint_states[:,None].expand(-1,depth+2,-1).clone()
        result[:,-1]=(torch.rand(checkpoint_states.shape,generator=generator,device=checkpoint_states.device)<.1).long()
        self.nfe+=depth+1
        return result

def component():
    return SMCComponentConfig(num_particles=64,num_steps=2,sequence_length=3,lambda_final=3.,
        ess_threshold=.5,resampling_method='systematic',final_resampling=False,seed=123,
        tilted_sampler_type='trajectory_resample_move',num_tempering_levels=3,mcmc_steps_per_level=4,
        mutation_policy='every_level',rollback_steps=(1,2),rollback_probabilities=(.5,.5))

def kernel_args():
    return dict(sampler=NativeBinaryPaths(),fixed_prefix_token_ids=None,
        reward_fn=lambda x:torch.where(x[:,-1]==1,math.log(.99),math.log(1e-4)).double(),
        event_fn=lambda x:x[:,-1]==1,config=component(),device='cpu')

@pytest.mark.parametrize('levels',[3,6])
def test_explicit_linear_schedule_matches_original_kernel_exactly(levels):
    old_args=kernel_args();old_args['config']=replace(old_args['config'],num_tempering_levels=levels)
    new_args=kernel_args();new_args['config']=replace(new_args['config'],num_tempering_levels=levels)
    old=run_trajectory_resample_move_component(**old_args)
    new=run_scheduled_trajectory_component(**new_args,schedule=[3*k/levels for k in range(1,levels+1)])
    for key in ['terminal_tokens','rewards','events','final_log_weights','root_ancestors']:
        assert torch.equal(getattr(old,key),getattr(new,key))
    assert old.log_normalizer_estimate==new.log_normalizer_estimate
    assert old.genealogy==new.genealogy and old.mutation_diagnostics==new.mutation_diagnostics

@pytest.mark.parametrize('grid',[[0,1,3],[.1,.1,3],[.1,1,2],[.1,3],[.1,float('nan'),3]])
def test_invalid_schedules_fail_before_sampling(grid):
    with pytest.raises(ValueError):validate_schedule(grid,component())

def test_normalizer_ledger_covers_resampling_and_no_resampling():
    result=run_scheduled_trajectory_component(**kernel_args(),schedule=[1,2,3])
    assert any(d['resampled'] for d in result.level_diagnostics)
    assert any(not d['resampled'] for d in result.level_diagnostics)
    assert not result.level_diagnostics[-1]['resampled']
    p={'summary':{'log_normalizer_estimate':result.log_normalizer_estimate},
       'level_diagnostics':result.level_diagnostics,'mutation_diagnostics':result.mutation_diagnostics}
    config={'smc':{'num_particles':64},'_exploration':{'schedule':[1,2,3],'moves_per_level':4}}
    assert audit_population(p,config)
    assert result.log_normalizer_estimate==pytest.approx(sum(d['log_normalizer_increment'] for d in result.level_diagnostics))
    p['level_diagnostics'][1]['log_normalizer_increment']+=.1
    with pytest.raises(AssertionError,match='normalizer'):audit_population(p,config)

def test_design_is_frozen_and_new_seeds_and_costs_are_correct():
    study,configs=study_configs(STUDY);cases=plan(STUDY)['configurations']
    assert [(x['N'],x['levels'],x['moves_per_level']) for x in cases]==[(64,6,4),(64,6,8),(128,6,4)]
    assert cases[1]['expected_transitions']==pytest.approx(56793.6)
    assert cases[0]['expected_transitions']==pytest.approx(29452.8)
    assert cases[2]['expected_transitions']==pytest.approx(58905.6)
    seeds=[c['smc']['base_seed']+1000000*i for i in range(8) for c in configs]
    all_seeds=seeds+[crude_seed(s,925081) for s in seeds]
    assert len(set(all_seeds))==48 and min(seeds)>=1400000000
    for c in configs:
        m=c['smc']['tilted_sampler']
        assert m['rollback_steps']==[4,8,16,24,32]
        assert m['rollback_probabilities']==[.2]*5
        assert c['smc']['resampling_method']=='systematic' and c['smc']['ess_threshold']==.5
        assert c['_exploration']['schedule']==[.5,1.,1.5,2.,2.5,3.]

def test_report_and_plotter_do_not_pool_equal_N_configurations(tmp_path):
    study,configs=study_configs(STUDY);root=tmp_path/study['output_subdirectory'];(root/'runs').mkdir(parents=True)
    for config in configs:
        c=config['_exploration'];rows=[]
        for a,p in [(.9,.00758),(.99,.00222)]:
            rows.append({'configuration':c['id'],'configuration_label':c['label'],'N':c['N'],
                'levels':len(c['schedule']),'moves_per_level':c['moves_per_level'],'threshold':a,
                'p_reference':p,'seed':config['smc']['base_seed'],'comparison':'matched_transitions',
                'lambda_final':3,'smc':0.,'snis':0.,'crude':0.,'native_transitions':3300,'crude_n':100})
        (root/'runs'/f"{c['id']}.json").write_text(json.dumps({'rows':rows}))
    report(STUDY,tmp_path)
    summaries=list(csv.DictReader((root/'summary.csv').open()))
    assert len(summaries)==18 and all(x['runs']=='1' and float(x['zero_percent'])==100 for x in summaries)
    html=(root/'REPORT.html').read_text();assert html.count('<svg ')==2
    path=BASE/'plots/plot_camera_ready_exploration.py';spec=importlib.util.spec_from_file_location('exploration_plot',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=mod.make_figures(root/'runs.csv',root,title_prefix='SYNTHETIC LAYOUT TEST — ')
    assert len(out)==2
    audit=json.loads((root/'exploration_plot_audit.json').read_text())
    assert len(audit['groups'])==36
    assert {x['configuration'] for x in audit['groups']}=={'A','B','C'}
    assert all(x['runs']==1 and x['zeros']==1 for x in audit['groups'])

@pytest.mark.parametrize('workers',[2,3])
def test_gpu_partitions_cover_each_run_once_and_keep_original_seeds(workers):
    study,configs=study_configs(STUDY)
    serial={(c['_exploration']['id'],s) for c,s in assigned_runs(study,configs)}
    shards=[{(c['_exploration']['id'],s) for c,s in assigned_runs(study,configs,i,workers)} for i in range(workers)]
    assert set.union(*shards)==serial and sum(map(len,shards))==24
    assert all(shards[i].isdisjoint(shards[j]) for i in range(workers) for j in range(i))
    assert sorted(map(len,shards))==([12,12] if workers==2 else [6,9,9])

def test_merged_report_reads_independent_worker_directories(tmp_path):
    study,configs=study_configs(STUDY);root=tmp_path/study['output_subdirectory']
    for rank in range(3):
        directory=root/'workers'/f'worker_{rank}'/'runs';directory.mkdir(parents=True)
        for config,seed in assigned_runs(study,configs,rank,3):
            c=config['_exploration'];rows=[]
            for a,p in [(.9,.00758),(.99,.00222)]:
                rows.append({'configuration':c['id'],'configuration_label':c['label'],'N':c['N'],
                    'levels':6,'moves_per_level':c['moves_per_level'],'threshold':a,'p_reference':p,
                    'seed':seed,'comparison':'matched_transitions','smc':0.,'snis':0.,'crude':0.,'native_transitions':3300,'crude_n':100})
            (directory/f'{seed}.json').write_text(json.dumps({'rows':rows}))
    report(STUDY,tmp_path)
    rows=list(csv.DictReader((root/'runs.csv').open()));summary=list(csv.DictReader((root/'summary.csv').open()))
    assert len(rows)==48 and len({r['seed'] for r in rows})==24
    assert len(summary)==18 and all(int(r['runs'])==8 for r in summary)
    assert json.loads((root/'report_status.json').read_text())['status']=='COMPLETE'

@pytest.mark.parametrize('workers',[1,3])
def test_driver_interleaves_cases_and_resumes_saved_runs(tmp_path,monkeypatch,workers):
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    import diffusion_toxicity_smc_experiment.exploratory_population as recorder
    import diffusion_toxicity_smc_experiment.exploratory as driver
    calls=[]
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(gpu,'Context',lambda config:SimpleNamespace(config=config))
    def population(context,config,study,seed):
        n=config['smc']['num_particles'];case=config['_exploration'];r=len(case['schedule']);m=case['moves_per_level']
        calls.append(('SMC',case['id'],seed))
        summary={'num_particles':n,'seed':seed,'log_normalizer_estimate':0.,'total_decoder_particle_steps':n*(32+r*m*16),'mcmc_proposals':n*r*m,
            'mcmc_acceptance_rate':.5,'distinct_root_ancestors':n,'distinct_terminal_sequences':n,'final_smc_ess':n,'runtime_seconds':1.}
        return {'summary':summary,'particles':[{'root_ancestor':i,'event_toxic_probability':.1} for i in range(n)],
            'corrections':{str(a):estimates([-math.log(n)]*n,[math.log(.1)]*n,0.,[.1]*n,a,3.) for a in study['thresholds']}}
    def crude(context,seed,thresholds,*,count,policy='matched_transitions',**kwargs):
        calls.append(('MC',seed));return {'seed':seed,'n':count,'scores':[.1]*count,'native_transitions':33*count,'runtime_seconds':.1,
            'policy':policy,'overshoot_seconds':None,'estimates':{str(a):0. for a in thresholds}}
    monkeypatch.setattr(recorder,'population',population);monkeypatch.setattr(gpu,'crude',crude);monkeypatch.setattr(driver,'audit_population',lambda *args:True)
    for rank in range(workers):driver.run(STUDY,tmp_path,worker_id=rank,workers=workers)
    driver.report(STUDY,tmp_path)
    smc=[x for x in calls if x[0]=='SMC']
    main=[x for x in smc if x[2]>=1400000000]
    assert len(smc)==24+workers and len(main)==24 and [x[1] for x in main]==['A','B','C']*8
    root=tmp_path/'camera_ready_exploration_v2';rows=list(csv.DictReader((root/'runs.csv').open()))
    assert len(rows)==48 and len({r['seed'] for r in rows})==24
    assert len({r['crude_seed'] for r in rows})==24
    count=len(calls)
    for rank in range(workers):driver.run(STUDY,tmp_path,worker_id=rank,workers=workers)
    assert len(calls)==count

def test_parallel_SLURM_worker_rejects_multiple_visible_GPUs_before_model_load(tmp_path,monkeypatch):
    import diffusion_toxicity_smc_experiment.exploratory as driver
    monkeypatch.setenv('SLURM_JOB_ID','test')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:3)
    with pytest.raises(RuntimeError,match='exactly one GPU'):
        driver.run(STUDY,tmp_path,worker_id=0,workers=3)
