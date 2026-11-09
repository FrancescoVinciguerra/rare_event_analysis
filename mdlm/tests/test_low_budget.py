import math
import csv
import json
from pathlib import Path

import pytest
import torch
import importlib.util

from diffusion_toxicity_smc_experiment.low_budget import (
    estimates, transition_budget, plan, study_configs, crude_seed, report,
)
from diffusion_toxicity_smc_experiment.configuration import validate_benchmark_config

STUDY=Path(__file__).resolve().parents[1]/'configs/mdlm_owt_camera_ready_low_budget.yaml'


@pytest.mark.parametrize('threshold,expected',[(.9,.2),(.99,.05),(1.,0.)])
def test_exact_finite_target_reverse_corrections(threshold,expected):
    p=torch.tensor([.8,.15,.05],dtype=torch.float64)
    q=torch.tensor([.1,.95,.995],dtype=torch.float64)
    z=(p*q).sum()
    answer=estimates((p*q/z).log(),q.log(),float(z.log()),q,threshold,1.)
    assert answer['smc']==pytest.approx(expected)
    assert answer['snis']==pytest.approx(expected)
    assert answer['reverse_mass']==pytest.approx(1.)


def test_normalized_weights_required():
    with pytest.raises(ValueError,match='normalized'):
        estimates([0.,0.],[-1.,-2.],0.,[.9,.99],.9,1.)


def test_budget_counts_rejected_proposals_and_terminal_removal():
    # Four proposals, regardless of acceptance; every proposal has one final draw.
    s={'num_particles':2,'total_decoder_particle_steps':64+4+8+16+32,'mcmc_proposals':4}
    cost=transition_budget(s)
    assert cost['native_transitions']==130
    assert cost['terminal_categorical_transitions']==6
    assert cost['crude_n']==3
    assert cost['unused_transitions']==31


def test_frozen_ids_plan_and_separate_streams():
    study,configs=study_configs(STUDY)
    assert [c['smc']['num_particles'] for c in configs]==[16,32,64,128,256]
    sampler_seeds=[]
    for config in configs:
        validate_benchmark_config(config)
        s=config['smc']; m=s['tilted_sampler']
        assert s['tilted_lambdas']==[3.]
        assert s['tilted_component_weights_a']==[1.]
        assert not s['final_resampling']
        assert s['resampling_method']=='systematic'
        assert m['num_tempering_levels']==3 and m['mcmc_steps_per_level']==4
        sampler_seeds.extend(s['base_seed']+1000000*j for j in range(study['independent_runs']))
    seeds=sampler_seeds+[crude_seed(s,stream) for s in sampler_seeds for stream in (924071,924072)]
    assert len(seeds)==len(set(seeds))
    values=plan(STUDY)['configurations']
    assert values[0]['expected_native_transitions']==pytest.approx(3945.6)
    assert values[-1]['expected_native_transitions']==pytest.approx(63129.6)


def test_report_refuses_to_invent_results(tmp_path):
    with pytest.raises(ValueError,match='No completed'):
        report(STUDY,tmp_path)


def test_report_keeps_zeros_in_each_threshold_and_method(tmp_path):
    root=tmp_path/'camera_ready_low_budget_v1'
    directory=root/'runs'
    directory.mkdir(parents=True)
    rows=[{'N':19,'seed':1200000000,'threshold':a,'p_reference':p,
           'smc':0.,'snis':0.,'crude':0.,'native_transitions':3333,'crude_n':101}
          for a,p in [(.9,.00758),(.99,.00222)]]
    (directory/'N19_seed1200000000.json').write_text(json.dumps({'rows':rows}))
    report(STUDY,tmp_path)
    with (root/'summary.csv').open() as handle:
        summaries=list(csv.DictReader(handle))
    assert len(summaries)==6
    assert all(float(row['zero_percent'])==100 for row in summaries)
    assert all(float(row['median_log_relative_error'])==pytest.approx(math.log10(2)) for row in summaries)
    assert all(float(row['ratio_q99'])==0 for row in summaries)


def plot_module():
    path=Path(__file__).resolve().parents[1]/'reproduction/plots/plot_camera_ready_low_budget.py'
    spec=importlib.util.spec_from_file_location('camera_ready_plot',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_statistics_keep_zero_runs_and_extreme_overestimates():
    module=plot_module()
    values=[0.,0.,0.,0.,100.]
    result=module.statistics(values)
    assert result['med']==0 and result['q3']==0
    assert result['mean']==20
    assert result['fliers'].tolist()==[100.]
    assert module.metric([0],.00222,'log_error')[0]==pytest.approx(math.log10(2))
    assert module.metric([0],.00222,'ratio')[0]==0


def test_two_summary_figures_from_every_independent_run(tmp_path):
    module=plot_module()
    path=tmp_path/'layout_test_runs.csv'
    rows=[]
    for a,p in [(.9,.00758),(.99,.00222)]:
        for n in [16,32,64,128,256]:
            for seed,ratio in enumerate([0,0,.5,.8,1,1.2,1.5,2,5,1000]):
                rows.append({'N':n,'seed':seed,'threshold':a,'p_reference':p,
                    'comparison':'matched_transitions','lambda_final':3,
                    'smc':ratio*p,'snis':min(1.,ratio*p*2),'crude':min(1.,ratio*p)})
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    outputs=module.make_figures(path,tmp_path,title_prefix='LAYOUT TEST — SYNTHETIC VALUES\n')
    assert len(outputs)==2 and all(p.stat().st_size>10000 for p in outputs)
    audit=json.loads((tmp_path/'camera_ready_plot_audit.json').read_text())
    assert audit['source_rows']==100 and audit['zeros_included']
    assert len(audit['groups'])==60
    assert all(g['runs']==10 and g['zeros']==2 and g['outside_zoom']>0 for g in audit['groups'])
    assert max(g['maximum'] for g in audit['groups'] if g['kind']=='ratio')==1000


def test_clock_and_transition_rows_are_not_pooled_in_report(tmp_path):
    root=tmp_path/'camera_ready_low_budget_v1';directory=root/'runs';directory.mkdir(parents=True)
    rows=[]
    for policy in ['matched_transitions','matched_wall_clock']:
        for a,p in [(.9,.00758),(.99,.00222)]:
            rows.append({'N':16,'seed':1200000000,'threshold':a,'p_reference':p,
                'comparison':policy,'smc':p,'snis':0.,'crude':0.,
                'native_transitions':3900,'crude_n':118 if policy=='matched_transitions' else 200,
                'wall_clock_ratio':1.02,'smc_runtime_seconds':40.,
                'distinct_root_ancestors':8,'distinct_terminal_sequences':16})
    (directory/'N16_seed1200000000.json').write_text(json.dumps({'rows':rows}))
    report(STUDY,tmp_path)
    status=json.loads((root/'report_status.json').read_text())
    assert status['completed_populations']==1 and status['expected_populations']==40
    with (root/'summary.csv').open() as handle: summaries=list(csv.DictReader(handle))
    assert len(summaries)==12 and all(r['runs']=='1' for r in summaries)
    html=(root/'REPORT.html').read_text(encoding='utf-8')
    assert html.count('<svg ')==2 and 'PILOT PARZIALE' in html
    assert 'pari transizioni' in html and 'pari wall clock nominale' in html


def test_gpu_driver_uses_agreed_existing_kernel_configuration():
    from diffusion_toxicity_smc_experiment.camera_ready_gpu import component_config
    _,configs=study_configs(STUDY)
    for config in configs:
        c=component_config(config,123)
        assert c.tilted_sampler_type=='trajectory_resample_move'
        assert c.num_steps==32 and c.lambda_final==3
        assert c.num_tempering_levels==3 and c.mcmc_steps_per_level==4
        assert c.rollback_steps==(4,8,16,24,32)
        assert c.final_resampling is False and c.resampling_method=='systematic'


def test_timed_crude_produces_complete_native_paths_and_keeps_zeros(monkeypatch):
    from types import SimpleNamespace
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    calls=[]
    class Sampler:
        nfe=0
        def reset_nfe(self): self.nfe=0
        def sample_trajectory(self,**kwargs):
            calls.append(kwargs['batch_size']);self.nfe+=33
            return torch.zeros((kwargs['batch_size'],34,128),dtype=torch.long),None
    class Functions:
        def reward(self,tokens): self.last_event_rows=[{'toxic_probability':.1} for _ in tokens]
        def event(self,tokens): pass
    context=SimpleNamespace(device=torch.device('cpu'),sampler=Sampler(),functions=Functions(),
        config={'decoder':{'sequence_length':128,'denoising_steps':32},'smc':{'microbatch_size':16}},
        synchronize=lambda:None)
    tick=iter(np*.05 for np in range(1000))
    monkeypatch.setattr(gpu.time,'perf_counter',lambda:next(tick))
    result=gpu.crude(context,345,[.9,.99],count=100)
    assert calls==[100] and result['native_transitions']==3300
    assert result['estimates']=={'0.9':0.,'0.99':0.}
    count=gpu.wall_clock_count(40.,.2)
    assert count==200
    result=gpu.crude(context,346,[.9,.99],count=count,policy='matched_wall_clock',target_seconds=40.)
    assert result['policy']=='matched_wall_clock' and result['n']==200
    assert result['sample_count_chosen_before_generation']


def test_paired_driver_checkpoints_and_resumes_without_regeneration(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    counts={'population':0,'crude':0,'context':0}
    def context(config):
        counts['context']+=1
        return SimpleNamespace(config=config)
    def fake_population(context,config,study,seed):
        counts['population']+=1;n=config['smc']['num_particles']
        particles=[{'root_ancestor':i,'token_ids':[0], 'reward':math.log(.1),
            'event_toxic_probability':.1,'final_log_weight':-math.log(n)} for i in range(n)]
        summary={'num_particles':n,'seed':seed,'log_normalizer_estimate':0.,
            'total_decoder_particle_steps':n*(32+12*16),'mcmc_proposals':12*n,
            'mcmc_acceptance_rate':.5,'distinct_root_ancestors':n,
            'distinct_terminal_sequences':1,'final_smc_ess':n,'runtime_seconds':1.}
        return {'summary':summary,'particles':particles,'corrections':{
            str(a):estimates([-math.log(n)]*n,[math.log(.1)]*n,0.,[.1]*n,a,3.) for a in study['thresholds']}}
    def fake_crude(context,seed,thresholds,*,count,policy='matched_transitions',target_seconds=None):
        counts['crude']+=1
        return {'seed':seed,'n':count,'scores':[.1]*count,'native_transitions':33*count,
            'runtime_seconds':.1*count,'policy':policy,'overshoot_seconds':0.,
            'estimates':{str(a):0. for a in thresholds}}
    monkeypatch.setattr(gpu,'Context',context)
    monkeypatch.setattr(gpu,'population',fake_population)
    monkeypatch.setattr(gpu,'crude',fake_crude)
    gpu.run(STUDY,tmp_path)
    assert counts=={'context':1,'population':41,'crude':84}
    root=tmp_path/'camera_ready_low_budget_v1'
    with (root/'runs.csv').open() as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==160 and len({r['seed'] for r in rows})==40
    assert len({r['crude_seed'] for r in rows})==80
    status=json.loads((root/'report_status.json').read_text())
    assert status['completed_populations']==40 and status['status']=='PILOT COMPLETO'
    gpu.run(STUDY,tmp_path)
    assert counts=={'context':1,'population':41,'crude':84}
