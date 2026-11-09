"""Potential selection, exact finite targets, and the actual checkpoint path."""
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_toxicity_smc_experiment.potentials import ToxicityPotential, potential_from_config
from diffusion_toxicity_smc_experiment.run_smc import UnconditionalToxicityFunctions
from diffusion_toxicity_smc_experiment.low_budget import estimates

BASE=Path(__file__).resolve().parents[1]


def test_imported_sampler_comes_from_this_source_snapshot():
    import diffusion_toxicity_smc_experiment as package
    assert (BASE/'src').resolve() in Path(package.__file__).resolve().parents


def test_old_phi_preserves_saved_reward_exactly_and_new_phi_uses_logits():
    row=dict(reward=-0.123456789,logits=[0.,3.],toxic_probability=1.)
    assert ToxicityPotential().from_classifier_row(row)==row['reward']
    assert ToxicityPotential('logit').from_classifier_row(row)==3.
    # Finite logits remain usable even when float32 softmax has rounded q to 1.
    row['logits']=[-100.,100.]
    assert ToxicityPotential('logit').from_classifier_row(row)==200.
    assert ToxicityPotential('logit').probability(-1000.)==0.
    assert ToxicityPotential('logit').probability(1000.)==1.
    with pytest.raises(ValueError):ToxicityPotential('unknown')
    with pytest.raises(FloatingPointError):
        ToxicityPotential('logit').from_classifier_row(dict(logits=[0.,float('inf')]))


def test_observed_downward_move_acceptance_and_monotone_transform():
    high=.99955579;low=.99927776
    old=ToxicityPotential();new=ToxicityPotential('logit')
    assert math.exp(2.5*(old.threshold_value(low)-old.threshold_value(high)))==pytest.approx(.999304761)
    assert math.exp(2.5*(new.threshold_value(low)-new.threshold_value(high)))==pytest.approx(.296459206)
    assert math.exp(8*(new.threshold_value(low)-new.threshold_value(high)))==pytest.approx(.020430915)
    assert min(1.,math.exp(2.5*(new.threshold_value(high)-new.threshold_value(low))))==1.


@pytest.mark.parametrize('name',['log_probability','logit'])
def test_finite_native_rollback_MH_balance_and_both_reverse_identities(name):
    # One full rollback refreshes from native P. Enumerate the transition
    # matrix rather than judging invariance from a noisy histogram.
    q=np.array([.05,.9,.99927776,.99955579]);P=np.array([.7,.28,.015,.005])
    potential=ToxicityPotential(name)
    phi=np.array([potential.threshold_value(v) for v in q])
    previous=0.;previous_pi=P.copy();previous_z=1.
    for tilt in [.5,1.,2.5]:
        gamma=P*np.exp(tilt*phi);z=gamma.sum();pi=gamma/z
        accept=np.minimum(1.,np.exp(tilt*(phi[None,:]-phi[:,None])))
        M=P[None,:]*accept
        M[np.diag_indices(len(q))]+=1.-M.sum(axis=1)
        assert np.allclose(M.sum(axis=1),1.)
        assert np.allclose(pi[:,None]*M,(pi[:,None]*M).T)
        assert np.allclose(pi@M,pi)
        G=np.exp((tilt-previous)*phi)
        ratio=previous_pi@G
        assert previous_z*ratio==pytest.approx(z)
        assert np.allclose((previous_pi*G/ratio)@M,pi)
        result=estimates(np.log(pi),phi,math.log(z),q,.9995,tilt)
        assert result['smc']==pytest.approx(.005)
        assert result['snis']==pytest.approx(.005)
        assert result['reverse_mass']==pytest.approx(1.)
        previous,previous_pi,previous_z=tilt,pi,z


def test_classifier_reward_selection_does_not_change_events_or_add_forwards():
    class Tokenizer:
        def batch_decode(self,rows,**kw):return [' '.join(map(str,row)) for row in rows]
        def decode(self,row,**kw):return ' '.join(map(str,row))
    class Evaluator:
        positive_label_index=1
        calls=0
        def score_texts(self,texts,**kw):
            self.calls+=1
            return [dict(logits=[-2.,3.],reward=-.0067153485,
                         toxic_probability=.993307173,is_event=True) for _ in texts]
    outputs=[]
    for phi in ['log_probability','logit']:
        e=Evaluator()
        f=UnconditionalToxicityFunctions(generator_tokenizer=Tokenizer(),evaluator=e,
            reward_completion_tokens=100,event_completion_tokens=100,
            reward_max_length=512,event_max_length=512,
            event_text_scope='first_generated_mdlm_tokens_only',
            completion_trimming_tokenizer=None,phi=phi)
        rewards=f.reward(torch.arange(128)[None]);events=f.event(torch.arange(128)[None])
        assert e.calls==1 and f.last_event_rows is f.last_reward_rows
        outputs.append((rewards,events,f.last_event_rows))
    assert outputs[0][0].item()==-.0067153485 and outputs[1][0].item()==5.
    assert torch.equal(outputs[0][1],outputs[1][1])
    assert outputs[0][2]==outputs[1][2]


class LogitFunctions:
    potential=ToxicityPotential('logit')
    def reward(self,x):
        q=[.9997 if int(v)==1 else .0001 for v in x[:,-1]]
        self.last_event_rows=[dict(toxic_probability=v) for v in q]
        return torch.tensor([self.potential.threshold_value(v) for v in q],dtype=torch.float64)
    def event(self,x):
        self.reward(x)
        return x[:,-1]==1


def test_actual_recorder_uses_sigmoid_for_events_and_current_logit_for_correction(monkeypatch):
    from test_exploratory import NativeBinaryPaths,component
    from diffusion_toxicity_smc_experiment import exploratory_population as recorder
    from diffusion_toxicity_smc_experiment.exploratory import audit_population
    monkeypatch.setattr(recorder,'component_config',lambda *_:component())
    context=SimpleNamespace(functions=LogitFunctions(),sampler=NativeBinaryPaths(),
                            device=torch.device('cpu'),synchronize=lambda:None)
    config={'smc':{'num_particles':64,'phi':'logit'},
            'decoder':{'denoising_steps':2},
            '_exploration':{'schedule':[1.,2.,3.],'moves_per_level':4}}
    study=dict(phi='logit',thresholds=[.99,.999],record_population_checkpoints=True)
    result=recorder.population(context,config,study,123)
    assert audit_population(result,config)
    assert result['phi']['name']=='logit'
    assert len(result['population_checkpoints'])==13
    for cp in result['population_checkpoints']:
        assert cp['phi']=='logit'
        assert all(0<=q<=1 for q in cp['scores'])
        assert np.allclose(cp['scores'],[ToxicityPotential('logit').probability(v) for v in cp['rewards']])
        for a in study['thresholds']:
            value=estimates(cp['normalized_log_weights'],cp['rewards'],cp['log_z_hat'],cp['scores'],a,cp['lambda'])
            assert cp['corrections'][str(a)]==value
    assert result['population_checkpoints'][-1]['corrections']==result['corrections']
    assert not result['population_checkpoints'][-1]['resampled_at_level']
    assert any(cp['resampled_at_level'] for cp in result['population_checkpoints'])


def test_logit_study_changes_only_phi_and_preserves_seeds_and_native_reference():
    from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs,assigned_runs
    from diffusion_toxicity_smc_experiment.configuration import decoder_id,evaluator_id
    old,old_configs=study_configs(BASE/'configs/mdlm_owt_same_run_ladder.yaml')
    new,new_configs=study_configs(BASE/'configs/mdlm_owt_same_run_logit.yaml')
    assert new['reuse_only_MC'] and new['independent_runs']==12
    assert [s for _,s in assigned_runs(old,old_configs,0,1)]==[s for _,s in assigned_runs(new,new_configs,0,1)]
    assert decoder_id(old_configs[0])==decoder_id(new_configs[0])
    assert evaluator_id(old_configs[0])==evaluator_id(new_configs[0])
    normalized=copy.deepcopy(new_configs[0]);normalized['smc'].pop('phi')
    assert normalized==old_configs[0]
    assert potential_from_config(new_configs[0]).name=='logit'
    assert new['output_subdirectory']!=old['output_subdirectory']


def test_fresh_driver_saves_each_seed_resumes_and_never_calls_crude(tmp_path,monkeypatch):
    from test_exploratory import NativeBinaryPaths
    from diffusion_toxicity_smc_experiment import same_run_reuse as runner
    from diffusion_toxicity_smc_experiment import camera_ready_gpu as gpu
    study,configs=runner.study_configs(BASE/'configs/mdlm_owt_same_run_logit.yaml')
    study=copy.deepcopy(study);study['independent_runs']=1
    monkeypatch.setattr(runner,'study_configs',lambda _: (study,configs))
    for key in ['SLURM_JOB_ID','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'cpu-test')
    device=dict(hostname='cpu-test',SLURM_STEP_GPUS=None,CUDA_VISIBLE_DEVICES=None,
                SLURM_CPUS_PER_TASK=None)
    cal_root=tmp_path/'old';cal_path=cal_root/'workers/worker_0/calibration_test.json'
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps(dict(seconds_per_path=.172,timing_device=device,
        excluded_from_estimates_and_measured_run_times=True)))
    loads=[]
    def context(config):
        loads.append(config)
        return SimpleNamespace(functions=LogitFunctions(),sampler=NativeBinaryPaths(),
            device=torch.device('cpu'),synchronize=lambda:None,config=config)
    monkeypatch.setattr(gpu,'Context',context)
    def forbidden(*args,**kwargs):raise AssertionError('No new crude MC permitted')
    monkeypatch.setattr(gpu,'crude',forbidden)
    for _ in range(2):
        runner.run_fresh('unused',tmp_path,0,3,6300,cal_root)
    assert len(loads)==1
    worker=tmp_path/study['output_subdirectory']/'workers/worker_0'
    files=list((worker/'populations').glob('*.json'))
    assert len(files)==1
    saved=json.loads(files[0].read_text())
    assert len(saved['population_checkpoints'])==65
    assert saved['phi']['name']=='logit'
    assert json.loads((worker/'manifest.json').read_text())['new_MC_generation'] is False
    pool=tmp_path/'old_MC_pool';baseline=pool/'workers/worker_0/baselines/test.json'
    baseline.parent.mkdir(parents=True)
    (pool/'manifest.json').write_text(json.dumps({'study':{'reference':study['reference']}}))
    (pool/'report_status.json').write_text(json.dumps(dict(completed_runs=1,expected_runs=1)))
    baseline.write_text(json.dumps(dict(n=1000,seed=987,scores=[.1]*1000,
        native_transitions=33000,sample_count_chosen_before_generation=True)))
    runner.report('unused',tmp_path,pool)
    root=tmp_path/study['output_subdirectory']
    with (root/'checkpoint_runs.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==130 and {r['phi'] for r in rows}=={'logit'}
    assert all(float(r['crude'])==0 for r in rows)
    assert json.loads((root/'comparison_policy.json').read_text())['phi']['name']=='logit'
