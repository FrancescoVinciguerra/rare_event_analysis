"""Observing the same SMC run must preserve its states, weights and RNG."""
import math
from types import SimpleNamespace

import pytest
import torch

from test_exploratory import kernel_args
from diffusion_toxicity_smc_experiment.exploratory_kernel import run_scheduled_trajectory_component
from diffusion_toxicity_smc_experiment.low_budget import estimates
from diffusion_toxicity_smc_experiment.population_checkpoints import first_event_checkpoint


def test_each_MH_checkpoint_preserves_sampler_and_counts_rejected_proposals():
    plain = run_scheduled_trajectory_component(**kernel_args(), schedule=[1., 2., 3.])
    checkpoints = []

    def observe(**state):
        state = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                 for k, v in state.items()}
        checkpoints.append(state)

    observed = run_scheduled_trajectory_component(
        **kernel_args(), schedule=[1., 2., 3.], population_observer=observe)
    for field in ['terminal_tokens', 'rewards', 'events', 'final_log_weights', 'root_ancestors']:
        assert torch.equal(getattr(plain, field), getattr(observed, field))
    assert plain.log_normalizer_estimate == observed.log_normalizer_estimate
    assert plain.genealogy == observed.genealogy
    assert plain.mutation_diagnostics == observed.mutation_diagnostics
    assert len(checkpoints) == 1 + 3 * 4
    expected_cost = 64 * 3  # two native steps and categorical completion
    assert checkpoints[0]['native_transitions'] == expected_cost
    for checkpoint, mutation in zip(checkpoints[1:], observed.mutation_diagnostics):
        expected_cost += mutation['rollback_particle_steps'] + mutation['proposals']
        assert checkpoint['native_transitions'] == expected_cost
        assert checkpoint['sweep'] == mutation['sweep'] + 1
        assert checkpoint['target_lambda'] == mutation['target_lambda']
    for level in [1, 2, 3]:
        group = [c for c in checkpoints if c['level'] == level]
        assert all(torch.equal(c['normalized_log_weights'], group[0]['normalized_log_weights']) for c in group)
        assert all(c['log_normalizer_estimate'] == group[0]['log_normalizer_estimate'] for c in group)
    assert any(c['resampled'] for c in checkpoints)
    assert all(not c['resampled'] for c in checkpoints if c['level'] == 3)
    last = checkpoints[-1]
    result = estimates(last['normalized_log_weights'], last['rewards'],
        last['log_normalizer_estimate'], torch.exp(last['rewards']), .9,
        last['target_lambda'])
    direct = estimates(observed.final_log_weights, observed.rewards,
        observed.log_normalizer_estimate, torch.exp(observed.rewards), .9, 3.)
    assert result == direct


def test_first_event_rule_needs_one_particle_and_keeps_failures():
    checkpoints = [
        {'num_particles': 4, 'corrections': {'0.99': {'event_particles': hits, 'smc': estimate}}}
        for hits, estimate in [(0, 0.), (1, .00222), (2, .4), (4, .00222)]]
    assert first_event_checkpoint(checkpoints, .99) is checkpoints[1]
    assert first_event_checkpoint(checkpoints[:1], .99) is None
    assert first_event_checkpoint(checkpoints, .99, minimum_event_particles=2) is checkpoints[2]
    with pytest.raises(ValueError):
        first_event_checkpoint(checkpoints, .99, minimum_event_particles=0)


def test_measured_MC_uses_each_saved_checkpoint_time_and_distinct_streams(tmp_path, monkeypatch):
    from diffusion_toxicity_smc_experiment.accuracy_budget import checkpoint_comparisons
    import diffusion_toxicity_smc_experiment.camera_ready_gpu as gpu
    calls=[]

    def crude(context,seed,thresholds,*,count,target_seconds,policy):
        calls.append((seed,count,target_seconds))
        return dict(seed=seed,n=count,runtime_seconds=target_seconds,
            estimates={'0.99':0.},scores=[.1]*count,policy=policy)

    monkeypatch.setattr(gpu,'crude',crude)
    correction={'event_particles':2,'smc':.002,'snis':.003,
                'reverse_mass':1.,'reverse_ess_fraction':.5,'reverse_max_weight':.3}
    snapshots=[dict(num_particles=4,level=k,sweep=j,**{'lambda':float(k)},
        runtime_seconds_cumulative=t,native_transitions_cumulative=100+10*i,
        log_z_hat=0.,smc_ess_fraction=.8,resampled_at_level=False,
        corrections={'0.99':dict(correction,event_particles=h)})
        for i,(k,j,t,h) in enumerate([(1,1,3.,2),(1,4,7.,3),(2,4,19.,3)])]
    population=dict(population_checkpoints=snapshots,
        first_event={'0.99':{'detected':True,'first_checkpoint_index':0}})
    study=dict(thresholds=[.99],reference={'samples':10000,'discoveries':[22]},
        crude_stream=19,minimum_event_particles=1,timing_tolerance_fraction=.05)
    config={'_exploration':{'id':'test','moves_per_level':4,'schedule':[1.,2.]}}
    terminal=dict(n=38,runtime_seconds=19.,estimates={'0.99':0.})
    rows,stopped=checkpoint_comparisons(None,config,study,29,population,terminal,
        root=tmp_path,seconds_per_path=.5,timing_device={'gpu':'test'})
    assert [(count,t) for _,count,t in calls]==[(6,3.),(14,7.)]
    assert len({seed for seed,_,_ in calls})==2
    assert len(rows)==3 and all(r['MC_comparison_measured'] for r in rows)
    assert len(stopped)==1 and stopped[0]['sweep']==1 and stopped[0]['first_event_detected']
    # Resume reads saved baselines without generating fresh MC observations.
    checkpoint_comparisons(None,config,study,29,population,terminal,
        root=tmp_path,seconds_per_path=.5,timing_device={'gpu':'test'})
    assert len(calls)==2


def test_prepared_same_run_study_has_no_independent_endpoint_chains():
    from pathlib import Path
    from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs,assigned_runs
    path=Path(__file__).resolve().parents[1]/'configs/mdlm_owt_same_run_ladder.yaml'
    study,configs=study_configs(path)
    assert [c['smc']['num_particles'] for c in configs]==[128]
    assert study['record_population_checkpoints'] and study['same_run_ladder']
    assert study['minimum_event_particles']==1 and 'occupancy_fraction' not in study
    assert study['thresholds']==[.99,.999] and study['reference']['discoveries']==[444,26]
    assert all(c['_exploration']['schedule']==[.5*k for k in range(1,17)] for c in configs)
    tasks=assigned_runs(study,configs,0,1)
    assert len(tasks)==12 and len({seed for _,seed in tasks})==12
    assert all(len(assigned_runs(study,configs,rank,3))==4 for rank in range(3))


def test_real_recorder_keeps_every_MH_population_and_matches_terminal_estimates(monkeypatch):
    from test_exploratory import NativeBinaryPaths,component
    from diffusion_toxicity_smc_experiment import exploratory_population as recorder
    class Functions:
        def reward(self,x):
            scores=[.99 if int(v)==1 else .0001 for v in x[:,-1]]
            self.last_event_rows=[{'toxic_probability':q} for q in scores]
            return torch.tensor([math.log(q) for q in scores],dtype=torch.float64)
        def event(self,x):
            self.reward(x)
            return x[:,-1]==1
    monkeypatch.setattr(recorder,'component_config',lambda *_:component())
    context=SimpleNamespace(functions=Functions(),sampler=NativeBinaryPaths(),device=torch.device('cpu'),synchronize=lambda:None)
    config={'_exploration':{'schedule':[1.,2.,3.]}}
    study={'thresholds':[.99,.999],'record_population_checkpoints':True,'minimum_event_particles':1}
    output=recorder.population(context,config,study,123)
    snapshots=output['population_checkpoints']
    assert len(snapshots)==13
    assert snapshots[-1]['corrections']==output['corrections']
    assert snapshots[-1]['runtime_seconds_cumulative']==output['summary']['runtime_seconds']
    assert all(a['runtime_seconds_cumulative']<=b['runtime_seconds_cumulative'] for a,b in zip(snapshots,snapshots[1:]))
    assert output['first_event']['0.99']['detected']
    assert not output['first_event']['0.999']['detected']
    assert output['first_event']['0.999']['first_checkpoint_index'] is None
    assert any(cp['score_source'].startswith('original classifier') for cp in snapshots)
