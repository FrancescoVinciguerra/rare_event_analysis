import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from diffusion_toxicity_smc_experiment import same_run_reuse as reuse


def test_clock_budget_and_disjoint_blocks_are_fixed_before_MC_outcomes():
    assert reuse.native_clock_count(10.99,.2)==54
    scores=np.arange(120.)
    a,start_a,size=reuse.fixed_mc_block(scores,seed_rank=2,total_seeds=12,required=4)
    b,start_b,_=reuse.fixed_mc_block(scores,seed_rank=3,total_seeds=12,required=7)
    assert a.tolist()==[20,21,22,23] and b.tolist()==[30,31,32,33,34,35,36]
    assert start_a==20 and start_b==30 and size==10
    # Missing SMC seeds do not change the block of any other planned seed.
    again,_,_=reuse.fixed_mc_block(scores,seed_rank=3,total_seeds=12,required=2)
    assert again.tolist()==b[:2].tolist()
    with pytest.raises(ValueError):reuse.fixed_mc_block(scores,seed_rank=3,total_seeds=12,required=11)


def test_report_reuses_MC_for_every_MH_population_and_retains_zero_estimates(tmp_path):
    from test_exploratory import NativeBinaryPaths
    from diffusion_toxicity_smc_experiment.exploratory_population import population
    study_path=Path(__file__).resolve().parents[1]/'configs/mdlm_owt_same_run_ladder.yaml'
    study,configs=reuse.study_configs(study_path)
    config=copy.deepcopy(configs[0]);config['decoder']['sequence_length']=3
    seed=2006000000
    class Functions:
        def reward(self,x):
            scores=[.99 if int(v)==1 else .0001 for v in x[:,-1]]
            self.last_event_rows=[{'toxic_probability':q} for q in scores]
            return torch.tensor([math.log(q) for q in scores],dtype=torch.float64)
        def event(self,x):self.reward(x);return x[:,-1]==1
    context=SimpleNamespace(functions=Functions(),sampler=NativeBinaryPaths(),device=torch.device('cpu'),synchronize=lambda:None)
    p=population(context,config,study,seed)
    device=dict(hostname='gpu007',SLURM_STEP_GPUS='0',CUDA_VISIBLE_DEVICES='0',SLURM_CPUS_PER_TASK='4')
    p['timing_device']=device
    root=tmp_path/study['output_subdirectory'];worker=root/'workers/worker_0'
    (worker/'populations').mkdir(parents=True)
    (worker/'populations'/f'L128_seed{seed}.json').write_text(json.dumps(p))
    (worker/'calibration_test.json').write_text(json.dumps(dict(seconds_per_path=100.,timing_device=device,
        excluded_from_estimates_and_measured_run_times=True)))
    pool=tmp_path/'completed_MC';(pool/'workers/worker_0/baselines').mkdir(parents=True)
    (pool/'manifest.json').write_text(json.dumps({'study':{'reference':study['reference']}}))
    (pool/'report_status.json').write_text(json.dumps(dict(completed_runs=1,expected_runs=1)))
    scores=[.0001]*240;scores[120]=.9995
    (pool/'workers/worker_0/baselines/fixed.json').write_text(json.dumps(dict(n=240,seed=987,scores=scores,native_transitions=33*240,sample_count_chosen_before_generation=True)))
    reuse.report(study_path,tmp_path,pool)
    rows=list(csv.DictReader((root/'checkpoint_runs.csv').open()))
    assert len(rows)==130 and all(r['MC_comparison_measured']=='False' for r in rows)
    assert all(r['MC_pool_start']=='120' and r['crude_n']=='1' and r['crude']=='1.0' for r in rows)
    assert all(float(r['smc'])==0 and r['eligible_population']=='False' for r in rows if float(r['threshold'])==.999)
    assert all(r['crude_runtime_seconds']=='' for r in rows)
    assert json.loads((root/'comparison_policy.json').read_text())['new_MC_samples_generated']==0
    assert json.loads((root/'report_status.json').read_text())['completed_runs']==1
    # The first-event table also retains the seed that never finds the event.
    first=list(csv.DictReader((root/'first_event_runs.csv').open()))
    assert len(first)==2 and any(r['first_event_detected']=='False' for r in first)
