import importlib.util
import json
from pathlib import Path
import sys

import pytest


directory=Path(__file__).resolve().parents[1]/'analysis'
sys.path.insert(0,str(directory))
try:
    spec=importlib.util.spec_from_file_location('assumed_clock_analysis',directory/'compare_levels_assumed_clock.py')
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def test_lambda_one_gets_one_third_of_final_time_and_MC_samples():
    seconds,count=analysis.prefix_budget(300.,2,6,.2,1500)
    assert seconds==100.
    assert count==500


def test_missing_MC_observations_are_not_fabricated_or_upsampled():
    with pytest.raises(ValueError,match='do not cover'):
        analysis.prefix_budget(300.,6,6,.2,1499)


def test_prefix_uses_saved_observations_and_retains_early_zero_estimates(tmp_path,monkeypatch):
    source=tmp_path/'source';clock=tmp_path/'clock';name='A_N2_seed17.json'
    original=source/'workers/worker_0/populations'/name;original.parent.mkdir(parents=True)
    original.write_text(json.dumps({'summary':{'seed':17,'runtime_seconds':12.},'level_diagnostics':[{}]*6}))
    baseline=clock/'workers/worker_0/baselines'/name;baseline.parent.mkdir(parents=True)
    scores=[.1]*4+[.95]+[.1]*7
    baseline.write_text(json.dumps({'seed':123,'n':12,'scores':scores,'target_seconds':12.,'estimates':{'0.9':1/12}}))
    (baseline.parent.parent/'calibration.json').write_text(json.dumps({'seconds_per_path':1.,'excluded_from_final_estimates':True}))
    rows=[{'configuration':'A','seed':17,'level':k,'threshold':.9} for k in range(1,7)]
    monkeypatch.setattr(analysis,'recover',lambda _:({'reference':{}},rows,{'population_policy':'actual continuing run'}))
    _,actual,audit=analysis.compare(source,clock)
    assert [r['crude_n'] for r in actual]==[2,4,6,8,10,12]
    assert [r['crude'] for r in actual]==pytest.approx([0.,0.,1/6,1/8,1/10,1/12])
    assert all(not r['intermediate_time_measured'] for r in actual)
    assert audit['terminal_MC_reproduction_checks']==1
    assert not audit['new_simulations'] and not audit['selection_depends_on_MC_outcomes']
