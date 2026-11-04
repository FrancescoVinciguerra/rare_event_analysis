import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'analysis/same_run_timing.py'
spec=importlib.util.spec_from_file_location('same_run_timing',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_measured_increments_keep_initial_cost_and_consistent_MC_prefix_rounding():
    records=[]
    for level,sweep,elapsed,n in [(0,0,10.,33),(1,1,10.5,35),(1,4,12.,40),(2,4,15.,50)]:
        for threshold in [.99,.999]:
            records.append(dict(seed=1,level=level,sweep=sweep,threshold=threshold,
                smc_runtime_seconds_cumulative=elapsed,calibration_seconds_per_path=.3,crude_n=n))
    rows=module.timing_rows(records)
    assert len(rows)==4
    assert sum(r['smc_since_previous_population_seconds'] for r in rows)==15.
    assert sum(r['crude_new_prefix_samples'] for r in rows)==50
    assert rows[1]['smc_since_previous_population_seconds']==.5
    assert rows[1]['crude_equivalent_in_population_update']==1
    assert rows[1]['crude_new_prefix_samples']==2
    assert rows[2]['smc_level_update_seconds']==2.
    assert rows[2]['crude_new_prefix_samples_in_level']==7
    assert rows[3]['smc_level_update_seconds']==3.
    assert rows[3]['crude_new_prefix_samples_in_level']==10


def test_duplicate_event_rows_cannot_hide_inconsistent_population_times():
    rows=[dict(seed=1,level=0,sweep=0,smc_runtime_seconds_cumulative=t,
               calibration_seconds_per_path=.3,crude_n=33) for t in [10.,11.]]
    with pytest.raises(ValueError,match='disagree'):module.timing_rows(rows)
