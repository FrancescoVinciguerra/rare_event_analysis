import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'analysis/analyze_saved_levels.py'
spec=importlib.util.spec_from_file_location('saved_level_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

@pytest.mark.parametrize('lam',[.5,1,1.5,2,2.5,3])
def test_current_tilt_correction_recovers_known_original_probabilities(lam):
    p=np.array([.8,.15,.05]);q=np.array([.1,.95,.995]);z=float((p*q**lam).sum());w=p*q**lam/z
    for threshold,expected in [(.9,.2),(.99,.05),(.9999,0.)]:
        result=analysis.corrections(np.log(w),np.log(q),math.log(z),lam,threshold)
        assert result['smc']==pytest.approx(expected,abs=1e-13)
        assert result['snis']==pytest.approx(expected,abs=1e-13)
        assert result['reverse_mass']==pytest.approx(1.,abs=1e-13)

def test_ambiguous_log_threshold_is_not_silently_classified():
    with pytest.raises(ValueError,match='ambiguous'):
        analysis.corrections(np.log([.5,.5]),np.log([.9,.95]),0.,.5,.9)

@pytest.mark.parametrize('case_id',['A','A0p999_B25_L80'])
def test_intermediate_population_comes_from_next_ledger_not_current_or_terminal(tmp_path,case_id):
    p=np.array([.8,.2]);q=np.array([.1,.95]);z=float((p*q**.5).sum());w=p*q**.5/z
    # A later mutation has a different score vector. The incoming level-2
    # ledger, not level-1's old scores or the terminal scores, gives Phi(tau_1).
    final_q=np.array([.8,.95]);final_w=np.array([.5,.5]);final_z=.2
    numerator=.5/.95;denominator=.5/.8+.5/.95
    data={'summary':{'num_particles':2,'seed':17,'log_normalizer_estimate':math.log(final_z),'runtime_seconds':1.},
      'particles':[{'reward':float(math.log(x)),'final_log_weight':math.log(.5)} for x in final_q],
      'corrections':{'0.9':{'smc':final_z*numerator,'snis':numerator/denominator,'reverse_mass':final_z*denominator,'event_particles':1}},
      'level_diagnostics':[
        {'eta':.5,'log_normalizer_estimate':math.log(z),'resampled':False,'distinct_root_ancestors':2,'rewards_before_mutation':np.log([.05,.2]).tolist()},
        {'eta':1.,'log_normalizer_estimate':math.log(final_z),'log_z_before_weight':math.log(z),'previous_normalized_log_weights':np.log(w).tolist(),
         'rewards_before_mutation':np.log(q).tolist(),'resampled':False,'distinct_root_ancestors':2}],
      'mutation_diagnostics':[{'level':k,'proposals':2,'rollback_particle_steps':2} for k in [1,2]]}
    worker=tmp_path/'workers/worker_0';(worker/'populations').mkdir(parents=True);(worker/'runs').mkdir()
    filename=f'{case_id}_N2_seed17.json'
    (worker/'populations'/filename).write_text(json.dumps(data));(worker/'runs'/filename).write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'study':{'configurations':[{'id':case_id,'N':2,'moves_per_level':1}],
        'thresholds':[.9],'reference':{'samples':100,'discoveries':[20]}}}))
    _,rows,audit=analysis.recover(tmp_path)
    assert rows[0]['lambda']==.5 and rows[0]['smc']==pytest.approx(.2) and rows[0]['snis']==pytest.approx(.2)
    assert rows[0]['runtime_seconds'] is None and rows[1]['runtime_seconds']==1.
    assert rows[0]['configuration']==case_id and rows[0]['lambda_final']==1.
    assert audit['populations']==1 and not audit['simulations_rerun']
