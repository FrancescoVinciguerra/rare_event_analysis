"""Verify saved same-run estimators and reused-MC budgets without simulations."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

import numpy as np

EXPERIMENT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(EXPERIMENT/'src'))
from diffusion_toxicity_smc_experiment.accuracy_budget import study_configs,assigned_runs
from diffusion_toxicity_smc_experiment.exploratory import audit_population
from diffusion_toxicity_smc_experiment.low_budget import estimates
from diffusion_toxicity_smc_experiment.potentials import potential_from_config
from diffusion_toxicity_smc_experiment.same_run_reuse import calibration_for,mc_pool,fixed_mc_block,native_clock_count


def verify(results,study_path,pool_root):
    results=Path(results)
    study,configs=study_configs(study_path)
    tasks=sorted(assigned_runs(study,configs,0,1),key=lambda item:item[1])
    ranks={seed:rank for rank,(_,seed) in enumerate(tasks)}
    config_for={seed:c for c,seed in tasks}
    with (results/'checkpoint_runs.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    lookup={(int(r['seed']),int(r['level']),int(r['sweep']),float(r['threshold'])):r for r in rows}
    assert len(lookup)==len(rows),'Repeated checkpoint rows'
    pool,sources=mc_pool(Path(pool_root),study)
    policy=json.loads((results/'comparison_policy.json').read_text(encoding='utf-8'))
    canonical=lambda records:[dict(r,file=r['file'].replace('\\','/')) for r in records]
    assert canonical(sources)==canonical(policy['MC_sources']),'MC source hashes differ'
    assert policy['new_MC_samples_generated']==0
    blocks=json.loads((results/'saved_MC_blocks.json').read_text(encoding='utf-8'))
    seen=[];checked=0;max_relative=0.

    def close(a,b):
        nonlocal max_relative
        a=float(a);b=float(b)
        assert math.isfinite(a) and math.isfinite(b)
        if a==b:return
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=0.),(a,b)
        max_relative=max(max_relative,abs(a-b)/max(abs(a),abs(b)))

    for file in sorted(results.glob('workers/worker_*/populations/*.json')):
        raw=file.read_bytes();p=json.loads(raw);seed=p['summary']['seed']
        config=config_for[seed];seen.append(seed)
        potential=potential_from_config(config)
        assert p.get('phi',{}).get('name','log_probability')==potential.name
        audit_population(p,config)
        checkpoints=p['population_checkpoints']
        assert len(checkpoints)==1+len(config['_exploration']['schedule'])*config['_exploration']['moves_per_level']
        source=next(s for s in policy['SMC_sources'] if s['seed']==seed)
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
        _,cal=calibration_for(results,p);cost=cal['seconds_per_path']
        maximum=max(native_clock_count(cp['runtime_seconds_cumulative'],cost) for cp in checkpoints)
        block,start,size=fixed_mc_block(pool,seed_rank=ranks[seed],total_seeds=len(tasks),required=maximum)
        saved=blocks[str(seed)]
        assert start==saved['pool_start'] and size==saved['fixed_block_size']
        assert np.array_equal(block,np.asarray(saved['scores']))
        for cp in checkpoints:
            assert cp.get('phi','log_probability')==potential.name
            assert np.allclose([potential.probability(v) for v in cp['rewards']],cp['scores'],rtol=0,atol=2*np.finfo(np.float32).eps)
            count=native_clock_count(cp['runtime_seconds_cumulative'],cost)
            prefix=block[:count]
            for threshold in study['thresholds']:
                value=estimates(cp['normalized_log_weights'],cp['rewards'],cp['log_z_hat'],cp['scores'],threshold,cp['lambda'])
                row=lookup[(seed,cp['level'],cp['sweep'],threshold)]
                for key,actual in value.items():
                    close(actual,cp['corrections'][str(threshold)][key])
                    close(actual,row[key])
                hits=int((prefix>=threshold).sum())
                assert int(row['crude_n'])==count and int(row['crude_event_particles'])==hits
                close(hits/count,row['crude'])
                assert row['MC_comparison_measured']=='False'
                assert row['crude_runtime_seconds']==''
                close(cp['lambda'],row['lambda'])
                checked+=1
    assert len(seen)==len(set(seen))==len(tasks),'Missing independent seeds'
    assert set(seen)==set(config_for)
    assert checked==len(rows)
    output=dict(status='COMPLETE_VALIDATED',SMC_runs=len(seen),checkpoint_event_estimator_pairs=checked,
        independent_MC_source_files=len(sources),available_MC_pool_samples=len(pool),
        max_relative_recomputation_difference=max_relative,relative_tolerance=1e-12,
        normalizer_ledger_verified=True,no_final_resampling_verified=True,
        rejected_proposal_cost_verified=True,MC_blocks_and_all_prefix_estimates_verified=True,
        new_simulations_run=False)
    (results/'delivery_validation.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--mc-pool',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.results,a.study,a.mc_pool),indent=2))
