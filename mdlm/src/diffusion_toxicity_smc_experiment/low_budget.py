"""Small-budget camera-ready pilot; exact existing RollBack kernel, no new model.

The same population estimates both thresholds. No aggregation across tilts.
Native DDPM steps AND stochastic terminal-removal transitions count in the
primary budget. Rejected proposals cost their entire proposed suffix.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .configuration import decoder_id, evaluator_id, fingerprint, load_yaml
from .pipeline import estimates as estimates
from .storage import atomic_write_json, initialize_manifest


def study_configs(study_path: Path):
    study = load_yaml(study_path)
    base = load_yaml(study_path.parent / study['base_config'])
    if (decoder_id(base), evaluator_id(base)) != (
        study['reference']['decoder_id'], study['reference']['evaluator_id']
    ):
        raise ValueError('Frozen reference decoder/evaluator do not match')
    if study['thresholds'] != [0.9, 0.99]:
        raise ValueError('This study fixes the two requested thresholds')
    configs = []
    for index, n in enumerate(study['particle_counts']):
        config = copy.deepcopy(base)
        config['smc']['tilted_sampler']['mcmc_steps_per_level'] = int(study['mcmc_steps_per_level'])
        config['smc']['tilted_sampler']['num_tempering_levels'] = int(study['num_tempering_levels'])
        config['smc'].update(num_particles=int(n),
            independent_runs=int(study['independent_runs']),
            num_independent_runs=int(study['independent_runs']),
            tilted_lambdas=[float(study['lambda_final'])],
            tilted_component_weights_a=[1.0], diagnostic_lambdas=[],
            base_seed=int(study['base_seed']) + 10000 * index)
        configs.append(config)
    return study, configs


def plan(study_path: Path):
    study, configs = study_configs(study_path)
    rows = []
    for config in configs:
        s = config['smc']; m = s['tilted_sampler']
        sweeps = m['num_tempering_levels'] * m['mcmc_steps_per_level']
        mean_d = sum(d * p for d, p in zip(m['rollback_steps'],m['rollback_probabilities']))
        n = s['num_particles']; t = config['decoder']['denoising_steps']
        expected = n * (t + 1 + sweeps * (mean_d + 1))
        equivalent = expected / (t + 1)
        rows.append({'N': n, 'lambda_final': s['tilted_lambdas'][0],
            'levels': m['num_tempering_levels'], 'moves_per_level': m['mcmc_steps_per_level'],
            'rollback_depths': m['rollback_steps'],
            'expected_native_transitions': expected,
            'expected_crude_paths_before_floor': equivalent,
            'note': 'Actual per-run budget determines floor(cost/33), not this expectation',
            'expected_crude_events': {str(a): equivalent*k/study['reference']['samples']
                for a,k in zip(study['thresholds'],study['reference']['discoveries'])},
            'runs': study['independent_runs'], 'sampler_base_seed': s['base_seed']})
    return {'study': study, 'configurations': rows}


def transition_budget(summary):
    n = int(summary['num_particles'])
    ddpm = int(summary['total_decoder_particle_steps'])
    terminal = n + int(summary['mcmc_proposals'])
    total = ddpm + terminal
    return {'ddpm_particle_steps': ddpm,'terminal_categorical_transitions': terminal,
        'native_transitions': total,'crude_n': total//33,
        'unused_transitions': total%33}


def crude_seed(sampler_seed, stream=924071):
    return int(np.random.SeedSequence([int(sampler_seed),int(stream)]).generate_state(1,dtype=np.uint64)[0] % np.uint64(2**63-1))


def run(study_path: Path, output_root: Path):
    from .camera_ready_gpu import run as run_paired_gpu
    return run_paired_gpu(study_path,output_root)


def report(study_path: Path, output_root: Path, render=True):
    study,_=study_configs(study_path)
    root=output_root/study['output_subdirectory']
    files=sorted((root/'runs').glob('N*_seed*.json'))
    if not files: raise ValueError('No completed pilot runs; report will not invent results')
    rows=[r for path in files for r in json.loads(path.read_text(encoding='utf-8'))['rows']]
    keys=[(r.get('comparison','matched_transitions'),r['N'],r['seed'],r['threshold']) for r in rows]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate run')
    def write_csv(name,records):
        with (root/name).open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write_csv('runs.csv',rows)
    records=[]
    for policy,n,a in sorted({(r.get('comparison','matched_transitions'),r['N'],r['threshold']) for r in rows}):
        g=[r for r in rows if r['N']==n and r['threshold']==a and r.get('comparison','matched_transitions')==policy]
        for method in ['smc','snis','crude']:
            values=np.array([r[method] for r in g])
            ratio=values/g[0]['p_reference']; metric=np.log10(1+abs(ratio-1))
            records.append({'comparison':policy,'N':n,'threshold':a,'method':method,'runs':len(g),
                'p_reference':g[0]['p_reference'],'zero_percent':100*float((values==0).mean()),
                'median_log_relative_error':float(np.median(metric)),
                'mean_native_transitions':float(np.mean([r['native_transitions'] for r in g])),
                'mean_crude_n':float(np.mean([r['crude_n'] for r in g])),
                **{f'ratio_q{int(100*q):02d}':float(np.quantile(ratio,q)) for q in [.1,.25,.5,.75,.9,.99]}})
    write_csv('summary.csv',records)
    if render:
        from .camera_ready_report import build_report
        build_report(study,root,rows,records)
    print(f'Pilot report: {len(files)} completed populations. Eight runs are not sufficient for final performance conclusions.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','run','report'])
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=Path('mdlm/artifacts/current'))
    args=parser.parse_args()
    if args.stage=='plan': print(json.dumps(plan(args.study),indent=2))
    elif args.stage=='run': run(args.study,args.output_root)
    else: report(args.study,args.output_root)

if __name__=='__main__': main()
