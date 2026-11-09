"""Offline comparison under the user's equal-time-per-level approximation.

No model or Monte Carlo simulation is run. Actual saved MC observations are
reused in their original order; the prefix length depends only on SMC's final
measured duration and an independent GPU calibration, never on MC outcomes.
Intermediate SMC times are ASSUMED, not measured. Initialization is included
in the measured final duration but is not separately budgeted in this model.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path

import numpy as np

from analyze_saved_levels import recover, summarize, figures, write_csv


METHODS=[('smc','SMC non normalizzato','#2878b5'),
         ('snis','SMC + SNIS','#ed8b23'),('crude','Crude MC','#929292')]


def prefix_budget(final_seconds, level, total_levels, seconds_per_path, available):
    """Budget is fixed without inspecting any MC score."""
    if not (final_seconds>0 and seconds_per_path>0 and 1<=level<=total_levels):
        raise ValueError('Invalid time model')
    assumed=final_seconds*level/total_levels
    count=math.floor(assumed/seconds_per_path)
    if not 1<=count<=available:
        raise ValueError('Saved MC observations do not cover the assumed budget')
    return assumed,count


def compare(source,clock):
    study,rows,reconstruction=recover(source)
    baselines={};provenance={}
    for file in sorted((clock/'workers').glob('worker_*/baselines/*.json')):
        baseline=json.loads(file.read_text(encoding='utf-8'))
        original=source/'workers'/file.parent.parent.name/'populations'/file.name
        population=json.loads(original.read_text(encoding='utf-8'))
        calibration=file.parent.parent/'calibration.json'
        cal=json.loads(calibration.read_text(encoding='utf-8'))
        case=file.name.split('_',1)[0];seed=int(population['summary']['seed'])
        key=(case,seed)
        if key in baselines:raise ValueError('Duplicate independent run')
        if baseline['n']!=len(baseline['scores']):raise ValueError('Incomplete saved MC scores')
        if not cal['excluded_from_final_estimates']:raise ValueError('Calibration used final MC outcomes')
        measured=float(population['summary']['runtime_seconds'])
        if not math.isclose(measured,baseline['target_seconds'],rel_tol=1e-12):
            raise ValueError('MC baseline belongs to another SMC timing target')
        baselines[key]=(baseline,cal,measured,len(population['level_diagnostics']))
        for path in [file,calibration]:
            provenance[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    terminal_checks=0
    for r in rows:
        b,cal,measured,total_levels=baselines[(r['configuration'],r['seed'])]
        assumed,count=prefix_budget(measured,r['level'],total_levels,cal['seconds_per_path'],b['n'])
        hits=sum(q>=r['threshold'] for q in b['scores'][:count])
        r.update(crude=hits/count,crude_event_particles=hits,crude_n=count,crude_seed=b['seed'],
                 assumed_elapsed_seconds=assumed,assumed_time_fraction=r['level']/total_levels,
                 measured_final_SMC_seconds=measured,calibrated_seconds_per_MC_path=cal['seconds_per_path'],
                 crude_prefix_calibrated_seconds=count*cal['seconds_per_path'],
                 timing_basis='assumed_equal_level_time_through_origin',
                 intermediate_time_measured=False,initialization_separately_budgeted=False)
        if r['level']==total_levels:
            if count!=b['n'] or not math.isclose(r['crude'],b['estimates'][str(r['threshold'])],abs_tol=1e-14):
                raise ValueError('Final MC estimate does not reproduce the saved follow-up')
            terminal_checks+=1
    audit={'new_simulations':False,'new_GPU_jobs':False,'independent_runs':len(baselines),
        'rows':len(rows),'terminal_MC_reproduction_checks':terminal_checks,
        'time_model':'t_assumed(k) = (k/6) * measured_SMC_time_at_level_6',
        'time_model_measured':False,'initialization_separately_budgeted':False,
        'MC_prefix_selection':'First n saved independent native trajectories, n=floor(t_assumed / independent calibration seconds per path)',
        'selection_depends_on_MC_outcomes':False,
        'levels_within_run_are_correlated':True,'averaging_levels':False,
        'SMC_population_policy':reconstruction['population_policy'],
        'finite_reference':study['reference'],'source_SHA256':provenance,
        'original_reconstruction':reconstruction}
    return study,rows,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--clock-results',type=Path,required=True)
    args=p.parse_args();root=args.results
    study,rows,audit=compare(root,args.clock_results)
    summary=summarize(rows,methods=METHODS)
    for s in summary:
        group=[r for r in rows if (r['configuration'],r['threshold'],r['lambda'])==
               (s['configuration'],s['threshold'],s['lambda'])]
        s['mean_assumed_seconds']=float(np.mean([r['assumed_elapsed_seconds'] for r in group]))
        s['mean_crude_n']=float(np.mean([r['crude_n'] for r in group]))
        s['min_crude_n']=min(r['crude_n'] for r in group)
        s['max_crude_n']=max(r['crude_n'] for r in group)
        s['time_model']='assumed_equal_time_per_level_not_measured'
    write_csv(root/'level_assumed_clock_estimates.csv',rows)
    write_csv(root/'level_assumed_clock_summary.csv',summary)
    (root/'level_assumed_clock_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    images=figures(root,study,rows,methods=METHODS,filename_prefix='levels_assumed_clock',
        title_suffix='confronto con MC a tempo APPROSSIMATO',
        subtitle='Ipotesi: tempo fino al tilt corrente = (tilt / 3) × tempo finale misurato. MC: prefissi delle traiettorie già salvate.',
        caption_tail='Tempi intermedi STIMATI, non misurati; costo iniziale non separato. Popolazioni della run originale con resampling intermedio, senza copie terminali.')
    note=('Confronto esplorativo sotto l’ipotesi richiesta: ogni livello costa uguale e il tempo cumulativo è '
          'approssimato con tilt/3 del tempo finale misurato. Il costo iniziale è ripartito proporzionalmente, '
          'non pagato separatamente: a tilt bassi questa scelta può sottostimare il costo SMC. '
          'Il numero di traiettorie MC è scelto con una calibrazione indipendente e i punteggi sono i primi '
          'n campioni già salvati, senza nuove simulazioni e senza selezionare gli esiti. '
          'Le stime dei diversi livelli non vengono mediate. Le 8 run di ogni configurazione sono indipendenti; '
          'i livelli entro una run sono correlati. Le popolazioni includono i resampling intermedi realmente '
          'eseguiti: non sono copie con esclusione del resampling al rispettivo livello finale. '
          'La probabilità di riferimento è stimata da 200.000 traiettorie, non è nota esattamente.')
    body='<h1>Livelli intermedi contro MC: ipotesi di tempi uniformi</h1><p>'+escape(note)+'</p>'
    for image in images:body+=f'<p><a href="{image.relative_to(root).as_posix()}"><img src="{image.relative_to(root).as_posix()}" style="width:100%"></a></p>'
    body+='<p><a href="level_assumed_clock_estimates.csv">Singole run</a> · <a href="level_assumed_clock_summary.csv">Riassunto</a> · <a href="level_assumed_clock_audit.json">Provenienza e ipotesi</a></p>'
    (root/'LEVELS_ASSUMED_CLOCK_REPORT.html').write_text('<!doctype html><html lang="it"><meta charset="utf-8"><style>body{font:16px Arial;margin:24px}</style>'+body+'</html>',encoding='utf-8')
    (root/'LEVELS_ASSUMED_CLOCK_ANALYSIS.md').write_text('# Confronto sotto ipotesi di tempi uniformi\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps({'independent_runs':audit['independent_runs'],'rows':len(rows),'new_simulations':False,
        'terminal_MC_checks':audit['terminal_MC_reproduction_checks'],'figures':[str(x) for x in images]},indent=2))


if __name__=='__main__':main()
