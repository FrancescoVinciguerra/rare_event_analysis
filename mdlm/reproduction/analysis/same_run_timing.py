"""Extract measured population-update times and calibrated MC-equivalent counts."""
from __future__ import annotations
import argparse
import csv
from html import escape
import math
from pathlib import Path
import statistics


def timing_rows(records):
    unique={}
    for r in records:
        key=(int(r['seed']),int(r['level']),int(r['sweep']))
        value=(float(r['smc_runtime_seconds_cumulative']),float(r['calibration_seconds_per_path']),int(r['crude_n']))
        if key in unique and unique[key]!=value:raise ValueError('Event rows disagree on the same population cost')
        unique[key]=value
    result=[]
    for seed in sorted({key[0] for key in unique}):
        previous_time=0.;previous_n=0;previous_level_time=None;previous_level_n=None
        for key in sorted(k for k in unique if k[0]==seed):
            _,level,sweep=key;elapsed,cost,n=unique[key]
            if elapsed<previous_time or n<previous_n or cost<=0:raise ValueError('Invalid cumulative cost')
            delta=elapsed-previous_time
            row=dict(seed=seed,level=level,sweep=sweep,**{'lambda':.5*level},
                smc_cumulative_seconds=elapsed,smc_since_previous_population_seconds=delta,
                seconds_per_MC_trajectory_calibrated=cost,crude_n_cumulative=n,
                crude_equivalent_in_population_update=math.floor(delta/cost),
                crude_new_prefix_samples=n-previous_n,
                smc_level_update_seconds=None,crude_equivalent_in_level_update=None,
                crude_new_prefix_samples_in_level=None,GPUs_per_estimate=1,CPUs_per_estimate=4)
            if level==0:
                previous_level_time=elapsed;previous_level_n=n
            elif sweep==4:
                if previous_level_time is None:raise ValueError('Missing initial population cost')
                row['smc_level_update_seconds']=elapsed-previous_level_time
                row['crude_equivalent_in_level_update']=math.floor((elapsed-previous_level_time)/cost)
                row['crude_new_prefix_samples_in_level']=n-previous_level_n
                previous_level_time=elapsed;previous_level_n=n
            result.append(row);previous_time=elapsed;previous_n=n
    return result


def build(root):
    records=list(csv.DictReader((root/'checkpoint_runs.csv').open(encoding='utf-8')))
    rows=timing_rows(records)
    path=root/'timing_steps.csv'
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=root/'REPORT.html'
    if report.exists():
        content=report.read_text(encoding='utf-8').split('<!-- same-run-timings -->')[0]
        content+='<!-- same-run-timings --><h2>Tempi delle popolazioni e budget MC equivalente</h2>'
        content+='<p>Tempi SMC misurati con sincronizzazione GPU; costo MC stimato dalla calibrazione. '
        content+='Una transizione fra livelli SMC include pesatura, eventuale resampling e quattro MH. '
        content+='Il confronto della stima completa usa il tempo cumulativo, inclusa la generazione iniziale. '
        content+='I nuovi campioni di ciascun prefisso MC sono differenze di numerosità cumulative; '
        content+='possono differire di uno dal floor del solo tempo incrementale.</p>'
        content+='<p><a href="timing_steps.csv">Tempi di ogni popolazione, per seed</a></p>'
        content+='<table><tr><th>Tilt</th><th>Seed</th><th>Secondi/livello: media [min,max]</th><th>MC aggiuntivi: media</th><th>MC cumulativi: media</th></tr>'
        for level in sorted({r['level'] for r in rows if r['sweep']==4}):
            group=[r for r in rows if r['level']==level and r['sweep']==4]
            times=[r['smc_level_update_seconds'] for r in group]
            cells=[f'{.5*level:g}',len(group),f'{statistics.mean(times):.2f} [{min(times):.2f}, {max(times):.2f}]',
                f"{statistics.mean(r['crude_new_prefix_samples_in_level'] for r in group):.1f}",
                f"{statistics.mean(r['crude_n_cumulative'] for r in group):.1f}"]
            content+='<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in cells)+'</tr>'
        content+='</table>';report.write_text(content,encoding='utf-8')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();print(build(args.results))
