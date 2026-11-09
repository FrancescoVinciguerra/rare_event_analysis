"""Empirical relative dispersion from saved independent runs; no simulation.

RV = sample_variance(estimator) / p_reference**2. This is not variance
divided by the squared empirical mean. RSD = sqrt(RV) = sample_sd / p_reference.
Every zero estimate is included. Select --statistic std for the RSD figure.
The main figure focuses on non-self-normalized SMC. SNIS is retained in the
machine-readable table, without contributing a curve to the figure.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyze(old_root, new_root, output, statistic='variance', *,
            checkpoint_file='checkpoint_runs.csv', suffix='', reference_note=None):
    if statistic not in {'variance','std'}:raise ValueError('Invalid dispersion statistic')
    is_std=statistic=='std'
    prefix='relative_std' if is_std else 'relative_variance'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources={};tables={}
    for phi,root in [('log_probability',Path(old_root)),('logit',Path(new_root))]:
        path=root/checkpoint_file
        try:
            source_path=Path(os.path.relpath(path,output)).as_posix()
        except ValueError:
            source_path=path.resolve().as_posix()
        sources[phi]=dict(file=source_path,SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        with path.open(encoding='utf-8',newline='') as f:
            tables[phi]=[r for r in csv.DictReader(f) if int(r['level'])>0 and int(r['sweep'])==4]
        keys=[(r['seed'],r['lambda'],r['threshold']) for r in tables[phi]]
        if len(keys)!=len(set(keys)):raise ValueError('Repeated independent run')
    events=sorted({float(r['threshold']) for r in tables['logit']})
    levels=sorted({float(r['lambda']) for r in tables['logit']})
    records=[]
    for phi,table in tables.items():
        for event in events:
            for tilt in levels:
                group=[r for r in table if float(r['threshold'])==event and float(r['lambda'])==tilt]
                if len(group)!=12 or {int(r['N']) for r in group}!={128}:
                    raise ValueError('Expected twelve complete N=128 runs at each fixed level')
                refs={float(r['p_reference']) for r in group}
                if len(refs)!=1:raise ValueError('Mixed probability references')
                ref=refs.pop()
                for method in ['smc','snis','crude']:
                    estimates=[float(r[method]) for r in group]
                    if not all(math.isfinite(v) and v>=0 for v in estimates):
                        raise ValueError('Invalid probability estimate')
                    variance=statistics.variance(estimates)
                    mean=statistics.mean(estimates)
                    mc_rv=statistics.mean((1-ref)/(ref*int(r['crude_n'])) for r in group)
                    records.append(dict(phi=phi,threshold=event,**{'lambda':tilt},method=method,
                        independent_runs=len(group),N=128,p_reference=ref,
                        mean_estimate=mean,mean_over_reference=mean/ref,
                        empirical_variance=variance,empirical_relative_variance=variance/ref**2,
                        empirical_standard_deviation=math.sqrt(variance),
                        empirical_relative_standard_deviation=math.sqrt(variance)/ref,
                        zero_estimates=sum(v==0 for v in estimates),
                        MC_relative_variance_using_reference=mc_rv if method=='crude' else None,
                        MC_relative_standard_deviation_using_reference=math.sqrt(mc_rv) if method=='crude' else None,
                        mean_MC_trajectories=statistics.mean(int(r['crude_n']) for r in group),
                        mean_SMC_seconds=statistics.mean(float(r['smc_runtime_seconds_cumulative']) for r in group)))
    csv_path=output/f'{prefix}_comparison{suffix}.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    fig,axes=plt.subplots(1,len(events),figsize=(7.5*len(events),6.8),squeeze=False)
    field='empirical_relative_standard_deviation' if is_std else 'empirical_relative_variance'
    theory_field='MC_relative_standard_deviation_using_reference' if is_std else 'MC_relative_variance_using_reference'
    labels=[('logit','smc',field,'SMC logit: empirica','#2878b5','o','-'),
            ('log_probability','smc',field,'SMC log(q): empirica','#6ea9d2','^','--'),
            ('logit','crude',field,'Crude MC: empirica','#888888','s','--'),
            ('logit','crude',theory_field,'Crude MC: teorica con riferimento','#222222',None,':')]
    audit=[]
    for axis,event in zip(axes[0],events):
        for phi,method,field,label,color,marker,style in labels:
            subset=sorted((r for r in records if r['phi']==phi and r['method']==method and r['threshold']==event),key=lambda r:r['lambda'])
            values=[r[field] for r in subset]
            axis.plot(levels,values,label=label,color=color,marker=marker,ls=style,lw=1.7,ms=4)
            audit.append(dict(threshold=event,phi=phi,method=method,field=field,
                              plotted_values=values,zeros_preserved=sum(v==0 for v in values)))
        ref=next(r['p_reference'] for r in records if r['threshold']==event)
        axis.axhline(1,color='#ad3434',lw=1,ls='--',alpha=.75)
        axis.set_title(f'Soglia q ≥ {event:g} · riferimento {ref:g}',fontsize=12)
        axis.set_xlabel(r'Tilt corrente $\lambda$')
        axis.set_ylabel(r'Deviazione standard relativa: $s(\widehat p_A)/\widehat p_{\rm ref}$' if is_std
                        else r'Varianza relativa: $s^2(\widehat p_A)/\widehat p_{\rm ref}^{\,2}$')
        if not is_std:axis.set_yscale('symlog',linthresh=.1,linscale=1)
        maximum=max(max(r['plotted_values']) for r in audit if r['threshold']==event)
        axis.set_ylim(-.008,maximum*1.35)
        ticks=[0,.25,.5,.75,1,1.5,2,3,4,5] if is_std else [0,.05,.1,.2,.5,1,2,5,10,20]
        ticks=[t for t in ticks if t<=maximum*1.35]
        axis.set_yticks(ticks,[f'{100*t:g}%' if is_std else f'{t:g}' for t in ticks])
        axis.set_xticks(levels,[f'{x:g}' for x in levels],fontsize=8)
        axis.grid(axis='y',alpha=.2)
        if event==.999:
            axis.text(.97,.96,('MC: RSD empirica = 0 fino a λ=2' if is_std else 'MC: RV empirica = 0 fino a λ=2')
                      +'\nperché tutte le 12 stime sono zero.',
                      transform=axis.transAxes,va='top',ha='right',fontsize=9,
                      bbox=dict(facecolor='white',alpha=.85,edgecolor='#dddddd'))
    fig.suptitle('SMC non normalizzato · '+('deviazione standard relativa' if is_std else 'varianza relativa')
                 +' tra run indipendenti',fontsize=15,y=.985)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.10),fontsize=9)
    footer=('N=128, 12 run per livello; tutti gli zeri inclusi. Popolazioni a tilt diversi della stessa run sono correlate.\n'
            'Deviazione standard della singola stima tra run: s/p_ref, senza divisione per √12. Linea rossa: 100%.\n'
            'MC teorica: sqrt(media[(1−p_ref)/(n_i p_ref)]), con n_i abbinato al tempo SMC tramite calibrazione.\n'
            + (reference_note or 'Riferimenti stimati da 200.000 traiettorie: 444 eventi e 26 eventi.')
            if is_std else
            'N=128, 12 run per livello; tutti gli zeri inclusi. Popolazioni a tilt diversi della stessa run sono correlate.\n'
            'Scala verticale symlog: lineare da 0 a 0,1, logaritmica oltre. Linea rossa: RV=1.\n'
            'MC teorica: media di (1−p_ref)/(n_i p_ref), con n_i abbinato al tempo SMC tramite calibrazione.\n'
            + (reference_note or 'Riferimenti stimati da 200.000 traiettorie: 444 eventi e 26 eventi.'))
    fig.text(.5,.015,footer,ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.065,right=.985,top=.89,bottom=.27,wspace=.24)
    figure_dir = output / 'plots'
    figure_dir.mkdir(exist_ok=True)
    image_path=figure_dir/f'{prefix}_SMC{suffix}.png'
    fig.savefig(image_path,dpi=300);plt.close(fig)
    (output/f'{prefix}_audit{suffix}.json').write_text(json.dumps(dict(
        definition='Sample standard deviation across 12 independent complete runs, divided by the finite probability reference' if is_std
                   else 'Sample variance across 12 independent complete runs, divided by the squared finite probability reference',
        MC_theory='sqrt(mean_i[(1-p_reference)/(n_i*p_reference)])' if is_std else 'mean_i[(1-p_reference)/(n_i*p_reference)]',
        MC_cost='calibrated wall-clock equivalent; no new MC generation',
        interpretation='Empirical dispersion can miss rare large importance weights with only 12 runs. See mean_over_reference.',
        relative_source_paths_from='directory containing this audit',
        sources=sources,curves=audit,new_simulations=False),indent=2)+'\n',encoding='utf-8')
    return dict(figure=str(image_path),table=str(csv_path),rows=len(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--statistic',choices=['variance','std'],default='variance')
    args=parser.parse_args()
    print(json.dumps(analyze(args.old,args.new,args.output,args.statistic),indent=2))
