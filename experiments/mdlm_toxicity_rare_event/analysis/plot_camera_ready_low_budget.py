"""Offline toy-style boxplots from real per-run CSV; never runs a sampler.

Run after downloading runs.csv. Dependencies: numpy and matplotlib only.
No epsilon for zeros, no clipping/winsorization, no averaging before metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS=[('smc','SMC non normalizzato','#2878b5'),
         ('snis','SMC + SNIS','#ed8b23'),('crude','Crude MC','#8d959f')]
REFERENCE={.9:(.00758,.00720922849129587,.007969687256908654),
           .99:(.00222,.0020230756570391337,.0024360459894034315)}
POLICIES={'matched_transitions':'pari transizioni','matched_wall_clock':'pari wall clock'}


def metric(values,p_reference,kind):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('All estimates must be finite and nonnegative; none may be filtered')
    ratio=values/p_reference
    if kind=='ratio': return ratio
    if kind=='log_error': return np.log10(1+np.abs(ratio-1))
    raise ValueError(kind)


def statistics(values):
    values=np.asarray(values,dtype=float)
    if not values.size: raise ValueError('No observations')
    if not np.isfinite(values).all(): raise ValueError('Nonfinite observations')
    return boxplot_stats(values,whis=1.5)[0]


def read_runs(path):
    with Path(path).open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows: raise ValueError('No experimental results available')
    keys=set()
    for row in rows:
        row['N']=int(row['N']); row['seed']=int(row['seed'])
        row['threshold']=float(row['threshold']); row['p_reference']=float(row['p_reference'])
        row['comparison']=row.get('comparison') or 'matched_transitions'
        if row['comparison'] not in POLICIES: raise ValueError('Unknown matching policy')
        if row['threshold'] not in REFERENCE: raise ValueError('Unexpected event')
        if not math.isclose(row['p_reference'],REFERENCE[row['threshold']][0],rel_tol=1e-12):
            raise ValueError('Reference differs from the frozen 200k reference')
        if 'lambda_final' in row and float(row['lambda_final'])!=3:
            raise ValueError('Expected the agreed final tilt 3')
        key=(row['comparison'],row['N'],row['seed'],row['threshold'])
        if key in keys: raise ValueError('Duplicated run; do not pool repeated levels')
        keys.add(key)
        for method,_,_ in METHODS:
            row[method]=float(row[method]); metric([row[method]],row['p_reference'],'ratio')
            if method in {'snis','crude'} and row[method]>1+1e-12:
                raise ValueError('SNIS and crude estimates must lie in [0,1]; do not clip invalid data')
    return rows


def make_figures(runs_csv,output_directory,title_prefix=''):
    rows=read_runs(runs_csv)
    out=Path(output_directory); out.mkdir(parents=True,exist_ok=True)
    groups=sorted({(r['comparison'],r['threshold']) for r in rows})
    sizes=sorted({r['N'] for r in rows})
    # Each plotted array consists of independent runs of exactly one setting.
    series={(policy,a,n,key):np.array([r[key] for r in rows if
        (r['comparison'],r['threshold'],r['N'])==(policy,a,n)],dtype=float)
        for policy,a in groups for n in sizes for key,_,_ in METHODS}
    audit={'source':str(Path(runs_csv).resolve()),'source_rows':len(rows),
        'box_percentiles':[25,75],'whisker_iqr':1.5,'zeros_included':True,
        'all_outliers_visible_in_full_panel':True,'groups':[]}
    paths=[]
    for kind in ['log_error','ratio']:
        arrays=[metric(v,REFERENCE[a][0],kind) for (policy,a,n,key),v in series.items() if len(v)]
        overall_max=max(float(v.max()) for v in arrays)
        upper=max(.4,overall_max*1.06+.015) if kind=='log_error' else max(2.,overall_max*1.08)
        zoom_upper=.4 if kind=='log_error' else 2.
        fig,axes=plt.subplots(len(groups),2,figsize=(20,5.3*len(groups)+2.1),squeeze=False)
        for i,(policy,a) in enumerate(groups):
            p,low,high=REFERENCE[a]
            annotation={}
            for col in range(2):
                ax=axes[i,col]
                for j,(key,label,color) in enumerate(METHODS):
                    for x,n in enumerate(sizes):
                        values=series[(policy,a,n,key)]
                        if not len(values): continue
                        transformed=metric(values,p,kind)
                        ax.bxp([statistics(transformed)],positions=[x+(j-1)*.24],widths=.205,
                            patch_artist=True,showfliers=True,showmeans=True,
                            boxprops={'facecolor':color,'edgecolor':color,'alpha':.7},
                            whiskerprops={'color':color},capprops={'color':color},
                            medianprops={'color':'black','linewidth':1.4},
                            meanprops={'marker':'D','markerfacecolor':'black','markeredgecolor':'white','markersize':4},
                            flierprops={'marker':'.','markersize':2.3,'markerfacecolor':color,'markeredgecolor':color,'alpha':.5})
                        if col==0:
                            entry={'kind':kind,'comparison':policy,'threshold':a,'N':n,'method':key,
                                'runs':len(values),'zeros':int((values==0).sum()),
                                'outside_zoom':int((transformed>zoom_upper).sum()),
                                'maximum':float(transformed.max())}
                            audit['groups'].append(entry); annotation[(n,key)]=entry
                ax.set_xlim(-.55,len(sizes)-.45)
                ax.set_ylim(0,zoom_upper if col==0 else upper)
                if kind=='ratio':
                    ax.axhspan(.5,1.5,color='#48a868',alpha=.13,zorder=0)
                    ax.axhspan(low/p,high/p,color='#8568af',alpha=.16,zorder=0)
                    ax.axhline(1,color='black',linestyle='--',linewidth=1)
                    if col==1:
                        ax.set_yscale('symlog',linthresh=1,linscale=1,base=10)
                        ticks=[0,.5,1]+[10.**k for k in range(1,int(math.ceil(math.log10(upper)))+1) if 10.**k<=upper]
                        ax.set_yticks(ticks,[f'{t:g}' for t in ticks])
                    ax.set_ylabel(r'$\widehat p_A/\widehat p_{\mathrm{ref}}$')
                else:
                    ax.axhline(0,color='#348855',linewidth=1.2)
                    ax.axhline(math.log10(2),color='#bc3434',linestyle='--',linewidth=1)
                    ax.set_ylabel(r'$\log_{10}(1+|\widehat p_A/\widehat p_{\mathrm{ref}}-1|)$')
                    if col==0:
                        relative=[0,.25,.5,.75,1,1.5]
                        ax.set_yticks(np.log10(1+np.array(relative)),[f'{np.log10(1+t):.3f} · {100*t:g}%' for t in relative])
                labels=[]
                for n in sizes:
                    count=len(series[(policy,a,n,'smc')])
                    if col==0 and count:
                        outside='/'.join(str(annotation[(n,k)]['outside_zoom']) for k,_,_ in METHODS)
                        zeros='/'.join(f"{100*annotation[(n,k)]['zeros']/count:g}" for k,_,_ in METHODS)
                        labels.append(f'{n}\nfuori: {outside}\nzeri [%]: {zeros}\nrun: {count}')
                    else: labels.append(f'{n}\nrun: {count}')
                ax.set_xticks(range(len(sizes)),labels,fontsize=9)
                ax.set_xlabel('Numero di particelle SMC N; budget crude abbinato')
                descriptor='Zoom' if col==0 else ('Distribuzione completa; lineare 0–1, log oltre 1' if kind=='ratio' else 'Distribuzione completa: tutti gli outlier')
                ax.set_title(f'Score ≥ {a:g} · {POLICIES[policy]} · {descriptor}\n'
                    f'Reference = {p:.5f}; IC 95% [{low:.5f}, {high:.5f}]',fontsize=12)
                ax.grid(axis='y',alpha=.18)
        handles=[Patch(facecolor=c,alpha=.7,label=label) for _,label,c in METHODS]
        handles += [Line2D([0],[0],color='black',label='Linea nera: mediana'),
            Line2D([0],[0],marker='D',color='none',markerfacecolor='black',label='Rombo nero: media di tutte le run')]
        if kind=='ratio':
            handles += [Patch(facecolor='#48a868',alpha=.2,label='Fascia 0,5–1,5'),
                Patch(facecolor='#8568af',alpha=.25,label='IC 95% della reference / reference')]
        else:
            handles += [Line2D([0],[0],color='#bc3434',linestyle='--',label='Errore relativo 100%\nanche quando la stima è zero')]
        title='Errore relativo logaritmico' if kind=='log_error' else 'Rapporto fra stima e reference'
        fig.suptitle(f'{title_prefix}{title} · MDLM, tilt 1→2→3, quattro mosse MH per livello',fontsize=17,y=.985)
        fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.047),fontsize=10)
        caption=('Box: 25°–75° percentile; baffi: 1,5 IQR. Mediana e media sono distinte; la media può cadere fuori dal box.\n'
            'Fuori scala: conteggi; zeri: percentuali. Ordine SMC / SNIS / crude, su tutte le run. Tutti gli outlier nel pannello completo.\n'
            'La reference è una stima da 200.000 campioni IID, non ground truth esatta. Nessuna media tra livelli o configurazioni.')
        if kind=='log_error': caption+='\nPiù basso è meglio; la metrica non distingue sovrastima e sottostima.'
        fig.text(.5,.014,caption,ha='center',fontsize=9)
        fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.22 if len(groups)==2 else .15,hspace=.72,wspace=.20)
        path=out/f'camera_ready_{kind}_boxplots.png'
        fig.savefig(path,dpi=300); plt.close(fig); paths.append(path)
    (out/'camera_ready_plot_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--title-prefix',default='')
    args=parser.parse_args()
    for path in make_figures(args.runs,args.output or args.runs.parent,args.title_prefix): print(path)
