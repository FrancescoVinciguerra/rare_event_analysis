"""Two figures from saved same-run checkpoints; no simulation or model load."""
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
from matplotlib.patches import Patch

METHODS=[('smc','SMC non normalizzato','#2878b5'),
         ('snis','SMC + SNIS','#ed8b23'),('crude','Crude MC','#929292')]


def render(root, *, checkpoint_file='checkpoint_runs.csv', suffix='', reference_note=None):
    rows=list(csv.DictReader((root/checkpoint_file).open(encoding='utf-8')))
    rows=[r for r in rows if int(r['level'])>0 and int(r['sweep'])==4]
    if not rows:raise ValueError('No completed level populations')
    if {int(r['N']) for r in rows}!={128}:raise ValueError('Do not pool particle counts')
    events=sorted({float(r['threshold']) for r in rows})
    levels=sorted({float(r['lambda']) for r in rows})
    potentials={r.get('phi','log_probability') for r in rows}
    if len(potentials)!=1:raise ValueError('Do not pool different Phi choices')
    phi_name=next(iter(potentials))
    phi_label='logit(q)' if phi_name=='logit' else 'log(q)'
    keys=[(r['configuration'],r['seed'],r['lambda'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate independent observation')
    calibrated=any(r.get('MC_comparison_available')=='True' and r['MC_comparison_measured']=='False' for r in rows)
    if not all(r.get('MC_comparison_available',r['MC_comparison_measured'])=='True' for r in rows):
        raise ValueError('A level lacks a valid MC comparison')
    paths=[];audit=[]
    for kind in ['log_error','ratio']:
        all_ratios=np.array([float(r[m])/float(r['p_reference']) for r in rows for m,_,_ in METHODS])
        if not np.isfinite(all_ratios).all() or (all_ratios<0).any():raise ValueError('Invalid estimates')
        transformed=all_ratios if kind=='ratio' else np.log10(1+abs(all_ratios-1))
        zoom=2. if kind=='ratio' else .4
        full=max(zoom,float(transformed.max())*1.08+.01)
        fig,axes=plt.subplots(2*len(events),2,figsize=(28,11*len(events)),squeeze=False)
        for e,a in enumerate(events):
            for view in [0,1]:
                for column in [0,1]:
                    ax=axes[2*e+view,column]
                    positive_only=column==0
                    for pos,lam in enumerate(levels):
                        group=[r for r in rows if float(r['threshold'])==a and float(r['lambda'])==lam]
                        counts=[]
                        for m_index,(method,_,color) in enumerate(METHODS):
                            ratios=np.array([float(r[method])/float(r['p_reference']) for r in group])
                            values=ratios if kind=='ratio' else np.log10(1+abs(ratios-1))
                            zero_count=int((ratios==0).sum())
                            if positive_only:values=values[ratios>0]
                            outside=int((values>zoom).sum())
                            counts.append(outside)
                            short=['S','SN','MC'][m_index]
                            ax.text(pos,1.018+.048*(2-m_index),f'{short}: {zero_count}/{len(group)}',
                                transform=ax.get_xaxis_transform(),color=color,ha='center',fontsize=6.5,
                                fontweight='bold',clip_on=False)
                            if len(values):
                                ax.boxplot(values,positions=[pos+(m_index-1)*.24],widths=.20,
                                    patch_artist=True,whis=1.5,showmeans=True,showfliers=True,
                                    boxprops=dict(facecolor=color,edgecolor=color,alpha=.7),
                                    whiskerprops=dict(color=color),capprops=dict(color=color),
                                    medianprops=dict(color='black',linewidth=1.4),
                                    meanprops=dict(marker='D',markersize=3.5,markerfacecolor='black',markeredgecolor='white'),
                                    flierprops=dict(marker='.',markersize=3,markerfacecolor=color,markeredgecolor=color))
                            audit.append(dict(kind=kind,threshold=a,**{'lambda':lam},method=method,
                                selection='positive_only' if positive_only else 'all_runs',view='zoom' if not view else 'full',
                                total_runs=len(group),zero_count=zero_count,plotted_count=len(values),outside_zoom=outside))
                        if view==0:
                            ax.text(pos,-.16,'/'.join(map(str,counts)),transform=ax.get_xaxis_transform(),
                                    ha='center',fontsize=6.5)
                    ax.set_xticks(range(len(levels)),[f'{x:g}' for x in levels],fontsize=8)
                    ax.set_xlim(-.6,len(levels)-.4);ax.set_ylim(0,full if view else zoom)
                    if kind=='ratio' and view:ax.set_yscale('symlog',linthresh=1)
                    ax.axhline(1 if kind=='ratio' else math.log10(2),color='#ad3434',ls='--',lw=1)
                    ax.set_xlabel(r'Tilt corrente $\lambda$'+ (' · sotto: fuori zoom S/SN/MC' if not view else ''),labelpad=21 if not view else 8)
                    ax.set_ylabel(r'$\widehat p_A/\widehat p_{\mathrm{ref}}$' if kind=='ratio'
                                  else r'$\log_{10}(1+|\widehat p_A/\widehat p_{\mathrm{ref}}-1|)$')
                    ax.grid(axis='y',alpha=.2)
                    ax.set_title(f'Score ≥ {a:g} · N=128 · '+('solo stime positive, zeri esclusi' if positive_only else 'tutte le run, zeri inclusi')
                                 +'\n'+('Distribuzione completa, tutti gli outlier' if view else 'Zoom sul centro'),pad=55,fontsize=12)
        fig.suptitle(('Rapporto stima/riferimento' if kind=='ratio' else 'Errore relativo logaritmico')
                     +f' · Φ = {phi_label} · popolazioni della STESSA run a tilt diversi',fontsize=18,y=.99)
        fig.text(.5,.964,'Conteggi sopra i box: run a zero / run totali. Ogni box aggrega seed indipendenti allo stesso tilt, dopo 4 MH.',ha='center',fontsize=12)
        fig.legend(handles=[Patch(facecolor=c,label=label) for _,label,c in METHODS],loc='lower center',ncol=3,bbox_to_anchor=(.5,.055))
        if calibrated:
            cost_text=('Tempo SMC cumulativo misurato. MC: prefissi di campioni indipendenti già salvati; '
                       'n = floor(tempo SMC / costo MC calibrato). Costo MC stimato, non rimisurato.')
        else:
            ratio_values=np.array([float(r['wall_clock_ratio']) for r in rows])
            mismatch=sum(r['clock_within_tolerance']!='True' for r in rows)
            cost_text=(f'Tempi completi misurati: MC/SMC fra {ratio_values.min():.3f} e {ratio_values.max():.3f}; '
                       +f'{mismatch} righe evento oltre ±5%, mantenute.')
        foot=('Linea nera: mediana. Rombo: media della quantità sull’asse. Box: 25°–75°; baffi: 1,5 IQR. '
              'Le statistiche dei positivi sono condizionate a una stima positiva.\n'
              +('Linea tratteggiata: rapporto 1.' if kind=='ratio' else 'Linea tratteggiata: errore relativo 100%, anche per stima zero.')
              +' Nessuna media fra tilt. Stime nulle e outlier conservati.\n'
              +cost_text+' Ogni stima usa 1 GPU e 4 CPU; 3 seed in parallelo.\n'
              +(reference_note or 'Riferimenti stimati: score 0,99 = 444/200.000; score 0,999 = 26/200.000.'))
        fig.text(.5,.013,foot,ha='center',fontsize=9)
        fig.subplots_adjust(left=.055,right=.99,top=.91,bottom=.14,hspace=.88,wspace=.19)
        figure_dir = root / 'plots'
        figure_dir.mkdir(exist_ok=True)
        path=figure_dir/f'same_run_ladder_{kind}{suffix}.png';fig.savefig(path,dpi=300);plt.close(fig);paths.append(path)
    (root/f'same_run_ladder_plot_audit{suffix}.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    report=root/'REPORT.html'
    if report.exists() and not suffix:
        content=report.read_text(encoding='utf-8').split('<!-- same-run-figures -->')[0]+'<!-- same-run-figures -->'
        for path in paths:content+=f'<p><a href="{path.relative_to(root).as_posix()}"><img src="{path.relative_to(root).as_posix()}" style="width:100%"></a></p>'
        report.write_text(content,encoding='utf-8')
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();print(json.dumps({'figures':[str(p) for p in render(args.results)]},indent=2))
