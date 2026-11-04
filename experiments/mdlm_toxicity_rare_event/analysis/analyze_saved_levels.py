"""Recover per-level SMC/SNIS estimates from existing GPU weighting ledgers.

NO simulations and NO independent tilted draws. For k<R, the incoming ledger
at k+1 contains the post-M_k rewards and weights, and the current log Z_hat.
These are populations of the ACTUAL continuing run, including any resampling
at k. They are NOT hypothetical terminal forks with resampling suppressed.
Intermediate wall times were not saved; no time-matched crude is invented.

Standalone dependencies: numpy, matplotlib, Python standard library.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from html import escape
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

METHODS=[('smc','SMC non normalizzato','#2878b5'),('snis','SMC + SNIS','#ed8b23')]

def logsumexp(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return -math.inf
    m=float(x.max());return m+math.log(float(np.exp(x-m).sum()))

def corrections(logw,phi,logz,level_lambda,threshold,*,threshold_tolerance=1e-6):
    logw=np.asarray(logw,dtype=float);phi=np.asarray(phi,dtype=float)
    if len(logw)!=len(phi) or not np.isfinite(logw).all() or not np.isfinite(phi).all():
        raise ValueError('Invalid saved particle vectors')
    if abs(logsumexp(logw))>1e-9:raise ValueError('SMC weights are not normalized')
    distance=float(abs(phi-math.log(threshold)).min())
    if distance<=threshold_tolerance:raise ValueError('Event is numerically ambiguous from log score; require original classifier scores')
    event=phi>=math.log(threshold);reverse=logw-level_lambda*phi
    denominator=logsumexp(reverse);numerator=logsumexp(reverse[event])
    normalized=reverse-denominator
    return {'smc':math.exp(logz+numerator),'snis':math.exp(numerator-denominator),
        'event_particles':int(event.sum()),'reverse_mass':math.exp(logz+denominator),
        'reverse_ess_fraction':math.exp(-logsumexp(2*normalized))/len(phi),
        'reverse_max_weight':float(np.exp(normalized).max()),
        'threshold_log_distance_min':distance}

def wilson(hits,total):
    p=hits/total;z=1.959963984540054;d=1+z*z/total
    c=(p+z*z/(2*total))/d
    h=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return c-h,c+h

def recover(root):
    study=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['study']
    configs={x['id']:x for x in study['configurations']};reference=study['reference']
    truth={a:h/reference['samples'] for a,h in zip(study['thresholds'],reference['discoveries'])}
    rows=[];sources={};terminal_checks=0
    # Both classifier outputs are float32. Keep a two-epsilon log-score guard,
    # verify it against saved original scores, and reject any closer boundary.
    threshold_tolerance=2*float(np.finfo(np.float32).eps)
    maximum_saved_log_score_discrepancy=0.
    files=sorted((root/'workers').glob('worker_*/populations/*.json'))
    for file in files:
        if not (file.parent.parent/'runs'/file.name).exists():continue
        p=json.loads(file.read_text(encoding='utf-8'));sources[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
        for particle in p['particles']:
            q=particle.get('event_toxic_probability')
            if q is not None and q>=.8:
                discrepancy=abs(math.log(q)-particle['reward'])
                maximum_saved_log_score_discrepancy=max(maximum_saved_log_score_discrepancy,discrepancy)
                if discrepancy>=threshold_tolerance:raise ValueError('Saved score/log-score mismatch exceeds numerical guard')
        matching=[case for case in configs if file.name.startswith(case+'_')]
        if not matching:raise ValueError(f'Population does not match a study configuration: {file.name}')
        case=max(matching,key=len);config=configs[case];d=p['level_diagnostics'];n=p['summary']['num_particles']
        cumulative=n*33
        for k,current in enumerate(d):
            lam=float(current['eta'])
            if k+1<len(d):
                nxt=d[k+1];logw=nxt['previous_normalized_log_weights'];phi=nxt['rewards_before_mutation'];logz=nxt['log_z_before_weight']
                if not math.isclose(logz,current['log_normalizer_estimate'],abs_tol=1e-10):raise ValueError('Wrong level normalizer')
            else:
                logw=[x['final_log_weight'] for x in p['particles']];phi=[x['reward'] for x in p['particles']]
                logz=p['summary']['log_normalizer_estimate']
            mutations=[m for m in p['mutation_diagnostics'] if m['level']==k+1]
            cumulative+=sum(m['rollback_particle_steps']+m['proposals'] for m in mutations)
            if len(mutations)!=config['moves_per_level']:raise ValueError('Missing mutation sweeps')
            for a in study['thresholds']:
                value=corrections(logw,phi,logz,lam,a,threshold_tolerance=threshold_tolerance)
                if k+1==len(d):
                    saved=p['corrections'][str(a)]
                    for field in ['smc','snis','reverse_mass']:
                        if not math.isclose(value[field],saved[field],rel_tol=1e-11,abs_tol=1e-30):raise ValueError('Final estimates fail reconstruction')
                    if value['event_particles']!=saved['event_particles']:raise ValueError('Final event count mismatch')
                    terminal_checks+=1
                rows.append({'configuration':case,'source_file':str(file.relative_to(root)),
                    'lambda_final':float(d[-1]['eta']),'N':n,'moves_per_level':config['moves_per_level'],
                    'seed':p['summary']['seed'],'level':k+1,'lambda':lam,'threshold':a,'p_reference':truth[a],
                    'log_z_hat':logz,'smc_ess_fraction':math.exp(-logsumexp(2*np.array(logw)))/n,
                    'smc_max_weight':float(np.exp(logw).max()),'resampled_at_level':current['resampled'],
                    'distinct_root_ancestors':current['distinct_root_ancestors'],
                    'native_transitions_cumulative':cumulative,
                    'matches_no_final_resampling_policy':not current['resampled'],
                    'runtime_seconds':p['summary']['runtime_seconds'] if k+1==len(d) else None,
                    **value})
        if 'total_decoder_particle_steps' in p['summary']:
            expected_transitions=int(p['summary']['total_decoder_particle_steps'])+n+sum(m['proposals'] for m in p['mutation_diagnostics'])
            if cumulative!=expected_transitions:raise ValueError('Final cumulative native-transition count differs from saved cost')
    keys=[(r['configuration'],r['seed'],r['level'],r['threshold']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run or level')
    audit={'simulations_rerun':False,'populations':len(sources),'rows':len(rows),
        'source_sha256':sources,'terminal_event_checks':terminal_checks,
        'reconstruction':'For k<R use next weighting ledger incoming weights, scores and log Z; final k uses terminal particle record.',
        'population_policy':'Actual continuing run, including intermediate resampling; not terminal no-resampling forks.',
        'intermediate_wall_times_available':False,'crude_wall_clock_comparison_invented':False,
        'individual_MH_sweep_populations_available':False,'levels_within_run_are_correlated':True,
        'averaging_estimates_across_levels':False,
        'log_score_threshold_tolerance':threshold_tolerance,
        'maximum_saved_log_score_discrepancy':maximum_saved_log_score_discrepancy,
        'minimum_log_score_distance_to_threshold':min(r['threshold_log_distance_min'] for r in rows)}
    return study,rows,audit

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(rows, methods=None):
    methods=METHODS if methods is None else methods
    out=[]
    for case,a,lam in sorted({(r['configuration'],r['threshold'],r['lambda']) for r in rows}):
        group=[r for r in rows if (r['configuration'],r['threshold'],r['lambda'])==(case,a,lam)]
        for method,_,_ in methods:
            x=np.array([r[method]/r['p_reference'] for r in group]);metric=np.log10(1+abs(x-1))
            out.append({'configuration':case,'N':group[0]['N'],'moves_per_level':group[0]['moves_per_level'],
                'threshold':a,'lambda':lam,'method':method,'runs':len(x),
                'zero_percent':float(100*(x==0).mean()),'median_log_relative_error':float(np.median(metric)),
                **{f'ratio_q{int(q*100):02d}':float(np.quantile(x,q)) for q in [.1,.25,.5,.75,.9,.99]},
                'resampling_runs':sum(bool(r['resampled_at_level']) for r in group)})
    return out

def figures(root,study,rows,*,methods=None,filename_prefix='levels',title_suffix=None,subtitle=None,caption_tail=None,caption_description=None,ratio_band=(.5,1.5)):
    methods=METHODS if methods is None else methods
    cases=sorted({r['configuration'] for r in rows});events=study['thresholds'];lambdas=sorted({r['lambda'] for r in rows})
    configs={x['id']:x for x in study['configurations']};paths=[];plot_audit=[]
    for kind in ['log_error','ratio']:
        all_ratios=np.array([r[m]/r['p_reference'] for r in rows for m,_,_ in methods])
        maximum=float((all_ratios if kind=='ratio' else np.log10(1+abs(all_ratios-1))).max())
        full=max(2.,maximum*1.08) if kind=='ratio' else max(.4,maximum*1.07+.01)
        zoom=2. if kind=='ratio' else .4
        fig,axes=plt.subplots(len(cases),2*len(events),figsize=(14.5*len(events),4.1*len(cases)+3),squeeze=False)
        for i,case in enumerate(cases):
            cfg=configs[case]
            for j,a in enumerate(events):
                hits=study['reference']['discoveries'][j];total=study['reference']['samples'];p=hits/total;lo,hi=wilson(hits,total)
                for view in range(2):
                    ax=axes[i,2*j+view];labels=[]
                    for pos,lam in enumerate(lambdas):
                        group=[r for r in rows if (r['configuration'],r['threshold'],r['lambda'])==(case,a,lam)]
                        outside=[];zeros=[]
                        for method_index,(m,_,color) in enumerate(methods):
                            ratio=np.array([r[m]/p for r in group]);values=ratio if kind=='ratio' else np.log10(1+abs(ratio-1))
                            spacing=.24 if len(methods)>2 else .29
                            ax.bxp([boxplot_stats(values,whis=1.5)[0]],positions=[pos+(method_index-(len(methods)-1)/2)*spacing],widths=spacing*.86,
                                patch_artist=True,showmeans=True,showfliers=True,
                                boxprops={'facecolor':color,'edgecolor':color,'alpha':.7},whiskerprops={'color':color},capprops={'color':color},
                                medianprops={'color':'black','linewidth':1.2},meanprops={'marker':'D','markerfacecolor':'black','markeredgecolor':'white','markersize':3.5},
                                flierprops={'marker':'.','markersize':2.5,'markerfacecolor':color,'markeredgecolor':color})
                            outside.append(int((values>zoom).sum()));zeros.append(int((ratio==0).sum()))
                            if view==0:plot_audit.append({'kind':kind,'case':case,'threshold':a,'lambda':lam,'method':m,'runs':len(values),'zeros':zeros[-1],'outside_zoom':outside[-1]})
                        counts='/'.join(map(str,outside));zero_counts='/'.join(map(str,zeros))
                        labels.append(f'{lam:g}\nfuori: {counts}\nzeri: {zero_counts}' if view==0 else f'{lam:g}')
                    ax.set_xlim(-.55,len(lambdas)-.45);ax.set_ylim(0,zoom if view==0 else full)
                    ax.set_xticks(range(len(lambdas)),labels,fontsize=7.3);ax.set_xlabel(r'Tilt corrente $\lambda$')
                    if kind=='ratio':
                        ax.axhline(1,color='black',linestyle='--',linewidth=1)
                        if ratio_band is not None:ax.axhspan(*ratio_band,color='#48a868',alpha=.12,zorder=0)
                        ax.axhspan(lo/p,hi/p,color='#8568af',alpha=.13,zorder=0)
                        if view:ax.set_yscale('symlog',linthresh=1,linscale=1,base=10)
                        ax.set_ylabel(r'$\widehat p_A/\widehat p_{\rm ref}$')
                    else:
                        ax.axhline(math.log10(2),color='#bc3434',linestyle='--',linewidth=1)
                        ax.set_ylabel(r'$\log_{10}(1+|\widehat p_A/\widehat p_{\rm ref}-1|)$')
                        if not view:
                            e=np.array([0,.25,.5,.75,1,1.5]);ax.set_yticks(np.log10(1+e),[f'{v:.3f} · {100*x:g}%' for v,x in zip(np.log10(1+e),e)],fontsize=8)
                    ax.set_title(f"{case}: N={cfg['N']}, {cfg['moves_per_level']} MH/livello · score ≥{a:g}\n"+
                        ('Zoom' if view==0 else 'Distribuzione completa, tutti gli outlier'),fontsize=10)
                    ax.grid(axis='y',alpha=.18)
        handles=[Patch(facecolor=c,alpha=.7,label=label) for _,label,c in methods]
        handles.extend([Line2D([0],[0],color='black',label='Mediana'),Line2D([0],[0],color='none',marker='D',markerfacecolor='black',label='Media di tutte le run')])
        if kind=='log_error':handles.append(Line2D([0],[0],color='#bc3434',linestyle='--',label='Errore relativo 100%, anche per stima zero'))
        else:
            if ratio_band is not None:handles.append(Patch(facecolor='#48a868',alpha=.15,label=f'Fascia {ratio_band[0]:g}–{ratio_band[1]:g}'))
            handles.append(Patch(facecolor='#8568af',alpha=.15,label='Incertezza della reference'))
        title='Errore relativo logaritmico' if kind=='log_error' else 'Rapporto stima/reference'
        fig.suptitle(title+' · '+(title_suffix or 'stime lungo le run SMC salvate'),fontsize=18,y=.985)
        fig.text(.5,.952,subtitle or 'Ogni punto usa il tilt corrente; popolazioni dopo MH con il resampling intermedio effettivamente eseguito.',ha='center',fontsize=11)
        fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.063),fontsize=10)
        order=' / '.join(m.upper() for m,_,_ in methods)
        caption=(f'Box: 25°–75° percentile; linea nera: mediana; rombo: media; baffi: 1,5 IQR. Fuori scala e zeri: conteggi {order}.\n'
            +(caption_description or '8 run indipendenti per configurazione. I livelli della stessa run sono correlati e non vengono mediati. Reference finite da 200.000 IID.')+'\n'
            +(caption_tail or 'Queste NON sono copie terminali senza resampling finale. Mancano i tempi intermedi: nessun confronto MC a pari wall clock è ricostruito.'))
        fig.text(.5,.013,caption,ha='center',fontsize=9)
        fig.subplots_adjust(left=.05,right=.99,top=.895,bottom=.205,hspace=.65,wspace=.29)
        path=root/f'{filename_prefix}_{kind}_boxplots.png';fig.savefig(path,dpi=300);plt.close(fig);paths.append(path)
    audit_name='level_plot_audit.json' if filename_prefix=='levels' else f'{filename_prefix}_plot_audit.json'
    (root/audit_name).write_text(json.dumps(plot_audit,indent=2),encoding='utf-8')
    return paths

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--data-only',action='store_true',help='Recover and validate saved populations without creating plots')
    a=parser.parse_args();root=a.results
    study,rows,audit=recover(root);summary=summarize(rows)
    write_csv(root/'level_estimates.csv',rows);write_csv(root/'level_summary.csv',summary)
    (root/'level_reconstruction_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if a.data_only:
        print(json.dumps({'rows':len(rows),'independent_runs':audit['populations'],'new_simulations':False,
            'terminal_checks':audit['terminal_event_checks'],'individual_MH_sweep_populations_available':False},indent=2))
        return
    paths=figures(root,study,rows)
    note=('Stime dopo le mutazioni lungo le 24 run originali. A ogni livello si usa il suo tilt, non 3. '
          'Per k<R, score, pesi e Z sono recuperati dagli input della pesatura successiva. '
          'Il livello 3 riproduce gli stimatori e gli eventi già salvati. '
          'Sono presenti i resampling intermedi della run originale: questi punti non riproducono '
          'run che terminino a ogni livello escludendo il rispettivo resampling finale. '
          'I tempi cumulativi intermedi e le copie terminali non sono salvati: nessun MC a pari tempo viene inventato.')
    body='<h1>Stimatori lungo i livelli delle run salvate</h1><p>'+escape(note)+'</p>'
    for path in paths:body+=f'<p><a href="{path.name}"><img src="{path.name}" style="width:100%"></a></p>'
    body+='<p><a href="level_estimates.csv">Singole run e livelli</a> · <a href="level_summary.csv">Quantili e zeri</a> · <a href="level_reconstruction_audit.json">Controlli e provenienza</a></p>'
    (root/'LEVELS_REPORT.html').write_text('<!doctype html><html lang="it"><meta charset="utf-8"><style>body{font:16px Arial;margin:24px}p{max-width:1600px}</style>'+body+'</html>',encoding='utf-8')
    (root/'LEVELS_ANALYSIS.md').write_text('# Analisi dei livelli salvati\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'independent_runs':audit['populations'],'new_simulations':False,'figures':[str(p) for p in paths]},indent=2))

if __name__=='__main__':main()
