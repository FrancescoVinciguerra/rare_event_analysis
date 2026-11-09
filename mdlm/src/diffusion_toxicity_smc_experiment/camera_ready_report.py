"""Portable, self-contained HTML/Markdown report with two embedded SVG figures.

Only numpy and the Python standard library; no installation on Avon required.
Figures are derived from individual estimates including all zeros and outliers.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path

import numpy as np

METHODS=[('smc','SMC non normalizzato','#2878b5'),('snis','SMC + SNIS','#ed8b23'),('crude','Crude MC','#8d959f')]
POLICIES={'matched_transitions':'pari transizioni','matched_wall_clock':'pari wall clock nominale'}
REFERENCE={.9:(.00758,.00720922849129587,.007969687256908654),.99:(.00222,.0020230756570391337,.0024360459894034315)}


def quartiles(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid estimates; no silent filtering')
    q1,median,q3=np.quantile(x,[.25,.5,.75]); iqr=q3-q1
    lo=x[x>=q1-1.5*iqr].min(); hi=x[x<=q3+1.5*iqr].max()
    return float(q1),float(median),float(q3),float(lo),float(hi),float(x.mean()),x[(x<lo)|(x>hi)]


def svg_figure(rows,kind):
    groups=sorted({(r.get('comparison','matched_transitions'),r['threshold']) for r in rows})
    sizes=sorted({r['N'] for r in rows}); data={}
    for policy,a in groups:
        p=REFERENCE[a][0]
        for n in sizes:
            subset=[r for r in rows if (r.get('comparison','matched_transitions'),r['threshold'],r['N'])==(policy,a,n)]
            for key,_,_ in METHODS:
                ratio=np.array([r[key]/p for r in subset])
                data[policy,a,n,key]=ratio if kind=='ratio' else np.log10(1+abs(ratio-1))
    maximum=max(float(v.max()) for v in data.values() if len(v))
    full=max(2.,maximum*1.08) if kind=='ratio' else max(.4,maximum*1.08+.01)
    width=2000; row_height=440; height=125+len(groups)*row_height+110
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
         '<rect width="100%" height="100%" fill="white"/>']
    def text(x,y,label,size=16,anchor='start',color='#222'):
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(label))}</text>')
    def line(x1,y1,x2,y2,color='#333',dash=''):
        out.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="1.5"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def rect(x,y,w,h,color,opacity=1.):
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{max(0,h):.2f}" fill="{color}" opacity="{opacity}"/>')
    title='Rapporto stima / reference' if kind=='ratio' else 'log10(1 + |stima/reference − 1|)'
    text(width/2,32,title,25,'middle')
    text(width/2,59,'MDLM · tilt 1→2→3 · 4 mosse MH per livello · ogni box contiene le singole run indipendenti',17,'middle')
    for j,(_,label,color) in enumerate(METHODS):
        x=280+j*460;rect(x,77,24,15,color,.7);text(x+32,91,label,17)
    for index,(policy,a) in enumerate(groups):
        for col in range(2):
            left=95+col*990;top=158+index*row_height;w=840;h=245
            limit=(2. if kind=='ratio' else .4) if col==0 else full
            log_axis=kind=='ratio' and col==1
            def transform(v): return v if not log_axis or v<=1 else 1+math.log10(v)
            def y(v): return top+h-h*transform(float(v))/transform(limit)
            def x(n,j): return left+(sizes.index(n)+.5)*w/len(sizes)+(j-1)*.22*w/len(sizes)
            text(left+w/2,top-30,f'Score ≥ {a:g} · {POLICIES[policy]} · '+('zoom' if col==0 else 'distribuzione completa'),19,'middle')
            text(left+w/2,top-10,f'Reference {REFERENCE[a][0]:.5f}; IC 95% [{REFERENCE[a][1]:.5f}, {REFERENCE[a][2]:.5f}]',14,'middle')
            if kind=='ratio':
                rect(left,y(1.5),w,y(.5)-y(1.5),'#48a868',.12)
                low,high=REFERENCE[a][1]/REFERENCE[a][0],REFERENCE[a][2]/REFERENCE[a][0]
                rect(left,y(high),w,y(low)-y(high),'#8568af',.18)
                ticks=[0,.5,1,1.5,2] if col==0 else [0,.5,1]+[10.**k for k in range(1,int(math.ceil(math.log10(limit)))+1) if 10.**k<=limit]
                labels=[f'{t:g}' for t in ticks]
            elif col==0:
                fractions=[0,.25,.5,.75,1,1.5];ticks=list(np.log10(1+np.array(fractions)))
                labels=[f'{v:.3f} · {100*f:g}%' for f,v in zip(fractions,ticks)]
            else: ticks=list(np.linspace(0,limit,6));labels=[f'{v:.2g}' for v in ticks]
            for tick,label in zip(ticks,labels):
                line(left,y(tick),left+w,y(tick),'#e4e4e4');text(left-9,y(tick)+5,label,14,'end')
            reference=1 if kind=='ratio' else math.log10(2)
            line(left,y(reference),left+w,y(reference),'#222' if kind=='ratio' else '#bc3434','6 4')
            line(left,top,left,top+h);line(left,top+h,left+w,top+h)
            clip=f'clip_{kind}_{index}_{col}'
            out.append(f'<defs><clipPath id="{clip}"><rect x="{left}" y="{top}" width="{w}" height="{h}"/></clipPath></defs><g clip-path="url(#{clip})">')
            for n in sizes:
                for j,(key,_,color) in enumerate(METHODS):
                    v=data[policy,a,n,key]
                    if not len(v): continue
                    q1,median,q3,lo,hi,mean,fliers=quartiles(v); cx=x(n,j);bw=.18*w/len(sizes)
                    line(cx,y(lo),cx,y(hi),color)
                    line(cx-bw/3,y(lo),cx+bw/3,y(lo),color);line(cx-bw/3,y(hi),cx+bw/3,y(hi),color)
                    rect(cx-bw/2,y(q3),bw,max(.6,y(q1)-y(q3)),color,.7)
                    line(cx-bw/2,y(median),cx+bw/2,y(median),'#111')
                    ym=y(mean)
                    out.append(f'<polygon points="{cx:.2f},{ym-4:.2f} {cx+4:.2f},{ym:.2f} {cx:.2f},{ym+4:.2f} {cx-4:.2f},{ym:.2f}" fill="black"/>')
                    for value in fliers:
                        out.append(f'<circle cx="{cx:.2f}" cy="{y(value):.2f}" r="1.8" fill="{color}" opacity=".55"/>')
            out.append('</g>')
            for n in sizes:
                cx=left+(sizes.index(n)+.5)*w/len(sizes);v=data[policy,a,n,'smc']
                text(cx,top+h+23,f'N={n}',17,'middle')
                text(cx,top+h+44,f'run: {len(v)}',13,'middle')
                if col==0:
                    outside='/'.join(str(int((data[policy,a,n,key]>limit).sum())) for key,_,_ in METHODS)
                    # Ratio=0 iff raw estimate=0. For log-error zeros must be
                    # counted from raw estimates, not from the transformed value.
                    raw=[r for r in rows if (r.get('comparison','matched_transitions'),r['threshold'],r['N'])==(policy,a,n)]
                    zeros='/'.join(str(sum(r[key]==0 for r in raw)) for key,_,_ in METHODS)
                    text(cx,top+h+65,'fuori: '+outside,12,'middle');text(cx,top+h+84,'zeri: '+zeros,12,'middle')
            text(left+w/2,top+h+111,'Numero di particelle SMC; budget crude abbinato'+(' · scala lineare 0–1 e log oltre 1' if log_axis else ''),15,'middle')
    text(width/2,height-69,'Box: 25°–75° percentile; baffi: 1,5 IQR. Linea nera: mediana; rombo nero: media di tutte le run.',16,'middle')
    text(width/2,height-46,'Conteggi fuori scala e zeri: SMC / SNIS / crude. Tutti gli outlier sono visibili nel pannello completo.',16,'middle')
    text(width/2,height-23,('Linea a 1, fascia verde 0,5–1,5, fascia viola: incertezza della reference.' if kind=='ratio' else 'Più basso è meglio. Linea tratteggiata: errore 100%, anche per stime zero. La metrica non distingue il segno.'),16,'middle')
    out.append('</svg>');return ''.join(out)


def table(headers,records):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in records)+'</tbody></table>'


def build_report(study,root:Path,rows,summary):
    if not rows: raise ValueError('No real run data for report')
    expected=study['independent_runs']*len(study['particle_counts'])
    completed=len({(r['N'],r['seed']) for r in rows})
    status='PILOT COMPLETO' if completed==expected else 'PILOT PARZIALE'
    unique={(r['N'],r['seed']):r for r in rows}
    diagnostics=[]
    for n in sorted({r['N'] for r in rows}):
        group=[r for r in unique.values() if r['N']==n]
        def med(key,scale=1):
            values=[r[key]*scale for r in group if r.get(key) is not None]
            return f'{np.median(values):.3g}' if values else '—'
        diagnostics.append([n,len(group),med('native_transitions'),med('smc_runtime_seconds'),
            med('acceptance_fraction'),med('smc_ess_fraction'),med('reverse_ess_fraction'),
            med('distinct_root_ancestors'),med('distinct_terminal_sequences')])
    figures=[svg_figure(rows,k) for k in ['log_error','ratio']]
    interpretation=[]
    for s in summary:
        interpretation.append([POLICIES[s['comparison']],s['threshold'],s['N'],
            dict((k,label) for k,label,_ in METHODS)[s['method']],s['runs'],
            f'{s["zero_percent"]:.1f}%',f'{s["ratio_q50"]:.3g}',
            f'{s["ratio_q90"]:.3g}',f'{s["ratio_q99"]:.3g}',f'{s["median_log_relative_error"]:.3g}'])
    clocks=[r['wall_clock_ratio'] for r in rows if r.get('comparison')=='matched_wall_clock' and r.get('wall_clock_ratio') is not None]
    clock_note=('Il numero di campioni crude a tempo è fissato prima della generazione usando una calibrazione indipendente. Il matching wall clock è approssimato; '
        +f'rapporto tempo crude/SMC mediano {np.median(clocks):.3f}, massimo {max(clocks):.3f}. '
        +'I tempi effettivi e lo sforamento sono nel CSV.' if clocks else
        'Non sono ancora disponibili baseline misurate a pari wall clock; il report non le ricostruisce dal solo conteggio delle transizioni.')
    title='RollBack SMC — camera-ready: confronto a budget limitato'
    css='body{font:16px Arial,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#222}h1{font-size:30px}svg{width:100%;height:auto}table{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto}td,th{padding:8px;border:1px solid #ddd;text-align:right}th{background:#eef2f6}.warning{background:#fff3cd;padding:18px;border-left:5px solid #c58a00}section{margin-top:30px}a{color:#17639a}'
    body=f'<h1>{title}</h1><p><b>{status}: {completed}/{expected} popolazioni</b> · {study["independent_runs"]} run previste per N.</p>'
    body+='<p class="warning">Otto run pilota per numerosità servono a verificare funzionamento e costi. Non bastano per dichiarare un vincitore o stimare affidabilmente le code delle distribuzioni.</p>'
    body+='<p>Tre livelli di tilt 1,2,3; quattro mosse RollBack–MH per livello. Stessi stimatori dalla stessa popolazione finale; nessuna media fra livelli. Soglie 0,9 e 0,99 valutate sulla stessa popolazione. Systematic quando ESS/N &lt; 0,5; nessun resampling finale.</p>'
    body+='<p>Reference da 200.000 completamenti IID: 1516 eventi alla soglia 0,9 (0,00758), 444 alla soglia 0,99 (0,00222). È una reference Monte Carlo con incertezza, non ground truth esatta.</p>'
    body+='<p>Il costo conta tutti i passi DDPM e il completamento categorico di ogni traiettoria e suffisso proposto, inclusi quelli rifiutati. A pari transizioni crude riceve floor(costo/33) traiettorie complete. Warm-up e caricamento modelli sono esclusi dai tempi per stima e registrati separatamente.</p>'
    body+='<p>'+escape(clock_note)+'</p>'
    body+='<p><a href="runs.csv">Stime di ogni run</a> · <a href="summary.csv">Tabella completa</a> · <a href="REPORT.md">Sintesi Markdown</a> · <a href="manifest.json">Manifest</a></p>'
    for heading,figure in zip(['Errore relativo logaritmico','Rapporto stima/reference'],figures): body+='<section><h2>'+heading+'</h2>'+figure+'</section>'
    body+='<section><h2>Distribuzioni delle singole run</h2>'+table(['Confronto','Soglia','N','Metodo','Run','Zero','Rapporto Q50','Q90','Q99','Mediana errore log'],interpretation)+'</section>'
    body+='<section><h2>Diagnostiche della popolazione SMC</h2><p>Mediane tra run. SMC non normalizzato e SNIS condividono particelle e genealogia; un ESS alto non garantisce accuratezza.</p>'+table(['N','Run','Transizioni','Tempo SMC [s]','Acceptance','ESS/N','ESS inverso/N','Radici','Terminali distinti'],diagnostics)+'</section>'
    body+='<p>Le due soglie e le due politiche di matching riutilizzano le stesse stime SMC: non sono repliche indipendenti aggiuntive. Il confronto della media con la reference richiede l’incertezza di entrambe; una mediana bassa non prova bias dello stimatore.</p>'
    (root/'REPORT.html').write_text('<!doctype html><html lang="it"><meta charset="utf-8"><title>'+title+'</title><style>'+css+'</style>'+body+'</html>',encoding='utf-8')
    md=[f'# {title}','',f'{status}: {completed}/{expected} popolazioni.','',
        'Risultati pilota: non trarre conclusioni definitive sulle performance.',
        'Reference finite IID, non probabilità esatte. Nessuna media fra tilt o configurazioni.','',clock_note,'',
        '| Confronto | Soglia | N | Metodo | Run | Zero | Rapporto Q50 | Q90 | Q99 | Mediana errore log |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    md+=['| '+' | '.join(map(str,r))+' |' for r in interpretation]
    md+=['','I due grafici completi, zoom, diagnostiche e note metodologiche sono incorporati in REPORT.html.',
        'PNG ad alta risoluzione: eseguire il plotter locale sul runs.csv scaricato.']
    (root/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (root/'report_status.json').write_text(json.dumps({'completed_populations':completed,
        'expected_populations':expected,'status':status,'independent_replicates_are_populations':True,
        'reference_is_exact':False,'all_zero_estimates_included':True},indent=2),encoding='utf-8')
    return root/'REPORT.html'
