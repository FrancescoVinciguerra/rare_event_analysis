"""Portable SVG figures for separate exploration configurations."""
import math
from html import escape
import numpy as np
from .camera_ready_report import METHODS, POLICIES, REFERENCE, quartiles

def svg_figure(rows,kind):
    groups=sorted({(r.get('comparison','matched_transitions'),r['threshold']) for r in rows})
    sizes=sorted({r['configuration'] for r in rows}); data={}
    case_labels={r['configuration']:r['configuration_label'] for r in rows}
    for policy,a in groups:
        p=REFERENCE[a][0]
        for n in sizes:
            subset=[r for r in rows if (r.get('comparison','matched_transitions'),r['threshold'],r['configuration'])==(policy,a,n)]
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
    text(width/2,59,'MDLM · schedule graduali fino a tilt 3 · ogni box contiene le singole run indipendenti',17,'middle')
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
                text(cx,top+h+23,case_labels[n],17,'middle')
                text(cx,top+h+44,f'run: {len(v)}',13,'middle')
                if col==0:
                    outside='/'.join(str(int((data[policy,a,n,key]>limit).sum())) for key,_,_ in METHODS)
                    # Ratio=0 iff raw estimate=0. For log-error zeros must be
                    # counted from raw estimates, not from the transformed value.
                    raw=[r for r in rows if (r.get('comparison','matched_transitions'),r['threshold'],r['configuration'])==(policy,a,n)]
                    zeros='/'.join(str(sum(r[key]==0 for r in raw)) for key,_,_ in METHODS)
                    text(cx,top+h+65,'fuori: '+outside,12,'middle');text(cx,top+h+84,'zeri: '+zeros,12,'middle')
            text(left+w/2,top+h+111,'Configurazione: N / livelli / mosse; crude a pari transizioni'+(' · scala lineare 0–1 e log oltre 1' if log_axis else ''),15,'middle')
    text(width/2,height-69,'Box: 25°–75° percentile; baffi: 1,5 IQR. Linea nera: mediana; rombo nero: media di tutte le run.',16,'middle')
    text(width/2,height-46,'Conteggi fuori scala e zeri: SMC / SNIS / crude. Tutti gli outlier sono visibili nel pannello completo.',16,'middle')
    text(width/2,height-23,('Linea a 1, fascia verde 0,5–1,5, fascia viola: incertezza della reference.' if kind=='ratio' else 'Più basso è meglio. Linea tratteggiata: errore 100%, anche per stime zero. La metrica non distingue il segno.'),16,'middle')
    out.append('</svg>');return ''.join(out)
