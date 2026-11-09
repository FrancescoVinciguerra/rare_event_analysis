"""Render exactly two scientific figures from measured terminal pilot runs."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();root=args.results
    spec=importlib.util.spec_from_file_location('shared_level_boxes',Path(__file__).resolve().parents[1] / 'analysis/analyze_saved_levels.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    study=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['study']
    with (root/'runs.csv').open(encoding='utf-8') as f:records=list(csv.DictReader(f))
    keys=[(r['configuration'],r['seed'],r['threshold']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run')
    rows=[]
    for r in records:
        if r['timing_basis']!='measured_complete_terminal_run':raise ValueError('An assumed time entered the measured report')
        rows.append({'configuration':f'N{int(r["N"])}','N':int(r['N']),'moves_per_level':int(r['moves_per_level']),
            'threshold':float(r['threshold']),'lambda':float(r['lambda_final']),'p_reference':float(r['p_reference']),
            **{m:float(r[m]) for m in ['smc','snis','crude']}})
    events=study['plot_thresholds']
    rows=[r for r in rows if r['threshold'] in events]
    ns=sorted({r['N'] for r in rows});tilts=sorted({r['lambda'] for r in rows})
    counts=[sum(r['N']==n and r['lambda']==lam and r['threshold']==a for r in rows)
        for n in ns for lam in tilts for a in events]
    if not counts or min(counts)==0:raise ValueError('Incomplete grid: report all CSVs, wait for one completed run per setting before plotting')
    view={**study,'thresholds':events,
        'reference':{**study['reference'],'discoveries':[study['reference']['discoveries'][study['thresholds'].index(a)] for a in events]},
        'configurations':[{'id':f'N{n}','N':n,'moves_per_level':4} for n in ns]}
    methods=[('smc','SMC non normalizzato','#2878b5'),('snis','SMC + SNIS','#ed8b23'),('crude','Crude MC','#929292')]
    images=module.figures(root,view,rows,methods=methods,filename_prefix='accuracy_budget',
        title_suffix='confronto a tempo MISURATO',
        subtitle='Ogni tilt è una run terminale indipendente, senza resampling finale. SMC e MC misurati sulla stessa GPU dopo warm-up.',
        caption_description=f'Run per gruppo: {min(counts)}–{max(counts)}. Tilt e N separati, nessuna media fra tilt. Reference finite da 200.000 IID.',
        caption_tail='Rombo: media della quantità sull’asse, calcolata su tutte le run. Zeri MC riportati sotto ogni tilt. Riferimento per score 0,999: solo 26/200.000 eventi.',ratio_band=None)
    html=root/'REPORT.html'
    if html.exists():
        content=html.read_text(encoding='utf-8');marker='<!-- scientific-boxplots -->'
        content=content.split(marker)[0]+marker
        for image in images:content+=f'<p><a href="{image.relative_to(root).as_posix()}"><img src="{image.relative_to(root).as_posix()}" style="width:100%"></a></p>'
        html.write_text(content,encoding='utf-8')
    print(json.dumps({'figures':[str(x) for x in images],'new_simulations':False},indent=2))


if __name__=='__main__':main()
