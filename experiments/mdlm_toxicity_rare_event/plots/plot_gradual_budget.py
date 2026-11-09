"""Compare positive-only and all-run distributions of saved estimates."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


METHODS = [('smc', 'SMC non normalizzato', '#2878b5'),
           ('snis', 'SMC + SNIS', '#ed8b23'), ('crude', 'Crude MC', '#929292')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    root = parser.parse_args().results
    with (root / 'runs.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if float(r['threshold']) == float(r['target_threshold'])]
    groups = sorted({(float(r['threshold']), float(r['budget_fraction'])) for r in rows})
    audit = []
    images = []
    for kind in ['log_error', 'ratio']:
        values_all = []
        for r in rows:
            for method, _, _ in METHODS:
                ratio = float(r[method]) / float(r['p_reference'])
                if kind == 'ratio':
                    values_all.append(ratio)
                else:
                    values_all.append(math.log10(1 + abs(ratio - 1)))
        maximum = max(values_all, default=0)
        zoom = .4 if kind == 'log_error' else 2
        full = max(zoom, maximum * 1.08 + .01)
        fig, axes = plt.subplots(len(groups), 4, figsize=(34, 4.7 * len(groups) + 2.6))
        for row_index, (threshold, fraction) in enumerate(groups):
            selected = [r for r in rows if (float(r['threshold']), float(r['budget_fraction'])) == (threshold, fraction)]
            cases = sorted({r['configuration'] for r in selected},
                           key=lambda case: next(float(r['lambda_final']) for r in selected if r['configuration'] == case))
            labels = []
            for position, case in enumerate(cases):
                runs = [r for r in selected if r['configuration'] == case]
                zeros = []
                outside = []
                for method_index, (method, _, color) in enumerate(METHODS):
                    ratios = np.array([float(r[method]) / float(r['p_reference']) for r in runs])
                    zeros.append(int((ratios == 0).sum()))
                    transformed = ratios if kind == 'ratio' else np.log10(1 + abs(ratios - 1))
                    outside.append(int((transformed > zoom).sum()))
                    for column in range(4):
                        positive_only = column % 2 == 0
                        values = transformed[ratios > 0] if positive_only else transformed
                        audit.append(dict(kind=kind, configuration=case, threshold=threshold,
                                          method=method, runs=len(runs), zero_count=zeros[-1],
                                          selection='positive_only' if positive_only else 'all_runs',
                                          view='zoom' if column < 2 else 'full',
                                          plotted_count=len(values), outside_zoom=outside[-1]))
                        ax = axes[row_index, column]
                        location = position + (method_index - 1) * .24
                        short_method = ['SMC', 'SNIS', 'MC'][method_index]
                        ax.text(position, 1.015 + .052 * (2 - method_index),
                                f'{short_method}: {zeros[-1]}/{len(runs)}',
                                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                                color=color, fontsize=8, fontweight='bold', clip_on=False)
                        if len(values):
                            ax.boxplot(values, positions=[location], widths=.20, whis=1.5,
                                       patch_artist=True, showmeans=True, showfliers=True,
                                       boxprops=dict(facecolor=color, edgecolor=color, alpha=.7),
                                       whiskerprops=dict(color=color), capprops=dict(color=color),
                                       medianprops=dict(color='black', linewidth=1.5),
                                       meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='white', markersize=4),
                                       flierprops=dict(marker='.', markerfacecolor=color, markeredgecolor=color, markersize=3))
                        else:
                            ax.text(location, .04, '0 positive', rotation=90, ha='center', va='bottom', fontsize=8)
                actual_fraction = np.mean([int(r['crude_n']) * float(r['p_reference']) for r in runs])
                n_min, n_max = min(int(r['N']) for r in runs), max(int(r['N']) for r in runs)
                n_label = str(n_min) if n_min == n_max else f'{n_min}–{n_max}'
                labels.append(f"λ={float(runs[0]['lambda_final']):g} · MH={runs[0]['moves_per_level']}\n"
                              f'N={n_label}\nMC≈{100 * actual_fraction:.0f}%\n'
                              f"fuori zoom {'/'.join(map(str, outside))}")
            for column in range(4):
                ax = axes[row_index, column]
                ax.set_xticks(range(len(cases)), labels, fontsize=8.5)
                ax.set_xlim(-.6, len(cases) - .4)
                ax.set_ylim(0, zoom if column < 2 else full)
                if kind == 'ratio' and column >= 2:
                    ax.set_yscale('symlog', linthresh=1)
                reference = math.log10(2) if kind == 'log_error' else 1
                ax.axhline(reference, color='#9e3333', linestyle='--', linewidth=1)
                ax.grid(axis='y', alpha=.2)
                population = 'SOLO STIME POSITIVE · zeri esclusi' if column % 2 == 0 else 'DISTRIBUZIONE COMPLETA · zeri inclusi'
                view = 'zoom' if column < 2 else 'tutte le code'
                ax.set_title(f'Score ≥ {threshold:g} · budget target {fraction:g} × primo evento MC\n'
                             + population + ' · ' + view, fontsize=10.5, pad=57)
                ax.set_ylabel(r'$\log_{10}(1+|\widehat p_A/\widehat p_{ref}-1|)$'
                              if kind == 'log_error' else r'$\widehat p_A/\widehat p_{ref}$')
        title = 'Errore logaritmico' if kind == 'log_error' else 'Rapporto stima/riferimento'
        title += ' · stime positive e tutte le run affiancate'
        fig.suptitle(title, fontsize=17, y=.99)
        fig.text(.5, .967, 'Etichette colorate sopra i box = NUMERO DI RUN A ZERO / RUN TOTALI. '
                 'La media e la mediana si riferiscono alle run presenti nel rispettivo pannello.', ha='center', fontsize=12)
        fig.legend(handles=[Patch(facecolor=c, label=label) for _, label, c in METHODS],
                   loc='lower center', ncol=3, bbox_to_anchor=(.5, .045))
        reference_text = ('Linea tratteggiata: errore relativo 100%, anche per stime zero.'
                          if kind == 'log_error' else 'Linea tratteggiata: stima uguale al riferimento (rapporto 1).')
        fig.text(.5, .012,
                 'Linea nera: mediana. Rombo: media della quantità sull’asse. Box: 25°–75°. Baffi: 1,5 IQR.\n'
                 + reference_text + ' Tutti gli outlier nei pannelli con le code.\n'
                 'Fuori zoom: SMC / SNIS / MC. Confronto a pari wall clock. Frazione del costo MC per il primo evento = n_MC × p_ref.\n'
                 '12 repliche per configurazione. Riferimenti indipendenti stimati; score 0,999 basato su 26/200.000 eventi.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(top=.91, bottom=.14, hspace=.85, wspace=.24)
        figure_dir = root / 'plots'
        figure_dir.mkdir(exist_ok=True)
        path = figure_dir / f'gradual_budget_{kind}_positive_and_all.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        images.append(path)
    (root / 'gradual_budget_positive_and_all_plot_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    report = root / 'REPORT.html'
    content = report.read_text(encoding='utf-8').split('<!-- gradual-budget-figures -->')[0]
    content += '<!-- gradual-budget-figures -->'
    for path in images:
        content += f'<p><a href="{path.relative_to(root).as_posix()}"><img src="{path.relative_to(root).as_posix()}" style="width:100%"></a></p>'
    report.write_text(content, encoding='utf-8')
    print(json.dumps({'figures': [str(p) for p in images], 'groups_audited': len(audit)}, indent=2))


if __name__ == '__main__':
    main()
