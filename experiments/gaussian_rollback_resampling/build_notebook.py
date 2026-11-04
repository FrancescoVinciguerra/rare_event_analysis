"""Author the standalone notebook. This builder is NOT required by Colab."""
from pathlib import Path
import re
import nbformat

HERE = Path(__file__).resolve().parent


def source_without_cli(name):
    return (HERE / name).read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]


def main():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md("""# RollBack SMC gaussiano: ESS/systematic e baseline senza resampling

**Notebook autocontenuto per Google Colab.** Carica questo `.ipynb` e scegli **Runtime → Run all**. Non servono altri file, Drive, un language model o GPU. Tutte le funzioni e i test sono incorporati qui.

Il confronto predefinito usa **2.000 run indipendenti per configurazione**, N=2048, dieci configurazioni SMC, con e senza resampling. La durata dipende dalla CPU (alcuni minuti). Le celle salvano dati e grafici nella directory relativa `rollback_smc_results` e mostrano le tabelle. Ridurre RUNS serve solo a una prova rapida, non riproduce l'esperimento consegnato.

Le formule IID della baseline non sono usate come varianza teorica del nuovo algoritmo. La teoria sotto distingue risultati esatti a N finito, implicazioni condizionali di una CLT e inferenza da run complete indipendenti."""),
        code("""# Install only missing runtime dependencies; Colab normally includes all four.
import importlib.util
import subprocess
import sys
required = {"numpy": "numpy>=1.26", "scipy": "scipy>=1.11", "pandas": "pandas>=2.0", "matplotlib": "matplotlib>=3.8"}
missing = [requirement for package, requirement in required.items() if importlib.util.find_spec(package) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, FileLink
print({"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__})
""")]
    theory = (HERE / "TEORIA.md").read_text(encoding="utf-8")
    for section in re.split(r"(?=\n## )", theory):
        if section.strip():
            cells.append(md(section.strip()))
    cells += [md("## Funzioni complete: modello, RollBack–MH, resampling e stimatori\n\nIn `run_smc`, i commenti **PAPER STEP 1/2/3** individuano pesatura, soglia ESS con esclusione finale, e mutazione. La riga `log_z += log_c` è esterna al ramo di resampling e viene eseguita sempre."),
              code(source_without_cli("smc_resampling.py")),
              md("## Test mirati\n\nI test includono entrambi i rami ESS, una popolazione concentrata che forza il resampling, e il divieto finale anche sotto soglia. Il test systematic integra esattamente i conteggi su tutti gli intervalli dell'offset. Le estrazioni tilted dirette compaiono solo nel test di invarianza."),
              code(source_without_cli("checks.py")),
              code("passed_checks = run_checks()\nassert len(passed_checks) == 12\nprint(f'Completati {len(passed_checks)} controlli.')"),
              md("## Parametri riproducibili\n\nT=16, R=8, una mutazione per livello, schedule lineare. Profondità uniforme su {1,...,D}; soglia ESS/N < 0.5; systematic in ordine input; mai resampling finale. Seed separati per crude MC (stream 0), baseline (stream 1) e algoritmo con resampling (stream 4). La baseline ricostruisce gli stessi contributi e seed del lavoro precedente, senza leggere i suoi file."),
              code("N = 2048\nRUNS = 2000\nROOT_SEED = 20260929\nOUTPUT_DIR = 'rollback_smc_results'"),
              md("## Una run leggibile: ESS, resampling e normalizzatore a ogni livello"),
              code("""demo_config = Config(tilt=3, max_depth=16)
demo = run_smc(np.random.default_rng(71008), 256, demo_config)
assert any(row['resampled'] for row in demo.levels)
assert not demo.levels[-1]['resampled']
display(pd.DataFrame(demo.levels)[['level', 'beta', 'ess_before', 'ess_after', 'resampled', 'log_c', 'z_hat', 'transitions_cumulative']])
print('Stime della run dimostrativa:', demo.estimates(demo_config.tilt))
"""),
              md("## Esperimenti e report: funzioni incorporate\n\nLe varianze sono calcolate sulle stime finali di run indipendenti dell'intera popolazione. Non si usa l'ESS come sostituto della varianza, né si trattano i discendenti come IID."),
              code(source_without_cli("experiment.py")),
              code("frame, summary, manifest = run_comparison(n=N, runs=RUNS, root_seed=ROOT_SEED, output_dir=OUTPUT_DIR)"),
              md("## Tabelle: accuratezza e costo\n\nBias rispetto a p=0.001; errore Monte Carlo del bias s/sqrt(B). I due stimatori della stessa popolazione condividono il costo, che non va sommato due volte. Tutte le run e diagnostiche sono salvate nel CSV."),
              code("""with pd.option_context('display.max_rows', 60, 'display.float_format', '{:.6g}'.format):
    display(summary[['lambda', 'D', 'policy', 'estimator', 'bias', 'bias_mc_se', 'variance', 'rmse', 'zero_frequency']])
    display(summary.query("estimator == 'ordinary'")[['lambda', 'D', 'policy', 'median_seconds', 'mean_transitions', 'resampling_run_fraction', 'mean_resampling_count']])
"""),
              md("## Grafici\n\nLa linea crude MC è la sua varianza analitica. Le altre linee sono varianze empiriche delle run complete; le barre sono circa 1.96 errori Monte Carlo della varianza stimata. Non sono una verifica di una CLT systematic dimostrata."),
              code("create_figures(frame, summary, manifest, output_dir=OUTPUT_DIR)"),
              md("## Identità del normalizzatore e sintesi dei risultati"),
              code("""comparison = write_report(frame, summary, manifest, output_dir=OUTPUT_DIR)
display(comparison)
display(summary.query("policy == 'ess_systematic' and estimator == 'ordinary'")[['lambda', 'D', 'mean_z_hat', 'z_hat_mc_se', 'true_z', 'mean_reverse_mass', 'reverse_mass_mc_se']])
best = summary.query("policy == 'ess_systematic' and `lambda` > 0").sort_values('rmse').iloc[0]
display(Markdown(f"**Minimo RMSE osservato nella griglia:** lambda={best['lambda']:g}, D={best['D']:g}, {best['estimator']}, RMSE={best['rmse']:.6g}. MSE × transizioni / crude MC = {best['mse_times_cost_relative_mc']:.3f}. La selezione è descrittiva, non una prova di ottimalità."))
display(Markdown(f"Resampling attivato in **{manifest['active_resampling_runs']} run**; escluso in tutte le run al livello finale. Il file RISULTATI.md contiene tutte le tabelle e la discussione dei limiti teorici."))
"""),
              md("## Esportazione\n\nL'archivio contiene CSV completi, manifest, report e grafici. In Colab si può scaricarlo dal pannello Files. Il notebook resta autonomo anche senza questo archivio: Run all rigenera ogni risultato."),
              code("""import zipfile
from pathlib import Path
output_path = Path(OUTPUT_DIR)
(output_path / 'test_results.json').write_text(json.dumps({'passed': passed_checks, 'count': len(passed_checks)}, indent=2), encoding='utf-8')
archive_path = Path('rollback_smc_results.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_path.iterdir()):
        if path.is_file():
            archive.write(path, arcname=path.name)
display(FileLink(str(archive_path)))
print('Run all completato: test, esperimenti, tabelle, grafici e archivio generati.')
""")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}})
    sources = "\n".join(c.source for c in cells)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", sources)
    assert "from smc_resampling import" not in sources and "from model import" not in sources
    nbformat.validate(notebook)
    destination = HERE / "RollBack_SMC_Colab.ipynb"
    nbformat.write(notebook, destination)
    print(f"Created standalone notebook: {len(cells)} cells")


if __name__ == "__main__":
    main()
