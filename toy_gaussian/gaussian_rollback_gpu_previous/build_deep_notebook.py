"""Embed all source in a Colab notebook: no imports from workspace files."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
model = (HERE / "deep_model.py").read_text(encoding="utf-8")
study = (HERE / "deep_study.py").read_text(encoding="utf-8")
start = study.index("from deep_model import (")
end = study.index("warmup)\n", start)+len("warmup)\n")
study = study[:start]+study[end:]
study = study[:study.index('if __name__ == "__main__":')]

notebook = nbf.v4.new_notebook()
notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "colab": {"name": "RollBack_SMC_deep_tilts_Colab.ipynb", "provenance": []}}
notebook.cells = [
    nbf.v4.new_markdown_cell(r"""# RollBack SMC: una run profonda, stime a ogni livello

Selezionare un runtime **GPU** in Colab e usare **Run all**. Il notebook è autonomo: codice e dipendenze sono nelle celle, senza import di file locali. Si conserva la notazione del LaTeX allegato. Una particella è sempre la traiettoria completa $\tau=(x_T,\ldots,x_0)$ sotto $P$, con $x_T=0$, $T=16$ e $Q_t(x_{t-1}\mid x_t)=\mathcal N(x_t,1/T)$. Per ciascuno dei due eventi $A=\{x_0\ge a\}$, $a=\Phi_{\rm normale}^{-1}(1-p_A)$ viene dal quantile **teorico** della normale standard, con $p_A\in\{10^{-4},10^{-5}\}$ esatto.

La **singola run principale** usa $N=128$, $R=20$, $\beta_k=0.25k$ e $\lambda=\beta_R=5$. Il kernel $M_k$ sceglie $d$ uniformemente in $\{1,\ldots,16\}$, conserva il prefisso, rigenera il suffisso con le transizioni di $P$ e applica MH con probabilità $1\wedge e^{\beta_k[\Phi(\tau')-\Phi(\tau)]}$. Si esegue una mutazione per livello. Dopo $G_k(\tau_{k-1})$, il resampling systematic avviene solo se ESS/N$<0.5$ **e** $k<R$, quindi mai dopo l'ultima pesatura. La ricorrenza di $\widehat Z_k$ usa il rapporto dei pesi normalizzati prima della mutazione a ogni livello, incluso quello senza resampling.

Dopo la mutazione di **ogni** livello calcoliamo separatamente, senza medie tra livelli,
$$\widehat p_{A,k}^{SMC}=\widehat Z_k\sum_n\overline W_k^{(n)}e^{-\beta_k\Phi(\tau_k^{(n)})}\mathbf1_A(\tau_k^{(n)}),$$
$$\widehat p_{A,k}^{SNIS}=\frac{\sum_n\overline W_k^{(n)}e^{-\beta_k\Phi(\tau_k^{(n)})}\mathbf1_A(\tau_k^{(n)})}{\sum_n\overline W_k^{(n)}e^{-\beta_k\Phi(\tau_k^{(n)})}}.$$
Il fattore inverso usa il **$\beta_k$ corrente**. Per confrontare il livello $k$ con una run che finirebbe lì, cloniamo la popolazione **dopo la pesatura e prima dell'eventuale resampling**, applichiamo $M_k$ sulla copia senza resampling finale e ne calcoliamo gli stessi stimatori. La run principale prosegue secondo la politica ESS; il costo delle copie è registrato separatamente e non incluso nel costo cumulativo principale.

Per ciascun livello sono registrati entrambe le stime, numero di particelle in $A$, ESS/N e massimo peso di $\overline W_k^{(n)}$, ESS/N e massimo peso dopo la correzione inversa su **tutte** le particelle, frazione di antenati iniziali distinti, tempo GPU e transizioni cumulative. Il tempo misura la stima completa con sincronizzazione CUDA prima e dopo ogni segmento; vengono riportati anche i normali generati dal codice GPU per il suffisso, che usa un buffer di lunghezza $T$ per particella.

Le repliche indipendenti dell'intera procedura stimano bias, varianza, RMSE e frequenza di zeri **a ciascun $\beta_k$ separatamente**. Il crude MC genera traiettorie complete da $P$ ed è confrontato con il costo cumulativo di ogni livello, sia in transizioni usate sia in tempo GPU. Per il confronto per transizioni riportiamo anche l'RMSE binomiale esatto $\sqrt{p_A(1-p_A)/n}$: per $10^{-5}$ un numero finito di repliche crude con pochi campioni può produrre molte stime nulle. Nessuna formula di varianza senza resampling è applicata alle run SMC con resampling."""),
    nbf.v4.new_code_cell("import importlib.util, subprocess, sys\nmissing = [x for x in ['numpy','scipy','pandas','matplotlib','torch'] if importlib.util.find_spec(x) is None]\nif missing: subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])\nprint('Dependencies ready; CUDA GPU required.')"),
    nbf.v4.new_code_cell(model),
    nbf.v4.new_code_cell(study),
    nbf.v4.new_code_cell("from IPython.display import display, Image\nOUT.mkdir(exist_ok=True)\nchecks = check_algorithm()\n(OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')\nprint('Tests passed:', checks)\nwarmup(20)"),
    nbf.v4.new_markdown_cell("## Prima: una sola run completa\n\nOgni riga è un livello della **medesima** run. Le tabelle della copia terminale mostrano cosa si otterrebbe fermandosi a quel livello senza resampling finale; le copie non alimentano i livelli successivi della run principale."),
    nbf.v4.new_code_cell("single = single_experiment()\npd.set_option('display.max_rows', 200)\ncolumns = ['level','beta','smc','snis','event_particles','smc_ess_fraction','smc_max_weight','reverse_ess_fraction','reverse_max_weight','distinct_root_fraction','main_cumulative_seconds','main_cumulative_transitions','terminal_copy_cumulative_seconds','terminal_copy_cumulative_transitions','extra_cumulative_seconds','extra_cumulative_transitions']\nfor probability in P_EVENTS:\n    for population in ('main','terminal_copy'):\n        print(f'p_A={probability:g}, {population}:')\n        display(single[(single.p_A==probability)&(single.population==population)][columns].reset_index(drop=True))\ndisplay(Image(filename=str(OUT/'single_run.png')))"),
    nbf.v4.new_markdown_cell("## Poi: repliche indipendenti e crude MC\n\nOgni riga di `summary.csv` si riferisce a uno specifico $\beta_k$, evento, popolazione e stimatore. Le stime **non** vengono mediate fra livelli. `runs.csv` conserva i seed e tutte le osservazioni delle run. `crude_budgets.csv` esplicita i campioni crude assegnati a ogni livello e mostra quanto tempo GPU abbiano effettivamente impiegato nella calibrazione."),
    nbf.v4.new_code_cell("summary, crude_summary, diagnostics, budgets = replicated_experiment(checks)\nprint('Per-level SMC summary:')\ndisplay(summary)\nprint('Per-level crude MC summary:')\ndisplay(crude_summary)\nprint('Transition and GPU-time budgets:')\ndisplay(budgets)\ndisplay(Image(filename=str(OUT/'rmse_by_level.png')))\ndisplay(Image(filename=str(OUT/'diagnostics_by_level.png')))"),
    nbf.v4.new_code_cell("print('Main-run final and selected intermediate levels:')\ndisplay(summary[(summary.population=='main') & summary.level.isin([1,5,10,15,20])][['p_A','level','beta','estimator','bias','variance','rmse','rmse_se','zero_fraction','median_event_particles','median_smc_ess_fraction','median_reverse_ess_fraction','median_root_fraction','median_main_cumulative_ms','median_main_transitions']])\nprint('Theoretical truth:', dict(zip(P_EVENTS,THRESHOLDS)))\nprint('Exact binomial crude MC benchmarks and empirical estimates are both in crude_summary.csv.')\nprint('No estimates from different beta levels were averaged.')"),
    nbf.v4.new_markdown_cell("""## Lettura dei risultati

Nel grafico delle stime della singola run, gli zeri sono visibili grazie alla scala simmetrica logaritmica. La copia terminale e la popolazione principale condividono la storia fino alla pesatura del livello; poi applicano mutazioni indipendenti, e la copia salta il resampling che una run terminata lì non avrebbe. La differenza a un livello non va interpretata come effetto causale isolato del resampling, perché cambia anche la mutazione stocastica.

Confrontare il RMSE a ogni $\beta_k$ con l'RMSE crude allo stesso **costo cumulativo**. Il budget in transizioni usa il numero di transizioni gaussiane effettivamente impiegate nel suffisso, mentre il benchmark GPU usa il tempo misurato dell'intera stima. Il codice CUDA disegna per praticità un buffer di $T$ normali a ogni proposta: il numero di variabili generate è quindi registrato separatamente dalle transizioni usate. Un ESS terminale alto può seguire un reset dei pesi e coesistere con pochi antenati; un ESS corretto alto non assicura che l'evento raro compaia nella popolazione. Gli intervalli RMSE delle run SMC riflettono l'incertezza Monte Carlo fra run complete, senza usare la CLT della versione senza resampling.""")
]
target = HERE / "RollBack_SMC_deep_tilts_Colab.ipynb"
nbf.validate(notebook)
nbf.write(notebook, target)
print(target)
