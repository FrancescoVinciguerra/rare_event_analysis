"""Build a standalone Colab notebook by embedding the exact benchmark source."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
model = (HERE / "gpu_model.py").read_text(encoding="utf-8")
study = (HERE / "study.py").read_text(encoding="utf-8")
start = study.index("from gpu_model import (")
end = study.index("warmup)\n", start) + len("warmup)\n")
study = study[:start] + study[end:]
study = study[:study.index('if __name__ == "__main__":')]

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "colab": {"name": "RollBack_SMC_GPU_Colab.ipynb", "provenance": []}}
nb.cells = [
    nbf.v4.new_markdown_cell(r"""# RollBack SMC gaussiano su GPU: due correzioni, una popolazione

Runtime Colab: **GPU**. `Run all` installa eventuali dipendenze mancanti, esegue i test, il pilot e gli esperimenti, salva tabelle e grafici nella cartella relativa `rollback_gpu_results/`. Nessun modello linguistico e nessun file locale esterno.

Una particella è la traiettoria completa $\tau=(x_T,\ldots,x_0)$, con $x_T=0$, $T=16$ e $Q_t(x_{t-1}\mid x_t)=\mathcal N(x_t,1/T)$. La legge originale è $P$, quindi $x_0\sim\mathcal N(0,1)$. Usiamo $\Phi(\tau)=x_0$ e $A=\{x_0\ge a\}$, dove $a=\Phi_{\rm normale}^{-1}(0.999)$ è calcolato dalla normale teorica: $p_A=P(A)=0.001$ esatto.

Il pilot usa $N=64$; l'esperimento principale usa $N=128$. Per ognuno dei **quattro tilt terminali separati** $\lambda\in\{0.5,1,1.5,2\}$, la schedule interna è $R=4$, $\beta_k=k\lambda/4$. $\pi_k(\tau)\propto P(\tau)e^{\beta_k\Phi(\tau)}$ e $G_k(\tau)=e^{(\beta_k-\beta_{k-1})\Phi(\tau)}$. Il kernel $M_k$ sceglie $d$ uniformemente in $\{1,\ldots,16\}$, conserva il prefisso, rigenera il suffisso secondo $P$ e accetta con probabilità $1\wedge e^{\beta_k[\Phi(\tau')-\Phi(\tau)]}$. Vi è **una** mutazione per livello.

Prima della mutazione, i pesi sono aggiornati con $G_k(\tau_{k-1}^{(n)})$. In entrambi i rami,
$$\widehat Z_k=\widehat Z_{k-1}\sum_n\overline W_{k-1}^{(n)}G_k(\tau_{k-1}^{(n)}),\qquad \widehat Z_0=1.$$
Dopo la normalizzazione dei pesi si applica resampling systematic solo se $\mathrm{ESS}/N<0.5$ e $k<R$; in tal caso i pesi diventano $1/N$, altrimenti restano quelli normalizzati. **Nessun resampling finale.** La mutazione $M_k$ segue in entrambi i casi.

Dalla **stessa popolazione** terminale si calcolano
$$\widehat p_{A,R}^{SMC}=\widehat Z_R\sum_n\overline W_R^{(n)}e^{-\lambda\Phi(\tau_R^{(n)})}\mathbf1_A(\tau_R^{(n)}),$$
$$\widehat p_{A,R}^{SNIS}=\frac{\sum_n\overline W_R^{(n)}e^{-\lambda\Phi(\tau_R^{(n)})}\mathbf1_A(\tau_R^{(n)})}{\sum_n\overline W_R^{(n)}e^{-\lambda\Phi(\tau_R^{(n)})}}.$$
Il confronto principale usa crude MC da **traiettorie complete sotto $P$** sullo stesso dispositivo. Il campionamento diretto di $x_0\sim\mathcal N(0,1)$ è mostrato separatamente perché disponibile solo grazie a questo toy model. Le barre d'errore provengono da repliche indipendenti dell'intero algoritmo; la CLT della versione senza resampling **non** è usata per queste run.

Per ogni livello registriamo $\mathrm{ESS}/N$ e $\max_n\overline W_k^{(n)}$ **dopo $G_k$ ma prima dell'eventuale resampling**, nonché la decisione di resampling. Alla fine registriamo queste due quantità per i pesi SMC $\overline W_R^{(n)}$. Per la correzione inversa normalizziamo **su tutte le particelle** $\overline W_R^{(n)}e^{-\lambda\Phi(\tau_R^{(n)})}$, senza $\mathbf1_A$, e ricalcoliamo $\mathrm{ESS}/N$ e peso massimo. La frazione di antenati iniziali distinti è aggiornata quando il resampling copia intere traiettorie. Le distribuzioni fra run sono mostrate con quantili e grafici, insieme alle associazioni descrittive con bias e RMSE."""),
    nbf.v4.new_code_cell("import importlib.util, subprocess, sys\nmissing = [name for name in ['numpy', 'scipy', 'pandas', 'matplotlib', 'torch'] if importlib.util.find_spec(name) is None]\nif missing:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])\nprint('Dependencies available; CUDA runtime is required.')"),
    nbf.v4.new_code_cell(model),
    nbf.v4.new_code_cell(study),
    nbf.v4.new_code_cell("summary = main()"),
    nbf.v4.new_code_cell("from IPython.display import display, Image\nprint('Parameters and effective execution:')\nprint((OUT / 'parameters.json').read_text())\nprint('N=64 pilot:')\ndisplay(pd.read_csv(OUT / 'pilot_summary_N64.csv'))\nprint('N=128 main experiment and matched GPU time:')\ndisplay(summary[['method','lambda','n','reps','bias','bias_se','variance','rmse','rmse_se','zero_fraction','median_ms','median_transitions','median_gaussian_draws','resampling_run_fraction']])\ncomparison = pd.read_csv(OUT / 'comparison.csv')\ndisplay(comparison)\ndisplay(Image(filename=str(OUT / 'rmse_comparison.png')))"),
    nbf.v4.new_code_cell("diagnostics = pd.read_csv(OUT / 'diagnostic_quantiles.csv')\nassociation = pd.read_csv(OUT / 'diagnostic_error_association.csv')\nprint('Diagnostic distributions: min, 5%, 25%, median, 75%, 95%, max by tilt and level:')\ndisplay(diagnostics)\nprint('Conditional bias, RMSE, and zero estimates by diagnostic quantile (descriptive):')\ndisplay(association)\ndisplay(Image(filename=str(OUT / 'weight_degeneracy.png')))\ndisplay(Image(filename=str(OUT / 'degeneracy_error_association.png')))"),
    nbf.v4.new_code_cell("best = comparison.loc[comparison.snis_rmse.idxmin()]\nprint(f\"A N=128, miglior reverse-SNIS nella griglia: lambda={best['lambda']:.2f}, RMSE={best.snis_rmse:.6f}, contro crude MC N=128: {best.crude_N128_rmse:.6f}.\")\nprint(f\"A pari wall clock GPU misurato: reverse-SNIS {best.snis_rmse:.6f} in {best.smc_median_ms:.3f} ms; crude MC da traiettorie complete {best.crude_matched_rmse:.6f} con {int(best.crude_matched_n)} traiettorie in {best.crude_matched_median_ms:.3f} ms.\")\nprint(f\"Errore standard Monte Carlo della differenza di RMSE: {best.snis_minus_crude_matched_RMSE_se:.6f}; rapporto dei tempi crude/SMC: {best.time_ratio_crude_over_smc:.3f}.\")\nprint(f\"Rapporti dei tempi su tutta la griglia: [{comparison.time_ratio_crude_over_smc.min():.3f}, {comparison.time_ratio_crude_over_smc.max():.3f}].\")\nprint('I due stimatori SMC condividono la popolazione e il costo. Un ESS alto da solo non garantisce una stima accurata di un evento raro.')"),
    nbf.v4.new_markdown_cell("""## Interpretazione

Il grafico di sinistra confronta **N=128** traiettorie per metodo; quello di destra confronta il costo misurato di una run SMC con il numero di traiettorie crude che la GPU genera in un tempo simile. Lo stesso tempo SMC vale per $\\widehat p_{A,R}^{SMC}$ e $\\widehat p_{A,R}^{SNIS}$ perché provengono dalla medesima popolazione. `summary.csv` riporta bias, varianza empirica, RMSE, errore standard Monte Carlo dell'RMSE, stime nulle, tempi e transizioni. `runs.csv` contiene ogni replica e seed. Il grafico include intervalli approssimati al 95% di sola incertezza Monte Carlo dell'RMSE. `level_diagnostics.csv` e `diagnostic_quantiles.csv` rendono visibili i livelli e le distribuzioni; `diagnostic_error_association.csv` confronta la degenerazione con errori e stime nulle.

La stima $\\widehat Z_R$ e il denominatore SNIS sono casuali; nel caso con resampling le storie delle particelle sono dipendenti. Non attribuiamo loro la varianza derivata per storie indipendenti senza resampling. La concentrazione genealogica può persistere anche quando il resampling reimposta i pesi a $1/N$; viceversa un ESS alto non assicura che l'evento raro sia stato osservato. Le associazioni condizionali sono descrittive, non causali. I risultati a particelle uguali e a tempo uguale rispondono a domande distinte.""")
]
target = HERE / "RollBack_SMC_GPU_Colab.ipynb"
nbf.validate(nb)
nbf.write(nb, target)
print(target)
