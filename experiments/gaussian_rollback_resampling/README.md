# RollBack SMC con resampling e notebook Colab autonomo

Il file principale da consegnare/caricare è **`RollBack_SMC_Colab.ipynb`**. Contiene dipendenze, teoria, tutte le funzioni, dodici test, parametri, 2.000 run per configurazione, baseline senza resampling, tabelle, grafici e generazione di un archivio dei risultati. In Google Colab: caricare il notebook e scegliere **Runtime → Run all**. Nessun file del repository, mount Drive, modello linguistico o GPU è necessario.

La cartella precedente `gaussian_rollback_smc` è conservata senza modifiche. Il nuovo notebook ricalcola la baseline con gli stessi parametri e seed della consegna precedente invece di leggerne i file. Gli esperimenti con resampling usano stream distinti.

## Corrispondenza col paper

- `paper/ai4good_rare_event_estimation.tex`, righe 500–502: otto livelli, uno sweep, systematic e soglia ESS/N < 0.5.
- Stesso file, righe 727–744: pesatura, eventuale resampling intermedio, MH a ogni livello, nessun resampling finale.
- `experiments/mdlm_toxicity_rare_event/src/diffusion_toxicity_smc_experiment/smc.py`, `resample_indices`: offset systematic unico, input non ordinati per stato e antenati nell'ordine del CDF.

Nel nuovo `smc_resampling.py`, la funzione `run_smc` contiene i commenti **PAPER STEP 1/2/3**. Le istruzioni decisive sono:

```python
log_c = float(logsumexp(unnormalized))
log_z += log_c
log_w = unnormalized - log_c
do_resample = bool(resampling and k < config.levels
                   and ess_before < config.ess_threshold * n)
```

`log_z` è aggiornato fuori dal ramo `if do_resample`; quel ramo azzera solo i log-pesi normalizzati a `-log(n)`, copia traiettorie complete e root IDs, e non modifica il normalizzatore. Segue la mutazione a beta_k. `SMCResult.estimates` implementa esattamente le due formule richieste.

Le profondità del toy restano uniformi in {1,...,D} con D=1,4,16, T=16. È una scelta conservata dalla baseline per isolare il resampling: non sono le profondità del decoder linguistico del paper.

## Teoria

`TEORIA.md`, incorporato anche nel notebook, deriva:

- ricorrenza di pesi e Z_hat a tutti i livelli;
- unbiasedness condizionata dei conteggi systematic e di Gamma_hat;
- E[Z_hat]=Z, E[p_ord]=p ed E[reverse_mass]=1 a N finito;
- varianza esatta via martingala, distinguendo il termine senza resampling e quello con covarianze systematic;
- identità esatta del rapporto SNIS e limiti di una sua linearizzazione;
- risultati per politiche diverse, che non vengono applicati al nuovo esperimento;
- inferenza corretta da run indipendenti dell'intero algoritmo.

Non viene affermata una CLT in N dimostrata per systematic non ordinato con soglia ESS fissa, né riutilizzata la varianza IID della baseline. La varianza campionaria tra run complete è invece riportata per ogni configurazione.

## Esecuzione locale facoltativa

Il notebook non richiede questi file separati. Per ispezionare/eseguire il codice localmente dalla radice del repository:

```powershell
.venv/Scripts/python.exe experiments/gaussian_rollback_resampling/checks.py
.venv/Scripts/python.exe experiments/gaussian_rollback_resampling/build_notebook.py
.venv/Scripts/python.exe experiments/gaussian_rollback_resampling/execute_notebook.py
```

Le utility di costruzione/esecuzione richiedono nbformat, nbclient e ipykernel; queste dipendenze sono strumenti di verifica locale, non requisiti aggiuntivi per Colab. `build_notebook.py` ricrea il notebook senza output, quindi va seguito da `execute_notebook.py` se si vuole conservare una copia eseguita.

`execute_notebook.py` apre un vero kernel Jupyter nella versione Python invocata ed esegue tutte le celle in ordine, in una **directory temporanea inizialmente vuota**, con errori non consentiti. Copia poi notebook eseguito, risultati e archivio nella cartella di consegna. `execution_verification.json` registra l'esito e il numero di celle eseguite.

## Parametri e dati

- N=2048, B=2000 run complete indipendenti, T=16, R=8, uno sweep per livello.
- Lambda=1,2,3; D=1,4,16; controllo aggiuntivo lambda=0,D=16.
- Seed principale 20260929; `SeedSequence([root_seed, stream, config_id, replicate])`.
- Stream 0 crude MC, stream 1 baseline originale, stream 4 ESS/systematic.
- `results/runs.csv`: tutte le stime e seed, ESS prima/dopo, livelli di resampling, percorso di log_c e log_Z, transizioni per livello, costo totale e genealogia.
- `results/summary.csv`: bias, errore Monte Carlo, varianza, RMSE, zeri, tempi, costo e attivazione del resampling.
- `results/manifest.json`: parametri, versioni e durata.
- `results/RISULTATI.md`: tutte le tabelle e figure.
- `rollback_smc_results.zip`: esportazione portabile dei risultati.

Il tempo misurato include la diagnostica e il calcolo dei due stimatori, che condividono il costo della popolazione. Il costo gaussiano conta NT più tutte le profondità rigenerate, anche nelle proposal rifiutate. Il confronto è a N uguale; il prodotto MSE per transizioni è un indicatore supplementare, non una simulazione a costo abbinato.
