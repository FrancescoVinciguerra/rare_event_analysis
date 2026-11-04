# Gaussian full-trajectory RollBack SMC, no resampling

La derivazione completa è in [DERIVAZIONE.md](DERIVAZIONE.md). I risultati dell'esecuzione sono descritti in [RISULTATI.md](RISULTATI.md).

## Esecuzione

Dalla radice del repository, con l'ambiente Python esistente:

```powershell
.venv/Scripts/python.exe -m pytest experiments/gaussian_rollback_smc/test_model.py -q
.venv/Scripts/python.exe experiments/gaussian_rollback_smc/run_experiment.py
```

Il controllo aggiuntivo del caso lambda=0 (4000 repliche con stream indipendente 3) si riproduce con:

```powershell
.venv/Scripts/python.exe experiments/gaussian_rollback_smc/check_zero_tilt.py
.venv/Scripts/python.exe experiments/gaussian_rollback_smc/report.py
```

Oppure usare qualsiasi Python con numpy, scipy, matplotlib e pytest installati:

```text
python -m pytest experiments/gaussian_rollback_smc/test_model.py -q
python experiments/gaussian_rollback_smc/run_experiment.py
```

Parametri predefiniti: T=16, R=8, una mutazione per livello, schedule lineare, N=2048, 2000 repliche per configurazione, 128 blocchi pilota indipendenti da 16384 storie ciascuno. Il pilot non contribuisce alle stime di probabilità del confronto. Si esplorano lambda=1,2,3 e profondità uniforme d in {1,...,D}, D=1,4,16, oltre al controllo lambda=0, D=16. Crude MC genera anch'esso traiettorie originali complete. Gli stimatori ordinario e SNIS condividono le stesse storie entro una replica; le repliche, le configurazioni, crude MC e pilot hanno stream separati.

Seed principale: 20260929. Ogni generatore NumPy usa `SeedSequence([seed, stream, config_id, replicate])`: stream 0 per crude MC, 1 per SMC, 2 per il pilot; il campo replicate del pilot è l'indice del blocco. Il confronto a numero di particelle fissato non è un confronto a costo fissato.

Esempio di esecuzione piccola in una directory distinta, senza sovrascrivere i risultati principali:

```text
python experiments/gaussian_rollback_smc/run_experiment.py --particles 1024 --replicates 50 --pilot-blocks 4 --pilot-size 4096 --output experiments/gaussian_rollback_smc/small_run
```

## File

- `model.py`: traiettorie complete, proposal e kernel MH, pesi in scala logaritmica, due stimatori.
- `test_model.py`: test deterministici delle formule e controlli statistici a seed fisso con tolleranza di sette errori standard Monte Carlo.
- `run_experiment.py`: repliche, pilot indipendente, metriche e tre grafici.
- `report.py`: genera `RISULTATI.md` dai dati salvati, senza nuove simulazioni.
- `check_zero_tilt.py`: controllo indipendente del caso lambda=0, salvato separatamente in `results/zero_tilt_followup.*`.
- `results/manifest.json`: parametri, versioni software, piattaforma, seed e durata totale.
- `results/replicates.csv`: tutte le stime, seed, tempi e transizioni per replica.
- `results/summary.csv`: bias, errore Monte Carlo del bias, varianza, errore Monte Carlo della varianza, RMSE, zeri, costo, accettazione e confronto con il pilot.
- `results/pilot_blocks.csv`: momenti per blocco indipendente, con seed.
- `results/pilot_diagnostics.json`: momenti complessivi, errori standard e diagnostica della concentrazione dei secondi momenti.
- `results/*.png`: grafici esportabili, nessun algoritmo ideale usato al posto del RollBack.

Il costo per replica SMC è esattamente NT più la somma delle profondità di tutte le proposal, incluse quelle rifiutate. Il tempo include generazione, mutazione e valutazione dei due stimatori; il tempo del pilot è separato. Entrambi gli stimatori hanno quasi tutto il costo in comune. I tempi dipendono dalla macchina e dal carico e non sono benchmark universali.

`variance_cost_vs_crude` confronta varianza empirica moltiplicata per transizioni con il valore teorico p(1-p)T del crude MC. È un indicatore di efficienza in termini di transizioni, non il risultato di una seconda simulazione a budget abbinato, né una correzione per il bias di SNIS.

Un pilot è etichettato stabile solo se l'errore standard relativo del secondo momento è <15%, la quota del massimo contributo <2%, e la differenza relativa tra le due metà <40%. Sono soglie diagnostiche, non una garanzia matematica di aver esplorato le code. Le barre nei grafici sono approssimazioni Monte Carlo al 95%, non intervalli simultanei. Per SNIS il confronto con sigma²/N resta asintotico.
