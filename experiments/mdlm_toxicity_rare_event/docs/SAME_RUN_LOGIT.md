# Same-run ladder: Phi selezionabile, confronto log(q) / logit(q)

Richiesta: ripetere le 12 ladder completate nel job 2079904 cambiando soltanto
Phi. Nessuna nuova generazione crude MC, nemmeno per warm-up o calibrazione.
I risultati e il codice precedenti restano disponibili.

## Definizione e selezione di Phi

Configurazione generale del sampler: `smc.phi: log_probability` (default,
Phi=log(q), comportamento precedente) oppure `smc.phi: logit`
(Phi=log(q/(1-q))). Nelle configurazioni di studio la chiave `phi` viene
propagata a `smc.phi` in ogni configurazione risolta.

La trasformazione e' implementata in `potentials.py`. Il logit si calcola dai
logits originali: per due classi, logit_toxic - logit_non_toxic. Non si applicano
epsilon o clipping alle probabilita'. Si conserva separatamente q per gli
eventi. Il classificatore, la definizione degli eventi e P non cambiano.

Tutti i fattori usano la stessa Phi: G_k=exp((lambda_k-lambda_{k-1})Phi),
accettazione MH min(1,exp(lambda_k(Phi'-Phi))), correzione exp(-lambda_k Phi).
Z_hat viene aggiornato prima della mutazione a ogni livello. Le mutazioni
non cambiano i pesi. L'osservatore recupera q con sigmoid(Phi) per il logit,
con exp(Phi) per lo score precedente; vicino a una soglia ricontrolla la
probabilita' originale del classificatore, includendone il tempo.

## Disegno invariato

- N=128; 12 seed distinti: 2000000000 + 1000000*i, i=0,...,11.
- Gli stessi seed del precedente studio consentono il confronto appaiato;
  dopo la prima selezione diversa, le traiettorie delle due varianti divergono.
- Una singola ladder per seed: lambda=0.5,1,...,8, 16 livelli, 4 MH/livello.
- Una pesatura/livello, systematic se ESS/N<0.5, nessun resampling finale.
- Rollback invariato: profondita' uniformi {4,8,16,24,32}, piu' il completamento
  categorico terminale. Processo nativo completo di 33 transizioni.
- Eventi q>=0.99 e q>=0.999 dalla stessa popolazione.
- Registrazione di tutte le 65 popolazioni: iniziale + dopo ogni mossa MH.
- Entrambi gli stimatori dopo ogni MH con il lambda corrente; nessuna media
  fra livelli, nessun arresto al primo evento e nessuna nuova copia terminale.
- Calibrazione temporale precedente sulla stessa GPU fisica e 4 CPU.
- Tre GPU parallele per i seed; ciascuna stima usa una GPU e 4 CPU.

I valori di lambda delle due Phi non hanno la stessa intensita'. Questa e'
una verifica controllata della richiesta a schedule invariata, non una promessa
di efficienza del logit fino a 8. Si salvano anche acceptance, ESS prima/dopo
resampling, ESS e massimo peso dopo correzione, genealogia e Z_hat.

## Riuso del crude MC e report

Pool precedente completo: camera_ready_gradual_budget, 192 baseline, 354420
traiettorie IID. I 200000 campioni del riferimento non sono usati per MC.
Stesso ordine dei file e stesso blocco disgiunto per seed del report precedente.
Per ogni checkpoint: n_MC=floor(tempo SMC cumulativo / costo MC calibrato).
La lunghezza del prefisso puo' cambiare con il tempo della nuova run; il costo
MC e' un equivalente calibrato, non una misura MC ripetuta.

Output: workers/*/populations, checkpoint_runs.csv, first_event_runs.csv,
runs.csv, summary.csv, timing_steps.csv, saved_MC_blocks.json,
comparison_policy.json, REPORT.html e due figure con zoom/code, positivi/tutte
le run, mediana, media e conteggi degli zeri. Nessuna media fra tilt.

## Risorse, origine, destinazione e comandi

Origine: C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main.
Destinazione: stuhvh@avon.scrtp.warwick.ac.uk tramite Hammer.
Progetto remoto: /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.
Nuova cartella risultati: experiments/mdlm_toxicity_rare_event/artifacts/current/camera_ready_same_run_logit.
Il sorgente congelato viene trasferito nella sottocartella `code` del nuovo
risultato. Non si sovrascrivono sorgenti o risultati sul cluster.
Il trasferimento avviene sul canale SSH gia' autenticato: payload tar.gz in
base64, SHA256 verificato prima dell'estrazione. Elenco e hash dei file sono
registrati nel bundle e nel manifest di deployment.

Risorse: partizione gpu, account stats, qos normal, nodo gpu007,
3 quadro_rtx_6000, 3 task da 4 CPU, 60 GB, massimo 2 ore.
Durata indicativa: circa un'ora, da verificare con le nuove run.

```bash
export PROJECT_DIR=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code
export OUTPUT_ROOT="$PROJECT_DIR/experiments/mdlm_toxicity_rare_event/artifacts/current"
export EXPERIMENT_DIR="$OUTPUT_ROOT/camera_ready_same_run_logit/code"
export CALIBRATION_ROOT="$OUTPUT_ROOT/camera_ready_same_run_ladder"
export LOG_DIR="$PROJECT_DIR/experiments/mdlm_toxicity_rare_event/artifacts/logs"
export VENV_DIR=/gpfs/home/stats/stuhvh/venvs/toxicity-smc
export HF_HOME=/gpfs/home/stats/stuhvh/hf-cache
export AVON_MODULES='GCCcore/13.2.0 Python/3.11.5'
sbatch --parsable --no-requeue --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --nodelist=gpu007 --ntasks=3 --cpus-per-task=4 --mem=60G --time=02:00:00 --job-name=mdlm-ladder-logit --chdir="$PROJECT_DIR" --export=ALL --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "$EXPERIMENT_DIR/scripts/slurm_avon_same_run_phi.sh"
```

Comando scientifico di ciascun task (worker=0,1,2):

```bash
python -m diffusion_toxicity_smc_experiment.same_run_reuse run-fresh --study "$EXPERIMENT_DIR/configs/mdlm_owt_same_run_logit.yaml" --output-root "$OUTPUT_ROOT" --calibration-root "$CALIBRATION_ROOT" --max-seconds 6300 --workers 3 --worker-id "$worker"
```

Solo calcolo sotto SLURM; nessun modello o esperimento sul nodo login.
Ogni run completa e' salvata atomicamente e verificata prima delle successive.
La ripresa salta i seed gia' completati e verifica Phi, configurazione e hash
del codice. Il job genera automaticamente CSV, report e figure prima di creare
l'archivio scaricabile.

## Verifica iniziale e isolamento dei test

Il primo invio, 2081343, e' terminato FAILED dopo 45 secondi prima di
caricare modelli o produrre popolazioni. La configurazione pytest del vecchio
repository reinseriva nel path un pacchetto omonimo storico. Il job usa ora
esplicitamente `pytest_snapshot.ini` e la root dello snapshot; un test verifica
anche il percorso effettivo del package importato. I 18 controlli del job sono
passati localmente con le stesse variabili SLURM simulate. La copia del codice
del tentativo fallito viene conservata prima della correzione. Nessun risultato
precedente o campione scientifico e' cancellato.
