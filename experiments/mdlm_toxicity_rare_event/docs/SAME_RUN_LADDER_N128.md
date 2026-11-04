# Stessa run SMC: N=128, tilt fino a 8

Autorizzazione: utente, 4 novembre 2026, «ok vai con 3 gpu».

12 seed indipendenti: 2000000000 + 1000000*i, i=0,...,11.
Ogni seed attraversa 0 -> 0.5 -> ... -> 8, 16 livelli, 4 MH per livello.
Si valutano soltanto q>=0.99 e q>=0.999 sulla stessa popolazione.
Decoder, classificatore, score Phi=log(q), rollback uniforme {4,8,16,24,32},
ESS<0.5 e systematic invariati. Pesi una sola volta per livello; Z aggiornato
sempre; nessun resampling al livello finale 8. Non ci sono copie terminali:
sono le popolazioni effettive della medesima run in prosecuzione.

Si conservano 65 popolazioni per seed (iniziale + 64 mosse MH), con score,
pesi normalizzati, Z, numero di eventi, ESS, tempi e transizioni cumulativi.
Almeno una particella nell'evento identifica una popolazione con stima positiva.
La run continua fino a 8. Nessuna media o aggregazione tra tilt o mosse.
I boxplot principali confrontano fra seed le 16 popolazioni dopo la quarta MH,
con mediana, media, zeri espliciti, positivi e tutte le run affiancati, zoom e code.

Baseline crude indipendenti, misurate sulla medesima GPU, a ogni fine livello
e al primo evento. Il numero di traiettorie e' fissato prima dei campioni MC,
tramite calibrazione temporale indipendente; durata effettiva e scostamento
sono salvati. Gli altri checkpoint MH non ricevono un confronto MC inventato.
Ogni baseline ha il suo tempo SMC cumulativo, non una frazione del tempo finale.
Tutte le repliche e gli zeri restano nei dati. La distribuzione dei soli positivi
non viene presentata come distribuzione incondizionata degli stimatori.

## Risorse e costo

Un job, 1 nodo, 3 GPU Quadro RTX6000, 3 task, 4 CPU/task, 60 GB, limite 12 ore.
Ogni task vede una GPU e svolge 4 seed; ogni singola stima SMC o MC usa
1 GPU e 4 CPU. OMP/MKL/OpenBLAS/NumExpr sono limitati a 4 thread.
Warm-up e calibrazione esclusi dai tempi delle stime, ma registrati.
La durata totale del job include tutte le baseline ai livelli e non coincide
con il costo di una singola stima. Limite massimo di allocazione: 36 GPU-ore.
Salvataggio dopo ogni popolazione SMC completa e ogni baseline MC; nessun rilancio automatico.

## Azione remota autorizzata

Origine: C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main.
Destinazione: avon.scrtp.warwick.ac.uk, directory
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.
Tre moduli modificati sono conservati prima dell'aggiornamento in un archivio;
nessun risultato precedente viene sostituito. Trasferimento con SHA256.
Nessuna installazione e nessun calcolo scientifico sul login node.

Comando di invio integrale, dopo controllo dei file e dry run:

```bash
sbatch --parsable --no-requeue --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=60G --time=12:00:00 --job-name=mdlm-same-run-ladder --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/scripts/slurm_avon_same_run_ladder.sh
```

Comando scientifico su ciascuna GPU assegnata, WORKER_ID=0,1,2:

```bash
python -m diffusion_toxicity_smc_experiment.accuracy_budget run --study /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_same_run_ladder.yaml --output-root /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/current --max-seconds 41400 --workers 3 --worker-id WORKER_ID
```

Output unico: artifacts/current/camera_ready_same_run_ladder,
con checkpoint_runs.csv (tutte le popolazioni), first_event_runs.csv,
runs.csv (terminali), summary.csv, REPORT.html, manifest.json e checkpoint grezzi.
Riferimenti indipendenti finiti: 444/200000 e 26/200000, non probabilita' esatte.
