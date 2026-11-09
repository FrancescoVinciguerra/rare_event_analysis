# Ripresa SMC con riuso del Monte Carlo esistente

L'utente ha richiesto di riutilizzare i campioni MC gia' disponibili e la
calibrazione temporale, evitando nuove baseline per ogni checkpoint.
Il job 2079596 e' stato fermato a 01:58:51 mentre eseguiva soltanto MC.
Le 3 run SMC complete e tutti i campioni MC prodotti sono conservati.

## Continuazione

Si eseguono soltanto i 9 seed SMC mancanti dei 12 gia' pianificati.
N=128, tilt 0.5,...,8, 16 livelli, 4 MH/livello, stessi stimatori,
resampling, rollback, pesi e score. Il codice scientifico originale non cambia;
la ripresa ne verifica gli SHA256 rispetto ai manifest del job originale.
Ogni seed mantiene le sue 65 popolazioni, iniziale + dopo ciascuna MH.
Si continua dopo il primo evento. Le stime zero sono conservate.

Risorse: 3 GPU sul nodo originale gpu007, 12 CPU, 60 GB, massimo 2 ore.
Ogni stima utilizza 1 GPU e 4 CPU. Il warm-up SMC e' separato e registrato.
Non vengono generati nuovi campioni crude MC, nemmeno per calibrazione.

## Confronto MC

Pool congelato: tutti i 192 file baseline di camera_ready_gradual_budget,
354420 campioni IID con decoder/evaluator identici. Non si usano i 200000
campioni del riferimento probabilistico come baseline.
Ordine dei file deterministico, nessuna selezione basata sugli esiti.
Ogni seed pianificato ha un blocco fisso disgiunto da 29535 campioni.
A ciascun checkpoint si usa il prefisso di lunghezza
floor(tempo SMC cumulativo misurato / costo per traiettoria MC calibrato).
I prefissi di uno stesso seed sono correlati; i blocchi dei diversi seed
sono disgiunti. Nessuna media tra livelli. Il pool disponibile e' sufficiente
per i budget osservati; un superamento del blocco genera un errore esplicito.
La calibrazione originale e' riutilizzata soltanto per la medesima GPU fisica
e allocazione di 4 CPU. Il costo MC e' un equivalente calibrato, non un nuovo
tempo misurato: CSV e grafici lo dichiarano. Le popolazioni MC di tutti i
checkpoint, comprese le singole MH, vengono ricavate senza simulazioni nuove.

Output nella stessa cartella camera_ready_same_run_ladder. I checkpoint
originali, le vecchie baseline e i manifest rimangono intatti. Nuovi file di
confronto: checkpoint_runs.csv, first_event_runs.csv, runs.csv, summary.csv,
saved_MC_blocks.json, comparison_policy.json, REPORT.html e due figure.

## Comandi della modifica autorizzata

Origine: C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main.
Destinazione: avon.scrtp.warwick.ac.uk:/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.
Tre file nuovi: same_run_reuse.py, slurm_avon_same_run_reuse.sh, questo documento.
Trasferimento con SHA256, senza sovrascrivere il codice scientifico precedente.

Comando gia' eseguito per fermare le baseline ridondanti:

```bash
scancel 2079596
```

Unico invio di continuazione, senza reinviare le 3 run gia' salvate:

```bash
sbatch --parsable --no-requeue --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --nodelist=gpu007 --ntasks=3 --cpus-per-task=4 --mem=60G --time=02:00:00 --job-name=mdlm-ladder-reuse --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/scripts/slurm_avon_same_run_reuse.sh
```
