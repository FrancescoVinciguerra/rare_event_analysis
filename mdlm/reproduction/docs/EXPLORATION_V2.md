# Esplorazione RollBack SMC con incrementi di tilt 0,5

Stato: codice e disegno preparati; nessuna nuova simulazione avviata.
Il pilot precedente resta in `artifacts/current/camera_ready_low_budget_v1`.
Le nuove configurazioni usano un'unica directory `camera_ready_exploration_v2`.

## Disegno congelato prima delle nuove run

Stesso decoder MDLM-OWT a 32 passi DDPM più completamento categorico,
lunghezza 128, stesso classificatore e primi 100 token, score Phi=log(q).
Eventi score >=0.9 e >=0.99 dalla stessa popolazione terminale.
Depth indipendenti e uniformi in {4,8,16,24,32}; systematic ESS/N<0.5;
pesatura, eventuale resampling, tutte le mosse MH. Nessun resampling finale.
Lo score, P, i kernel nativi e l'accettazione MH restano quelli del pilot.

Unica schedule per tutti i casi: **0 → 0.5 → 1 → 1.5 → 2 → 2.5 → 3**.
Sei livelli con incremento costante 0.5, come richiesto dall'utente.

| Caso | N | Schedule | Livelli | MH/livello | Transizioni attese | Crude equivalenti attesi |
|---|---:|---|---:|---:|---:|---:|
| A | 64 | uniforme 0.5 | 6 | 4 | 29452.8 | 892.5 |
| B | 64 | uniforme 0.5 | 6 | 8 | 56793.6 | 1721.0 |
| C | 128 | uniforme 0.5 | 6 | 4 | 58905.6 | 1785.0 |

Otto run indipendenti per caso, 24 in totale. Esecuzione interlacciata A,B,C
per ciascun indice di run, così un limite di tempo non esclude tutti i casi più
grandi. Seed sampler = 1400000000 + 10000*indice_caso + 1000000*indice_run;
crude usa SeedSequence con stream 925081, separato e registrato. Nessuna media
fra livelli o configurazioni. Tutte le nuove stime riguardano il tilt finale 3.

B e C confrontano più mutazioni contro più particelle a costo atteso simile.
A e C isolano N mantenendo la stessa schedule e le stesse mosse per livello.
A e B isolano il numero di mosse MH mantenendo N e schedule fissi.
La baseline primaria di OGNI run usa floor(costo_effettivo/33) traiettorie
native crude indipendenti. Contare inizializzazione e tutti i suffissi, anche
rifiutati. Il tempo è registrato, senza assegnare il budget a wall clock.
Non si aggiunge una seconda baseline a tempo, riducendo il costo esplorativo.

## Diagnostica dell'overlap e limiti

Le 5746 traiettorie crude già salvate per N=128 permettono di stimare i momenti
di q^lambda. Il rapporto [E_pi_a G]^2/E_pi_a[G^2] è un indicatore di overlap
tra target ideali, stimato dai momenti sotto P. NON è l'ESS che avrà la nuova
popolazione SMC finita, non tiene conto della dipendenza genealogica e non
dimostra che quattro mosse MH abbiano raggiunto la target. Il primo salto 0→0.5
dà un ESS relativo del 5.3% sul campione crude aggregato: non garantiamo che
la nuova schedule risolva la degenerazione. Non viene presentato
come risultato di nuove simulazioni né usato come garanzia di accuratezza.

Restano necessari i controlli sulle run effettive: ESS prima/dopo resampling,
massimo peso, genealogia, acceptance per livello e depth, ESS inverso e reverse
mass. La nuova implementazione salva anche gli input di OGNI pesatura,
log-incremento del normalizzatore e log(Z_hat), per ricostruire l'intero prodotto.
Il controllo della ricorrenza viene eseguito prima di salvare una run completa.

Gli stimatori SMC non normalizzato e SNIS usano la stessa popolazione. La
reference resta 1516/200000 e 444/200000, non una ground truth esatta.
Presentazione: soltanto due figure di performance, log10(1+|stima/reference-1|)
e stima/reference; zoom e code complete, colori precedenti, zeri inclusi,
conteggi espliciti, mediana e media distinte. I due casi con N=64 non vengono
mai aggregati. Conservare tutti i risultati, inclusi quelli sfavorevoli.
Qualunque scelta successiva richiede run di validazione con nuovi seed.

## Preparazione ed esecuzione

I file di produzione esistenti non vengono modificati. `exploratory_kernel.py`
è la versione materializzata del kernel originale con schedule esplicita e
tracciamento dei normalizzatori. La corrispondenza alla schedule lineare è
verificata con stessi seed, particelle, pesi, genealogie e accettazioni.
Nessun modello viene importato/caricato dai comandi `plan` o dai test.

Comando scientifico per ciascuna GPU sul nodo assegnato (WORKER_ID=0,1,2):

```bash
python -m diffusion_toxicity_smc_experiment.exploratory run --study /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/configs/mdlm_owt_exploration_v2.yaml --output-root /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/current --max-seconds 6600 --workers 3 --worker-id WORKER_ID
```

Risorse proposte: un job, **tre Quadro RTX6000 sullo stesso nodo, 3 task da
4 CPU (12 CPU totali), 60 GB RAM, massimo 2 ore**. Il preflight del 2 novembre
2026 conferma 3 GPU e 191500 MB RAM per nodo, MaxTRES gres/gpu=12 per job e
MaxTRESPU gres/gpu=24 per la QoS part_gpu. Sono limiti verificati, non una
garanzia sui tempi di attesa. SLURM 20.11.9 supporta `srun --exact`.

Ogni processo SLURM vede una sola GPU, carica il modello e svolge run complete
indipendenti con il proprio crude abbinato sulla stessa GPU. Le particelle
di una run restano sulla stessa GPU. L'assegnazione è indice_run modulo 3:
9,9,6 run complete per processo (3,3,2 per configurazione), con gli stessi seed
della versione seriale. Le directory sono separate, per evitare scritture
concorrenti. Il coordinatore unisce CSV e report controllando duplicati,
identità dello studio e hash del codice. Non aggrega configurazioni diverse.

Tempo indicativo: **1–1.5 ore di calcolo**, oltre a coda e inizializzazione.
Il massimo è 3 GPU × 2 ore = 6 GPU-ore. Parallelizzare riduce l'attesa,
non il costo totale delle simulazioni. I tre warm-up sono registrati
separatamente ed esclusi dalle stime. Checkpoint dopo popolazione, baseline e
run; CSV e report dei worker aggiornati dopo ogni run completa. Un unico report
finale con due figure. Nessun rilancio automatico, installazione o download.

## Azione remota da confermare

La guida personale `.codex-private/avon/WARWICK_SCRTP_HPC_GUIDE.md`, sezione 2,
richiede: «Attendere la conferma dell'utente prima di eseguire il comando
mostrato». La conferma riguarda il trasferimento dei nuovi file e UN job con tre GPU.
Le verifiche di sola lettura possono precederla.

ORIGINE: PC Windows, C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main.

DESTINAZIONE: avon.scrtp.warwick.ac.uk, directory
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.

Il pacchetto aggiunge soltanto il nuovo YAML, quattro nuovi moduli Python,
due script SLURM, il test, il plotter e questo documento. L'elenco con SHA256
è nel file `artifacts/camera_ready_exploration_v2_preparation.json`.
I file già esistenti e i loro hash sono verificati prima dell'installazione;
un file diverso nello stesso percorso fa interrompere il deploy.

Comando locale completo sul canale SSH già autenticato:

```powershell
& 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.venv/Scripts/python.exe' 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/tmp/avon_channel.py' --file 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.codex-private/avon/exploration_v2_deploy.sh'
```

Lo script collegato contiene preflight di sola lettura, trasferimento del
pacchetto con controllo SHA256, estrazione dei soli file nuovi, dry-run e questo
singolo invio, senza dipendenze o catene di job:

```bash
sbatch --parsable --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=60G --time=02:00:00 --job-name=mdlm-exploration-v2 --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/reproduction/scripts/slurm_avon_exploration_v2.sh
```

Output: artifacts/current/camera_ready_exploration_v2/{manifest.json,runs.csv,
summary.csv,REPORT.html,REPORT.md,report_status.json,workers/worker_0/,
workers/worker_1/,workers/worker_2/}. Ogni worker conserva popolazioni,
baseline, run complete, manifest e warm-up propri.
I due PNG vengono generati sul PC dai CSV reali dopo il download, senza nuove
simulazioni. Nessuna cancellazione o sostituzione dei risultati precedenti.
