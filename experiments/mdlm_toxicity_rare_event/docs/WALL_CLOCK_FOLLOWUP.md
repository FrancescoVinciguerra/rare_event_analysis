# Crude MC al tempo delle run SMC già eseguite

Si riutilizzano le 24 popolazioni e i tempi SMC del job 2076203: A N64/4 MH,
B N64/8 MH, C N128/4 MH; sei livelli 0.5,1,1.5,2,2.5,3. Nessuna run SMC ripetuta.
Target per run: il suo tempo originale misurato con sincronizzazione GPU,
inclusi generazione, score, pesatura, resampling, RollBack–MH e due correzioni.
Tempi medi A=156.76 s, B=302.25 s, C=311.76 s.

Il nuovo crude genera traiettorie native complete sotto lo stesso P e calcola
entrambi gli eventi, score>=0.9 e >=0.99. Si richiedono le tre GPU del nodo
ORIGINALE gpu007. SLURM_STEP_GPUS viene confrontato con il warm-up originale
per assegnare ciascuna baseline alla stessa GPU fisica della misura SMC.

Warm-up da 16 traiettorie e 3 calibrazioni indipendenti da 128 per GPU,
esclusi dalle stime finali e documentati separatamente. Prima di generare ogni
baseline finale si fissa n=floor(tempo_SMC/mediana_secondi_per_traiettoria).
Stream finale 925082, distinto dal vecchio crude a transizioni 925081 e da
warm-up/calibrazione. Nessuno stop che dipende dai risultati osservati.

Il matching temporale è approssimato e verificato dai tempi effettivi. Salvare
rapporto tempo_crude/tempo_SMC e scarti; segnalare scarti oltre il 5%, senza
filtrare o ripetere run sulla base dell'accuratezza. Nessuna ricostruzione di
stime o zeri dal solo tempo e nessuna trasformazione dei vecchi campioni.
Stessi due boxplot, tutti gli zeri e tutte le code; confronto precedente conservato.

Origine: C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main.
Destinazione: avon.scrtp.warwick.ac.uk, directory
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.
Nuovi file: YAML, modulo wall_clock_followup.py, due script, test, plotter e
questo documento. Dati nuovi: artifacts/current/camera_ready_exploration_v2_wall_clock.
Nessuna sostituzione di codice o dati esistenti. Un job, tre GPU, 12 CPU, 60 GB,
massimo un'ora; durata indicativa 40–45 minuti più coda. Nessun rilancio automatico.

Comando locale completo, sul canale autenticato:

```powershell
& 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.venv/Scripts/python.exe' 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/tmp/avon_channel.py' --file 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.codex-private/avon/wall_clock_deploy.sh'
```

Lo script verifica gli hash, trasferisce soltanto i nuovi file, esegue dry-run e:

```bash
sbatch --parsable --account=stats --partition=gpu --qos=normal --nodes=1 --nodelist=gpu007 --gres=gpu:quadro_rtx_6000:3 --ntasks=3 --cpus-per-task=4 --mem=60G --time=01:00:00 --job-name=mdlm-wall-clock --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/scripts/slurm_avon_wall_clock.sh
```

Comando scientifico per processo, WORKER_ID=0,1,2:

```bash
python -m diffusion_toxicity_smc_experiment.wall_clock_followup run --study /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/configs/mdlm_owt_wall_clock_followup.yaml --output-root /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/experiments/mdlm_toxicity_rare_event/artifacts/current --worker-id WORKER_ID
```

La guida personale WARWICK_SCRTP_HPC_GUIDE.md §2 richiede «Attendere la conferma
dell'utente prima di eseguire il comando mostrato»: trasferimento e job restano
in attesa di conferma del comando concreto. Il preflight è di sola lettura.
