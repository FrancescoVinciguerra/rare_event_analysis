# RollBack SMC a basso budget: distribuzioni delle singole stime

Stato: piano aggiornato alla richiesta di soglia 0.999, tilt fino a 3 e
popolazioni più piccole. Nessun nuovo job ancora inviato.

## Obiettivo

Confrontare SMC della misura non normalizzata, SMC+SNIS e crude MC per eventi
rari a costo GPU abbinato. Cercare stime positive sulla scala corretta, con
medie, mediane e code visibili; non definire un successo mediante soglie di
errore del 50% o una percentuale minima di run. Non mediare fra tilt.

La metrica principale rimane log10(1+abs(stima/riferimento-1)). Mostrarla sulle
singole run, inclusi tutti gli zeri, e mostrare anche il rapporto
stima/riferimento per distinguere sottostime e sovrastime. Ogni boxplot ha
linea nera per la mediana e rombo per la media della quantità sull'asse,
entrambi calcolati su TUTTE le run. Box 25%-75%, baffi 1.5 IQR, tutti gli
outlier. Zoom e distribuzione completa; conteggi fuori scala espliciti.
Gli zeri non sono una stima precisa e non vengono esclusi o sostituiti.

## Eventi e riferimento indipendente già disponibile

| Soglia score | Eventi su 200000 | p_riferimento | Ruolo |
|---|---:|---:|---|
|0.9|1516|0.00758|Controllo; non guida la ricerca di budget inferiori a MC|
|0.99|444|0.00222|Confronto principale|
|0.999|26|0.00013|Confronto principale, riferimento meno preciso|

Le soglie non sono probabilità. Per 0.999 il Wilson 95% del riferimento è
[0.0000887212,0.000190481], mostrato nei grafici del rapporto. Non trattare
questo riferimento come una ground truth esatta.
Origine: shared/output/github_reviewer_artifact/data/final_analysis/direct_reference.csv,
SHA256 b75e4ddd1a09871ae0fab7682769e8bcc05a3c1fa867a98aa5cac3d7a51bd535.
Decoder/evaluator verificati contro il manifest originale e congelati nello YAML.

## Disegno fissato

- N = 16,32,64.
- Tilt finali = 0.5,1,1.5,2,2.5,3.
- Ogni schedule parte da 0 e aggiunge 0.5 fino al proprio endpoint.
- Quattro mosse RollBack-MH per particella e livello.
- Venti run indipendenti per configurazione: 18 configurazioni, 360 popolazioni.
- Entrambi gli stimatori e tutte e tre le soglie dalla STESSA popolazione:
  aggiungere 0.999 non duplica generazione, modello o classificazione.
- Ogni endpoint è una vera run terminale separata, senza resampling finale.
  Nessuna ricostruzione di tempi mediante tilt/3 e nessuna media fra tilt.
- Stream e seed distinti e registrati per ogni popolazione e baseline MC.

Rimane invariata la pipeline MDLM: 32 passi DDPM più completamento categorico
stocastico, score log(q), d uniforme in {4,8,16,24,32}, systematic quando
ESS/N<0.5. Pesatura una volta, eventuale resampling, quattro MH alla temperatura
corrente. Le mutazioni non cambiano i pesi. Z_hat aggiornato a ogni livello.
Correzione inversa con il tilt della run corrente.

Costi INDICATIVI in transizioni native equivalenti (non sono i budget temporali):

| N | MC equivalenti attesi a tilt 0.5 | MC equivalenti attesi a tilt 3 |
|---|---:|---:|
|16|50.5|223.1|
|32|101.0|446.3|
|64|202.1|892.5|

La conversione usa E[costo]=N*(33+17.8*4*R), incluse proposte rifiutate.
Il confronto principale usa invece tempi REALI. I piccoli batch possono avere
costi GPU diversi da questa proporzione; la misura lo renderà visibile.
Per orientare la scala, 1/p_ref corrisponde a circa 450 traiettorie per 0.99 e
7692 per 0.999: è il tempo medio di attesa del PRIMO evento, non un requisito
sufficiente per un buon stimatore MC. Non introduciamo un criterio arbitrario
per definire 'buono'. Riportiamo distribuzioni delle stime e costo effettivo.

## Budget temporale e salvataggi

Warm-up e tre calibrazioni indipendenti da 128 traiettorie per GPU, esclusi
dalle singole stime e registrati separatamente. Sincronizzazione GPU prima e
dopo ogni misura. Includere generazione iniziale, score, pesatura,
resampling, tutte le proposte e decisioni MH e correzioni finali.
Prima di osservare il nuovo crude fissare n=floor(tempo_SMC/c), con c mediana
dei secondi per traiettoria nelle calibrazioni indipendenti. Generare n
traiettorie COMPLETE sotto lo stesso P e sulla stessa GPU fisica.
Registrare tempo reale MC, rapporto dei tempi e scarti oltre il 5%, senza
filtrare o ripetere run in base ad accuratezza o esiti.

CSV delle singole run con stime, n_MC, transizioni, tempi, seed e diagnostiche.
CSV riassuntivo con media e mediana delle stime, dei rapporti e della metrica,
conteggio e percentuale di zeri, costi. Per MC il conteggio degli zeri è quello
EFFETTIVO delle sue run, non una previsione teorica. Acceptance, ESS, ESS inverso,
pesi massimi, antenati iniziali e sequenze terminali distinti salvati nei dati.
Checkpoint atomici dopo popolazione e baseline. Le run complete non si ripetono.
Una misura SMC parziale non si abbina a una GPU fisica diversa durante la ripresa.

## Figure e interpretazione

Due figure riassuntive: metrica logaritmica e rapporto. Tre righe, una per N;
per ciascun evento 0.99/0.999, zoom e code complete. Colori coerenti fra metodi,
mediana e media visibili, zeri di SMC/SNIS/MC sotto ciascun tilt. La soglia 0.9
rimane nei CSV e nel report come controllo, senza altri plot di performance.
Nessuna banda di accettabilità al 50% e nessuna percentuale di successo.

Per selezionare una configurazione guardare insieme centro, media, code, zeri
e costo; un miglioramento della sola mediana non dimostra una superiorità
generale. Se il pilot suggerisce un risparmio, congelare i parametri e confermare
con nuovi seed e MC a più budget. Non inserire le run di ricerca in una conferma
indipendente. Una fase ulteriore richiede un piano separato dopo questi risultati.

## Verifiche e output

14 test mirati superati prima dell'invio, fra cui le tre soglie dalla stessa
popolazione, medie/mediane su tutte le run e nessun criterio di accuratezza,
seed distinti, correzione al tilt giusto e ripresa sicura. Sintassi e dry-run
Bash, plotter verificato separatamente con dati sintetici (non sono risultati).

Il controllo degli hash ha verificato che evaluation.py e run_smc.py su Avon
sono esattamente le versioni del precedente job 2076203. La copia locale ha
un'ottimizzazione successiva per evitare una classificazione duplicata: non
viene trasferita. Per confrontabilità di algoritmo e tempi si conservano le
versioni originali di Avon; i 14 test sono stati ripetuti con successo anche
in una copia isolata che usa quei due file originali. Il gate di invio ne
controlla gli hash originali, senza sostituire alcun file scientifico remoto.

Output nuovi in artifacts/current/camera_ready_accuracy_budget_pilot:
runs.csv, summary.csv, REPORT.html, due PNG ad alta risoluzione e registri sotto
workers. I risultati precedenti sono conservati. Nessun job già avviato.

Matplotlib non è installato nell'ambiente Avon verificato: il job produce
CSV/HTML e archivio dei risultati, senza installare nuove dipendenze. Le due
figure PNG vengono renderizzate con l'ambiente locale già testato dopo il
download; il relativo codice è incluso nei file scientifici.

## Risorse e comandi remoti

ORIGINE: C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main
DESTINAZIONE: avon.scrtp.warwick.ac.uk, directory
/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code.

UN job, 3 GPU quadro_rtx_6000 su un nodo, 12 CPU totali, 60 GB, massimo 6 ore,
con checkpoint e senza rilanci automatici. La griglia è più ampia ma N ridotto;
la durata effettiva dipende dall'efficienza dei piccoli batch. Se raggiunge il
limite salva un report PARZIALE con il numero effettivo di run, non risultati
inventati. Prima dell'invio verificare partizione, QoS, limiti, moduli e stato
corrente con il preflight di sola lettura.

File trasferiti/creati o aggiornati, relativi a mdlm:

- configs/mdlm_owt_accuracy_budget_pilot.yaml
- src/diffusion_toxicity_smc_experiment/accuracy_budget.py
- scripts/slurm_avon_accuracy_budget.sh
- scripts/submit_avon_accuracy_budget.sh
- plots/plot_accuracy_budget.py
- analysis/analyze_saved_levels.py (renderer condiviso; nessun dato modificato)
- tests/test_accuracy_budget.py
- docs/ACCURACY_VS_BUDGET.md
- docs/ACCURACY_VS_BUDGET_PLAN.json

Se il renderer condiviso esiste già su Avon, conservarne prima una copia in
analysis/analyze_saved_levels.before_accuracy_budget.py; fermarsi se esiste
già il backup. Gli altri file nuovi non devono già esistere. Prima del
trasferimento verificare gli hash dei moduli scientifici esistenti; fermarsi
in caso di differenze. Verificare gli hash dei nuovi file dopo estrazione.
Le altre scritture sono output nuovi, log e un archivio temporaneo di upload;
nessuna cancellazione dei dati precedenti.

Comandi locali completi sul canale SSH autenticato; leggere e verificare il
preflight prima di eseguire il trasferimento e l'unico invio autorizzato:

```powershell
& 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.venv/Scripts/python.exe' 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/tmp/avon_channel.py' --file 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.codex-private/avon/accuracy_budget_preflight.sh'
& 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.venv/Scripts/python.exe' 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/tmp/avon_channel.py' --file 'C:/Users/UTENTE/Downloads/rea_of_llms_supplementary_code-main/.codex-private/avon/accuracy_budget_deploy.sh'
```

Comando di invio completo dopo preflight e trasferimento:

```bash
sbatch --parsable --account=stats --partition=gpu --qos=normal --gres=gpu:quadro_rtx_6000:3 --nodes=1 --ntasks=3 --cpus-per-task=4 --mem=60G --time=06:00:00 --job-name=mdlm-accuracy-budget --chdir=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code --export=ALL --output=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.out --error=/gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/logs/%x-%j.err /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/reproduction/scripts/slurm_avon_accuracy_budget.sh
```

Comando scientifico per GPU, WORKER_ID=0,1,2:

```bash
python -m diffusion_toxicity_smc_experiment.accuracy_budget run --study /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/configs/mdlm_owt_accuracy_budget_pilot.yaml --output-root /gpfs/home/stats/stuhvh/rea_of_llms_supplementary_code/mdlm/artifacts/current --max-seconds 20700 --workers 3 --worker-id WORKER_ID
```

La guida personale WARWICK_SCRTP_HPC_GUIDE.md §2 richiede di mostrare i comandi
prima delle scritture e attendere conferma. La richiesta corrente modifica il
pilot precedentemente proposto senza aumentare le risorse massime.
