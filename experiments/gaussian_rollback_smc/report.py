"""Produce the human-readable Italian results report from saved metrics."""
import csv
import json
import math
import statistics
from pathlib import Path


def create_report(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((out / "pilot_diagnostics.json").read_text(encoding="utf-8"))
    with (out / "summary.csv").open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        rows.append({k: (v if k == "method" else v == "True" if k == "stable_pilot" else float(v)) for k, v in r.items()})
    n, reps = manifest["particles"], manifest["replicates"]
    find = lambda tilt, depth, method: next(r for r in rows if r["lambda"] == tilt and r["max_depth"] == depth and r["method"] == method)
    best = min((r for r in rows if r["config_id"] > 0), key=lambda r: r["rmse"])
    crude = rows[0]
    stable = [r for r in rows if r["config_id"] > 0 and r["stable_pilot"]]
    stability_text = (f"Nelle configurazioni con pilot stabile, i rapporti varianza empirica/predetta vanno da {min(r['empirical_over_predicted'] for r in stable):.3f} a {max(r['empirical_over_predicted'] for r in stable):.3f}. Vanno letti insieme agli errori Monte Carlo delle due stime, disponibili nei CSV. Non sono identità numeriche esatte."
                      if stable else "Nessuna configurazione a tilt positivo supera i criteri di stabilità del pilot: questo budget non consente un confronto affidabile dei secondi momenti.")
    unstable = sorted(set((int(r["lambda"]), int(r["max_depth"])) for r in rows if not r["stable_pilot"]))
    lines = [
        "# Risultati: RollBack SMC gaussiano senza resampling",
        "",
        "## Disegno riproducibile",
        "",
        f"Probabilità esatta p=0.001; soglia a={manifest['threshold']:.12f}. T=16 transizioni originali N(x_t,1/T), R=8 livelli lineari, una mutazione MH per livello. N={n} particelle per replica, {reps} repliche indipendenti per configurazione. Lambda=1,2,3; d uniforme in {{1,...,D}}, D=1,4,16. Controllo aggiuntivo lambda=0,D=16. Nessun resampling.",
        "",
        f"Per ogni configurazione, pilot indipendente di {manifest['pilot_blocks'] * manifest['pilot_size']:,} storie, in {manifest['pilot_blocks']} blocchi. Seed principale {manifest['seed']}; i seed completi di ogni replica/blocco sono nei CSV. Le due stime SMC condividono il medesimo campione entro la replica: il loro confronto è appaiato. Le righe non costituiscono test multipli corretti e la selezione del minimo RMSE è descrittiva.",
        "",
        f"Durata della pipeline sperimentale: {manifest['total_seconds']:.2f} secondi, inclusi pilot, salvataggi e grafici, esclusi test e scrittura della derivazione. Python {manifest['python']}, NumPy {manifest['numpy']}, SciPy {manifest['scipy']}. Versioni complete e piattaforma in `results/manifest.json`.",
        "",
        "## Risultati teorici, separati dalle osservazioni",
        "",
        "- Con W prodotto dei potenziali prima delle mutazioni, C=W exp(-lambda x_finale), U=C h e V=C, valgono E(U)=p ed E(V)=1 sotto la vera legge forward delle storie.",
        "- IS ordinario è la media di U: non distorto per ogni N, varianza esatta Var(U)/N.",
        "- SNIS è media(U)/media(V): consistente, generalmente distorto a N finito; coefficiente CLT E[(U-pV)^2]. Il suo rapporto di varianze col pilot è un confronto asintotico.",
        f"- Crude MC ha coefficiente CLT p(1-p)=0.000999, varianza della stima {0.000999/n:.9g}, RMSE teorico {(0.000999/n)**.5:.9g}, probabilità di stima nulla {0.999**n:.4%}.",
        "- I coefficienti ideali sotto pi_lambda non valgono per il sampler RollBack. Tutti i passaggi sono in `DERIVAZIONE.md`.",
        "",
        "## Accuratezza osservata a pari numero di particelle",
        "",
        "Bias = media delle repliche meno p. SE(bias) misura l'incertezza Monte Carlo nella stima del bias; non è l'errore standard di una singola stima. Varianza campionaria con denominatore B-1; RMSE rispetto al p esatto. Le unità delle colonne sono esplicite.",
        "",
        "| lambda | D | Metodo | Bias / 1e-5 | SE(bias) / 1e-5 | Varianza / 1e-7 | RMSE / 1e-4 | Zeri % |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['lambda']:g} | {r['max_depth']:g} | {r['method']} | {r['bias']/1e-5:.3f} | {r['bias_mc_se']/1e-5:.3f} | {r['variance']/1e-7:.4f} | {r['rmse']/1e-4:.4f} | {100*r['zero_frequency']:.2f} |")
    lines += ["", "## Costo osservato", "",
              "Una transizione conta una nuova innovazione gaussiana del decoder, anche nelle proposal rifiutate. Costo esatto NT+somma(d) per replica SMC. Il tempo mediano include entrambi gli stimatori; non va sommato due volte. Il pilot ha un costo aggiuntivo distinto. I tempi sono specifici di questa esecuzione.", "",
              "| lambda | D | Transizioni medie / replica | Costo / crude MC | Tempo mediano (ms) | Accettazione % | Pilot (s) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["method"] == "snis":
            continue
        cid = int(r["config_id"])
        pilot_seconds = diagnostics[str(cid)]["pilot_seconds"] if cid >= 0 else 0
        lines.append(f"| {r['lambda']:g} | {r['max_depth']:g} | {r['mean_transitions']:.1f} | {r['mean_transitions']/(16*n):.3f} | {1000*r['median_seconds']:.3f} | {100*r['acceptance']:.2f} | {pilot_seconds:.3f} |")
    lines += ["", "## Varianza empirica e predizione indipendente", "",
              "Le predizioni utilizzano il pilot delle vere storie RollBack. Per crude MC e lambda=0 è utilizzato il valore analitico esatto. Le colonne riportano coefficienti N Var(stima), non varianze delle stime. SE pilot è l'errore standard Monte Carlo del coefficiente stimato. Il rapporto è varianza empirica / (coefficiente predetto/N).", "",
              "| lambda | D | Metodo | N Var empirica / 1e-3 | Coeff. predetto / 1e-3 | SE pilot / 1e-3 | Rapporto | Pilot stabile |",
              "|---:|---:|:---|---:|---:|---:|---:|:---|"]
    for r in rows:
        lines.append(f"| {r['lambda']:g} | {r['max_depth']:g} | {r['method']} | {n*r['variance']/1e-3:.4f} | {r['predicted_sigma2']/1e-3:.4f} | {n*r['prediction_se']/1e-3:.4f} | {r['empirical_over_predicted']:.3f} | {'sì' if r['stable_pilot'] else 'no'} |")
    lines += ["", "La stabilità richiede SE relativo <15%, massimo contributo <2% del totale dei quadrati, e differenza relativa tra metà del pilot <40%. La diagnostica non certifica l'assenza di code non osservate. La quota dello 0.1% dei contributi maggiori è salvata in JSON; essendo l'evento raro, tale quota può essere alta anche per crude MC.", "",
              "Configurazioni con pilot non stabile (lambda,D): " + ", ".join(map(str, unstable)) + ".", "",
              "## Osservazioni sperimentali", "",
              f"1. Il minimo RMSE osservato nella griglia è {best['method']}, lambda={best['lambda']:g}, D={best['max_depth']:g}: {best['rmse']:.6g}. Il suo rapporto di varianza con il crude MC teorico a pari N è {best['variance']/(.000999/n):.3f}. È un risultato per questa griglia, non una dimostrazione di ottimalità.",
              f"2. Il crude MC ha media osservata {crude['mean']:.8f}, bias stimato {crude['bias']:.3g} con SE {crude['bias_mc_se']:.3g}; questa fluttuazione non cambia la sua non distorsione teorica. Per i confronti principali il riferimento è p esatto e la varianza teorica, non il solo crude MC empirico.",
              "3. " + stability_text,
              "4. Meno stime nulle non implica minore varianza: aumentare il tilt può migliorare la scoperta dell'evento e contemporaneamente concentrare i contributi corretti in poche storie. Per lambda=3 i momenti possono restare instabili anche con milioni di storie pilota; non attribuiamo automaticamente le discrepanze a un errore della CLT.",
              f"5. Per la configurazione con RMSE minimo, il rapporto varianza × transizioni rispetto a crude MC è {best['variance_cost_vs_crude']:.3f}. Un valore maggiore di 1 indica che il guadagno per particella non compensa il costo aggiuntivo in transizioni. Questo indicatore non è una replica a budget abbinato; per SNIS non incorpora il bias.",
              "6. SNIS e IS ordinario hanno zeri simultanei perché il denominatore SNIS è positivo. La loro differenza di varianza e bias dipende dalla variabilità del denominatore e dalla sua covarianza con il numeratore; non si assume che SNIS migliori sempre IS ordinario.",
              ""]
    with (out / "replicates.csv").open(encoding="utf-8") as f:
        pairs = [float(r["snis"])-float(r["ordinary"]) for r in csv.DictReader(f) if r["config_id"] == "9"]
    lines += [f"Per lambda=3,D=16, il confronto appaiato SNIS meno ordinario dà una differenza media di {statistics.mean(pairs):.6g}, con errore Monte Carlo {statistics.stdev(pairs)/math.sqrt(len(pairs)):.6g}. Poiché l'ordinario è teoricamente non distorto, questa differenza stima il bias di SNIS con maggiore precisione della sola media SNIS meno p. È circa il {100*statistics.mean(pairs)/.001:.2f}% di p: evidenza di un effetto della normalizzazione a N finito in questa configurazione.", ""]
    followup_path = out / "zero_tilt_followup.json"
    if followup_path.exists():
        followup = json.loads(followup_path.read_text(encoding="utf-8"))
        original_null = find(0, 16, "ordinary")
        lines += ["## Controllo aggiuntivo a tilt nullo", "",
                  f"Il controllo principale a lambda=0 ha rapporto di varianza empirica/teorica {original_null['empirical_over_predicted']:.3f}, insolitamente basso; non è stato eliminato né sostituito. Per verificarlo abbiamo eseguito altre {followup['replicates']} repliche indipendenti, N={followup['n']}, con seed principale {followup['seed']} e stream {followup['stream']}, salvate separatamente.", "",
                  f"Nel controllo aggiuntivo la varianza è {followup['variance']:.9g}, contro {followup['theoretical_variance']:.9g} teorica (rapporto {followup['variance']/followup['theoretical_variance']:.4f}). La differenza è {followup['variance_z_score']:.3f} errori standard della varianza campionaria, calcolati dal quarto momento binomiale esatto. La frequenza di zeri è {followup['zero_frequency']:.3%}, contro {followup['theoretical_zero_frequency']:.3%} teorico. Questo controllo è compatibile con la legge prevista; la deviazione del primo controllo resta documentata. Costo aggiuntivo: {followup['seconds']:.2f} secondi.", ""]
    image_prefix = "results/" if out.name == "results" else ""
    lines += ["## Verifiche", "",
              "Otto test mirati superati: marginale originale; prefisso e profondità uniforme; cancellazione algebrica del rapporto MH; invarianza di media, varianza e covarianza della legge tilted completa; pesi prima della mutazione e prodotto degli incrementi di normalizzazione; limiti senza mutazione e a tilt nullo; identità FK e inverse sui momenti e sull'evento; equivalenza algebrica degli stimatori. I controlli statistici usano seed separati e tolleranze Monte Carlo esplicite. L'invarianza teorica è dimostrata, non dedotta soltanto dai test.",
              "", "## Grafici", "",
              f"![Varianze: repliche e pilot]({image_prefix}variance_comparison.png)", "",
              f"![Accuratezza e costo]({image_prefix}accuracy_and_cost.png)", "",
              f"![Campionamento ideale e RollBack effettivo]({image_prefix}ideal_vs_actual.png)", "",
              "Le barre rappresentano circa 1.96 errori Monte Carlo della varianza o del bias stimati dalle repliche, e possono essere poco affidabili quando dominano le code. Le croci nere segnalano pilot instabili. Le curve ideali sono solo formule analitiche di un diverso esperimento.", ""]
    report_path = out.parent / "RISULTATI.md" if out.name == "results" else out / "RISULTATI.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results")
    create_report(parser.parse_args().output)
