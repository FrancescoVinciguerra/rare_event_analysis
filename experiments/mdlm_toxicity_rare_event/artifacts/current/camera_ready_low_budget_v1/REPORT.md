# RollBack SMC — camera-ready: confronto a budget limitato

PILOT PARZIALE: 30/40 popolazioni.

Risultati pilota: non trarre conclusioni definitive sulle performance.
Reference finite IID, non probabilità esatte. Nessuna media fra tilt o configurazioni.

Il numero di campioni crude a tempo è fissato prima della generazione usando una calibrazione indipendente. Il matching wall clock è approssimato; rapporto tempo crude/SMC mediano 1.005, massimo 1.013. I tempi effettivi e lo sforamento sono nel CSV.

| Confronto | Soglia | N | Metodo | Run | Zero | Rapporto Q50 | Q90 | Q99 | Mediana errore log |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| pari transizioni | 0.9 | 16 | SMC non normalizzato | 8 | 37.5% | 5.64e-05 | 1.93 | 5.95 | 0.301 |
| pari transizioni | 0.9 | 16 | SMC + SNIS | 8 | 37.5% | 0.000651 | 30.4 | 85.3 | 0.301 |
| pari transizioni | 0.9 | 16 | Crude MC | 8 | 25.0% | 1.58 | 2.31 | 2.43 | 0.309 |
| pari transizioni | 0.99 | 16 | SMC non normalizzato | 8 | 75.0% | 0 | 0.919 | 2.79 | 0.301 |
| pari transizioni | 0.99 | 16 | SMC + SNIS | 8 | 75.0% | 0 | 19.4 | 40.5 | 0.301 |
| pari transizioni | 0.99 | 16 | Crude MC | 8 | 100.0% | 0 | 0 | 0 | 0.301 |
| pari transizioni | 0.9 | 32 | SMC non normalizzato | 8 | 0.0% | 0.000694 | 0.213 | 0.651 | 0.301 |
| pari transizioni | 0.9 | 32 | SMC + SNIS | 8 | 0.0% | 0.521 | 41.9 | 101 | 0.3 |
| pari transizioni | 0.9 | 32 | Crude MC | 8 | 0.0% | 1.39 | 2.01 | 2.65 | 0.137 |
| pari transizioni | 0.99 | 32 | SMC non normalizzato | 8 | 62.5% | 0 | 0.135 | 0.418 | 0.301 |
| pari transizioni | 0.99 | 32 | SMC + SNIS | 8 | 62.5% | 0 | 7.59 | 23.4 | 0.301 |
| pari transizioni | 0.99 | 32 | Crude MC | 8 | 50.0% | 0.971 | 3.75 | 3.81 | 0.301 |
| pari transizioni | 0.9 | 64 | SMC non normalizzato | 8 | 0.0% | 0.0146 | 2.09 | 2.14 | 0.301 |
| pari transizioni | 0.9 | 64 | SMC + SNIS | 8 | 0.0% | 11.3 | 74.5 | 101 | 0.856 |
| pari transizioni | 0.9 | 64 | Crude MC | 8 | 0.0% | 1.24 | 1.73 | 1.89 | 0.139 |
| pari transizioni | 0.99 | 64 | SMC non normalizzato | 8 | 12.5% | 0.00273 | 2.53 | 6.59 | 0.301 |
| pari transizioni | 0.99 | 64 | SMC + SNIS | 8 | 12.5% | 7.93 | 167 | 335 | 0.733 |
| pari transizioni | 0.99 | 64 | Crude MC | 8 | 25.0% | 0.947 | 1.87 | 1.89 | 0.268 |
| pari transizioni | 0.9 | 128 | SMC non normalizzato | 6 | 0.0% | 1 | 2.36 | 3.16 | 0.109 |
| pari transizioni | 0.9 | 128 | SMC + SNIS | 6 | 0.0% | 20.6 | 48 | 60.1 | 1.26 |
| pari transizioni | 0.9 | 128 | Crude MC | 6 | 0.0% | 0.895 | 1.94 | 2.55 | 0.0711 |
| pari transizioni | 0.99 | 128 | SMC non normalizzato | 6 | 0.0% | 2.26 | 2.4 | 2.42 | 0.353 |
| pari transizioni | 0.99 | 128 | SMC + SNIS | 6 | 0.0% | 16.9 | 57 | 70.8 | 1.14 |
| pari transizioni | 0.99 | 128 | Crude MC | 6 | 0.0% | 1.39 | 1.65 | 1.86 | 0.148 |
| pari wall clock nominale | 0.9 | 16 | SMC non normalizzato | 8 | 37.5% | 5.64e-05 | 1.93 | 5.95 | 0.301 |
| pari wall clock nominale | 0.9 | 16 | SMC + SNIS | 8 | 37.5% | 0.000651 | 30.4 | 85.3 | 0.301 |
| pari wall clock nominale | 0.9 | 16 | Crude MC | 8 | 37.5% | 1.36 | 2.71 | 2.71 | 0.301 |
| pari wall clock nominale | 0.99 | 16 | SMC non normalizzato | 8 | 75.0% | 0 | 0.919 | 2.79 | 0.301 |
| pari wall clock nominale | 0.99 | 16 | SMC + SNIS | 8 | 75.0% | 0 | 19.4 | 40.5 | 0.301 |
| pari wall clock nominale | 0.99 | 16 | Crude MC | 8 | 87.5% | 0 | 1.85 | 5.74 | 0.301 |
| pari wall clock nominale | 0.9 | 32 | SMC non normalizzato | 8 | 0.0% | 0.000694 | 0.213 | 0.651 | 0.301 |
| pari wall clock nominale | 0.9 | 32 | SMC + SNIS | 8 | 0.0% | 0.521 | 41.9 | 101 | 0.3 |
| pari wall clock nominale | 0.9 | 32 | Crude MC | 8 | 12.5% | 1.07 | 2.17 | 2.17 | 0.178 |
| pari wall clock nominale | 0.99 | 32 | SMC non normalizzato | 8 | 62.5% | 0 | 0.135 | 0.418 | 0.301 |
| pari wall clock nominale | 0.99 | 32 | SMC + SNIS | 8 | 62.5% | 0 | 7.59 | 23.4 | 0.301 |
| pari wall clock nominale | 0.99 | 32 | Crude MC | 8 | 62.5% | 0 | 2.41 | 3.58 | 0.301 |
| pari wall clock nominale | 0.9 | 64 | SMC non normalizzato | 8 | 0.0% | 0.0146 | 2.09 | 2.14 | 0.301 |
| pari wall clock nominale | 0.9 | 64 | SMC + SNIS | 8 | 0.0% | 11.3 | 74.5 | 101 | 0.856 |
| pari wall clock nominale | 0.9 | 64 | Crude MC | 8 | 0.0% | 0.815 | 1.24 | 1.57 | 0.0793 |
| pari wall clock nominale | 0.99 | 64 | SMC non normalizzato | 8 | 12.5% | 0.00273 | 2.53 | 6.59 | 0.301 |
| pari wall clock nominale | 0.99 | 64 | SMC + SNIS | 8 | 12.5% | 7.93 | 167 | 335 | 0.733 |
| pari wall clock nominale | 0.99 | 64 | Crude MC | 8 | 25.0% | 0.912 | 2.1 | 2.68 | 0.154 |
| pari wall clock nominale | 0.9 | 128 | SMC non normalizzato | 6 | 0.0% | 1 | 2.36 | 3.16 | 0.109 |
| pari wall clock nominale | 0.9 | 128 | SMC + SNIS | 6 | 0.0% | 20.6 | 48 | 60.1 | 1.26 |
| pari wall clock nominale | 0.9 | 128 | Crude MC | 6 | 0.0% | 1.01 | 1.22 | 1.23 | 0.0782 |
| pari wall clock nominale | 0.99 | 128 | SMC non normalizzato | 6 | 0.0% | 2.26 | 2.4 | 2.42 | 0.353 |
| pari wall clock nominale | 0.99 | 128 | SMC + SNIS | 6 | 0.0% | 16.9 | 57 | 70.8 | 1.14 |
| pari wall clock nominale | 0.99 | 128 | Crude MC | 6 | 0.0% | 0.699 | 1.61 | 1.84 | 0.187 |

I due grafici completi, zoom, diagnostiche e note metodologiche sono incorporati in REPORT.html.
PNG ad alta risoluzione: eseguire il plotter locale sul runs.csv scaricato.
