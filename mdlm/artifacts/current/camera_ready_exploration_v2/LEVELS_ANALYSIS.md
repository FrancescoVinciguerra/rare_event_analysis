# Analisi dei livelli salvati

Stime dopo le mutazioni lungo le 24 run originali. A ogni livello si usa il suo tilt, non 3. Per k<R, score, pesi e Z sono recuperati dagli input della pesatura successiva. Il livello 3 riproduce gli stimatori e gli eventi già salvati. Sono presenti i resampling intermedi della run originale: questi punti non riproducono run che terminino a ogni livello escludendo il rispettivo resampling finale. I tempi cumulativi intermedi e le copie terminali non sono salvati: nessun MC a pari tempo viene inventato.
