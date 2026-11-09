# RollBack SMC con ESS e resampling systematic

## Politica effettivamente implementata

Il paper locale *ai4good_rare_event_estimation.tex*, descrizione sperimentale (righe 500–502) e algoritmo nell'appendice (righe 727–744), prescrive:

1. pesatura al nuovo livello;
2. resampling **systematic** se ESS/N < 0.5 e k < R;
3. mutazione RollBack–MH a beta_k, a ogni livello;
4. nessun resampling finale; conservazione dei pesi terminali.

La routine systematic usa un unico offset uniforme, conserva l'ordine di ingresso e produce antenati in ordine crescente, come il codice del paper. Non viene aggiunto sorting per stato né una permutazione. Il toy mantiene T=16, R=8, score x_0, e profondità uniforme in {1,...,D}, D=1,4,16: questi sono i parametri del toy/baseline, non le profondità del decoder linguistico del paper. Non si riproduce qui la media su molte popolazioni con tilt finali diversi: ogni lambda è una configurazione separata che stima lo stesso p.

## Modello e target

Una particella resta la traiettoria completa tau=(x_T,...,x_0), con x_T=0 e Q_t=N(x_t,1/T). La legge originale è

\[
P(d\tau)=\delta_0(dx_T)\prod_{t=T}^{1}Q_t(dx_{t-1}\mid x_t).
\]

Con s(tau)=x_0, h(tau)=1{x_0>=a}, a=Phi^{-1}(0.999), si ha x_0~N(0,1) e p=P(h)=0.001. I target sono

\[
\gamma_k(d\tau)=e^{\beta_k s(\tau)}P(d\tau),\quad
Z_k=e^{\beta_k^2/2},\quad \pi_k=\gamma_k/Z_k.
\]

Pertanto pi_k(A)=1-Phi(a-beta_k), che è diversa dalla probabilità originale p. Conservando (x_T,...,x_d) e rigenerando il suffisso nativo, il rapporto MH è exp{beta_k[s(tau')-s(tau)]}. Il kernel K_k, comprensivo degli sweep, conserva pi_k. Questo non rende indipendenti le particelle dopo resampling né garantisce equilibrio dopo un singolo sweep.

## Ricorrenza dei pesi e del normalizzatore

All'inizio w_0,i=1/N e Zhat_0=1. Prima della mutazione al livello k:

\[
g_{k,i}=G_k(\tau_{k-1,i})=e^{(\beta_k-\beta_{k-1})s(\tau_{k-1,i})},
\quad c_k=\sum_i w_{k-1,i}g_{k,i},
\]
\[
\boxed{\widehat Z_k=\widehat Z_{k-1}c_k,\qquad
\widetilde w_{k,i}=w_{k-1,i}g_{k,i}/c_k.}
\]

ESS_k=1/sum_i tilde(w)_i^2 viene calcolato **dopo** questo aggiornamento. La decisione è J_k=1{ESS_k<0.5N, k<R}.

- Se J_k=0, si conservano gli stati e si pone w_k,i=tilde(w)_k,i.
- Se J_k=1, si estraggono antenati systematic secondo tilde(w), si copiano intere traiettorie e root IDs, si pone w_k,i=1/N.
- In entrambi i casi si applica poi K_k indipendentemente a ciascuna particella selezionata, conservandone il peso.

Zhat_k non è azzerato quando si resampla. Ogni c_k entra nel prodotto, compresi i livelli senza resampling e quello finale. Si usa la scala logaritmica:

\[
\log c_k=\operatorname{logsumexp}_i[\log w_{k-1,i}+\log g_{k,i}],
\quad\log\widehat Z_k=\log\widehat Z_{k-1}+\log c_k.
\]

Senza resampling, Zhat_R=N^{-1}sum_i W_R(H_i), con W_R prodotto dei potenziali valutati prima delle mutazioni. Dopo resampling questa rappresentazione come media di N storie indipendenti **non vale**. La ricorrenza della popolazione è la definizione utilizzata.

## Stimatori

Scriviamo la misura empirica non normalizzata della popolazione

\[
\widehat\Gamma_k^N(f)=\widehat Z_k\sum_iw_{k,i}f(\tau_{k,i}).
\]

Con f_A=e^{-lambda s}h e f_1=e^{-lambda s}, definiamo

\[
U_N=\widehat\Gamma_R^N(f_A),\qquad V_N=\widehat\Gamma_R^N(f_1).
\]

Gli stimatori richiesti sono

\[
\boxed{\widehat p_{\rm ord}=U_N
=\widehat Z_R\sum_iw_{R,i}e^{-\lambda s_i}h_i,}
\]
\[
\boxed{\widehat p_{\rm SN}=U_N/V_N
=\frac{\sum_iw_{R,i}e^{-\lambda s_i}h_i}{\sum_iw_{R,i}e^{-\lambda s_i}}.}
\]

Il fattore Zhat cancella solo nel secondo. Il primo è uno stimatore SMC non normalizzato; chiamarlo IS ordinario non autorizza a trattare le sue particelle come osservazioni IID. V_N è chiamato `reverse_mass` nel codice e ha media esatta 1.

## Perché resampling adattivo systematic conserva la non distorsione

Condizioniamo sulla popolazione al livello k-1, comprendendo tutti i pesi e Zhat. Indichiamo questa informazione con F_{k-1}. Tutti i g_i, c_k, tilde(w)_i e la decisione J_k sono allora noti.

Sia O_i il numero di discendenti del genitore i quando J_k=1. Per systematic,

\[
E[O_i\mid\mathcal F_{k-1}]=N\widetilde w_i.
\]

La proprietà riguarda i conteggi, non l'indipendenza dei discendenti. Per verificarla, sia F_i=sum_{j<=i}tilde(w)_j e U~Unif(0,1); i punti sono (U+j)/N, j=0,...,N-1. Integrando il numero di punti nell'intervallo [F_{i-1},F_i) rispetto a U si ottiene la lunghezza N(F_i-F_{i-1})=N tilde(w)_i.

Se non si resampla,

\[
E[\widehat\Gamma_k^N(f)\mid\mathcal F_{k-1}]
=\widehat Z_k\sum_i\widetilde w_i K_kf(\tau_{k-1,i}).
\]

Se si resampla, prima si condiziona anche sui conteggi O e poi si integra:

\[
\begin{aligned}
E[\widehat\Gamma_k^N(f)\mid\mathcal F_{k-1}]
&=\widehat Z_k E\left[\frac1N\sum_iO_iK_kf(\tau_{k-1,i})\mid\mathcal F_{k-1}\right]\\
&=\widehat Z_k\sum_i\widetilde w_i K_kf(\tau_{k-1,i}).
\end{aligned}
\]

Le due espressioni coincidono. Definendo Q_k f=G_k K_k f,

\[
E[\widehat\Gamma_k^N(f)\mid\mathcal F_{k-1}]
=\widehat\Gamma_{k-1}^N(Q_kf).
\]

Poiché gamma_{k-1}Q_k=gamma_k per l'invarianza di K_k, l'induzione, con particelle iniziali IID da P, dà

\[
E\widehat\Gamma_k^N(f)=\gamma_k(f).
\]

In particolare, per ogni N finito,

\[
\boxed{E\widehat Z_R=Z_R,\quad E U_N=p,\quad E V_N=1.}
\]

La decisione ESS, essendo determinata prima dell'estrazione, non invalida questo argomento. L'ordine input del systematic può cambiare la varianza, ma non questa identità. Non si conclude E(U_N/V_N)=p.

## Decomposizione esatta della varianza per la politica effettiva

Questa è una formula a N finito, non una CLT importata da un'altra politica. Fissata una funzione terminale f_R, costruiamo all'indietro

\[
f_{k-1}=Q_k f_k=G_kK_kf_k,\qquad k=R,\ldots,1.
\]

Definiamo M_k=Gammahat_k^N(f_k). L'identità condizionata precedente implica E[M_k|F_{k-1}]=M_{k-1}. Quindi

\[
M_R-E M_R=(M_0-E M_0)+\sum_{k=1}^R(M_k-M_{k-1}).
\]

Per j<k, l'ortogonalità degli incrementi segue da

\[
E[(M_j-M_{j-1})(M_k-M_{k-1})]
=E[(M_j-M_{j-1})E(M_k-M_{k-1}\mid F_{k-1})]=0.
\]

Anche il termine iniziale è ortogonale agli incrementi successivi. Pertanto

\[
\boxed{\operatorname{Var}(\widehat\Gamma_R^N(f_R))
=\frac{\operatorname{Var}_P(f_0)}{N}
+\sum_{k=1}^R E[\mathcal V_k(f_k)].}
\]

Qui Vcal_k è la varianza condizionata dell'aggiornamento del livello, che ora calcoliamo. Condizionatamente a F_{k-1}, poniamo z=Zhat_k, w_i=tilde(w)_k,i,

\[
m_i=K_kf_k(\tau_{k-1,i}),\qquad
v_i=K_k(f_k^2)(\tau_{k-1,i})-m_i^2.
\]

**Ramo senza resampling.** Le mutazioni sono indipendenti condizionatamente agli stati e ai pesi. Dunque

\[
\boxed{\mathcal V_k^{\rm no}(f_k)=z^2\sum_iw_i^2v_i.}
\]

**Ramo con systematic.** Condizionando prima sui conteggi,

\[
E[M_k\mid O,F_{k-1}]=\frac zN\sum_i O_i m_i,
\quad
\operatorname{Var}(M_k\mid O,F_{k-1})=\frac{z^2}{N^2}\sum_iO_i v_i.
\]

La legge della varianza totale dà

\[
\boxed{\mathcal V_k^{\rm sys}(f_k)
=z^2\left[\frac1N\sum_iw_iv_i
+\frac1{N^2}\sum_{i,j}\operatorname{Cov}_{\rm sys}(O_i,O_j\mid F_{k-1})m_i m_j\right].}
\]

Nella formula complessiva si inserisce

\[
\mathcal V_k=(1-J_k)\mathcal V_k^{\rm no}+J_k\mathcal V_k^{\rm sys},\qquad J_R=0.
\]

La covarianza systematic è definibile esattamente con un integrale monodimensionale. Se O_i(u) sono i conteggi deterministici prodotti dall'offset u,

\[
\operatorname{Cov}_{\rm sys}(O_i,O_j\mid F_{k-1})
=\int_0^1[O_i(u)-Nw_i][O_j(u)-Nw_j]du.
\]

I conteggi sono costanti a tratti; per piccole popolazioni questo integrale si calcola esattamente suddividendo ai punti frac(NF_i). Uno dei test verifica media dei conteggi e formula di varianza condizionata con questa integrazione, e verifica che la matrice non coincida con quella multinomiale.

Per lo stimatore ordinario si prende f_R=f_A. Si ottiene così una formula esatta della sua varianza che incorpora selezione, genealogia e mutazione. Per la varianza di Zhat si prende f_R=1. Non valutiamo numericamente l'intera formula per N=2048: richiederebbe le funzioni backward e le aspettative sulla popolazione, non soltanto momenti di singole traiettorie. Le varianze riportate sono stimate da run complete indipendenti.

## SNIS: cosa è esatto e cosa è soltanto condizionale a una CLT

L'identità esatta è

\[
\widehat p_{\rm SN}-p=\frac{U_N-pV_N}{V_N}.
\]

La decomposizione martingala precedente si può applicare anche a f_R=e^{-lambda s}(h-p). Ne risulta una formula esatta per Var(U_N-pV_N), **non** per Var[(U_N-pV_N)/V_N]. In generale numeratore e denominatore sono dipendenti. Il bias esatto è E[(U_N-pV_N)/V_N], mentre E(U_N-pV_N)=0.

Se si dimostrasse per questa precisa politica una CLT congiunta

\[
\sqrt N\begin{pmatrix}U_N-p\\V_N-1\end{pmatrix}
\Rightarrow\mathcal N(0,\Sigma_{\rm policy}),
\]

allora il metodo delta darebbe coefficienti Sigma_11 per l'ordinario e (1,-p)Sigma(1,-p)^T per SNIS. Questa è un'implicazione condizionale: qui **non dimostriamo tale CLT per systematic non ordinato con soglia ESS fissa**, e non usiamo le formule IID della baseline per predire le nuove varianze. Analogamente, la consistenza SNIS segue se U_N->p e V_N->1; l'unbiasedness a N finito da sola non prova queste convergenze.

Il notebook mostra Var(U_N-pV_N) stimata tra run complete come diagnostica della linearizzazione del rapporto. È chiaramente etichettata come tale: non è un coefficiente CLT dimostrato né una predizione indipendente della varianza SNIS.

## Politiche diverse e limiti dei risultati di letteratura

Se al posto di systematic si usasse multinomial, la covarianza condizionata dei conteggi sarebbe N[diag(w)-ww^T], e il termine con resampling si ridurrebbe a

\[
\mathcal V_k^{\rm multi}=\frac{z^2}{N}\left[\sum_iw_iK_k(f_k^2)(\tau_i)
-\left(\sum_iw_iK_kf_k(\tau_i)\right)^2\right].
\]

Questa semplificazione riguarda **un altro resampler**, non viene applicata al codice o ai risultati. Le formule CLT classiche presentate da Del Moral, Doucet e Jasra (2006, sezione 3.3.2) considerano esplicitamente resampling multinomiale nei casi mai/sempre. [Articolo originale](https://www.cs.ubc.ca/~arnaud/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).

Per il resampling adattivo, Del Moral, Doucet e Jasra (2012) sviluppano risultati tramite una politica limite; la coincidenza del criterio limite con la soglia è un caso delicato. Le ipotesi su schema di selezione, separazione dalla soglia e funzioni non limitate vanno verificate: non basta usare ESS per invocare il teorema. [Articolo originale](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_adaptiveresamplingSMC.pdf).

Gerber, Chopin e Whiteley mostrano che la convergenza del systematic può dipendere dall'ordine degli input e che non esiste una garanzia universale per ordinamenti arbitrari. Questo non dimostra che il toy implementato sia inconsistente; impedisce però di trasferire automaticamente un teorema per altri resampler. [Articolo originale](https://arxiv.org/abs/1707.01845).

## Inferenza legittima tra run complete

Con N fissato, B esecuzioni indipendenti producono B stime IID Y_b dell'intero algoritmo. Quindi media, varianza campionaria e RMSE tra queste run sono strumenti validi senza supporre indipendenza delle particelle interne:

\[
\bar Y=B^{-1}\sum_bY_b,\quad
s_Y^2=(B-1)^{-1}\sum_b(Y_b-\bar Y)^2,\quad
\widehat{\rm RMSE}=\sqrt{B^{-1}\sum_b(Y_b-p)^2}.
\]

SE della media è s_Y/sqrt(B). Una CLT in **B**, a N fissato, riguarda la media delle run e ha centro E(Y_b): per l'ordinario è p, per SNIS può differire da p. L'errore Monte Carlo della varianza viene stimato dalla dispersione dei quadrati centrati tra run. Queste approssimazioni possono risentire di code pesanti e non sono intervalli simultanei.

Il codice registra per ogni run ESS a tutti i livelli, decisioni di resampling, log-incrementi, Zhat, reverse_mass, transizioni gaussiane incluse le proposal rifiutate, tempo e antenati distinti. Il confronto principale resta a N uguale; il prodotto varianza per costo in transizioni è un indicatore aggiuntivo, non un esperimento a budget abbinato.
