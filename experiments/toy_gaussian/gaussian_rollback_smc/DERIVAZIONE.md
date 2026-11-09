# RollBack SMC senza resampling: probabilità originale e varianze

Questa derivazione fissa l'ordine **peso prima, mutazione dopo** e non introduce resampling. L'asintotica è nel numero N di particelle, con T, R, schedule, legge delle profondità e numero di sweep fissati. Nessun parametro viene adattato usando la popolazione. La notazione W indica sempre un peso **non normalizzato**; i pesi normalizzati si chiamano w.

## 1. Legge originale del modello e target tilted

Una particella è una traiettoria completa del modello

\[
\tau=(x_T,x_{T-1},\ldots,x_0),\qquad x_T=0,
\]

con legge

\[
P(d\tau)=\delta_0(dx_T)Q_T(dx_{T-1}\mid x_T)\cdots Q_1(dx_0\mid x_1),
\quad Q_t(\cdot\mid x_t)=\mathcal N(x_t,1/T).
\]

Ponendo \(\epsilon_t=x_{t-1}-x_t\), le T innovazioni sono indipendenti, con media zero e varianza 1/T. Dunque

\[
x_0=\sum_{t=1}^T\epsilon_t\sim\mathcal N(0,T/T)=\mathcal N(0,1).
\]

Definiamo \(s(\tau)=x_0\), \(a=\Phi^{-1}(0.999)\), \(h(\tau)=\mathbf1\{s(\tau)\ge a\}\), e

\[
p=P(h)=1-\Phi(a)=0.001.
\]

Per \(0=\beta_0<\cdots<\beta_R=\lambda\),

\[
\gamma_k(d\tau)=e^{\beta_k s(\tau)}P(d\tau),\quad
Z_k=\gamma_k(1)=e^{\beta_k^2/2},\quad
\pi_k(d\tau)=\gamma_k(d\tau)/Z_k.
\]

La formula per Z deriva dalla funzione generatrice dei momenti della normale, oppure completando il quadrato:

\[
e^{\beta x}e^{-x^2/2}=e^{\beta^2/2}e^{-(x-\beta)^2/2}.
\]

Quindi sotto \(\pi_k\), \(x_0\sim\mathcal N(\beta_k,1)\), e

\[
\pi_k(A)=\overline\Phi(a-\beta_k)\ne p\quad(\beta_k>0).
\]

Inoltre il tilt si fattorizza sulle innovazioni: sotto \(\pi_k\) esse sono indipendenti \(\mathcal N(\beta_k/T,1/T)\). Questa rappresentazione viene usata **soltanto nel test di invarianza**, non per generare le particelle SMC.

## 2. Proposal di RollBack e kernel MH

Scegliamo \(d\sim\rho_D=\mathrm{Unif}\{1,\ldots,D\}\), indipendentemente da traiettoria e pesi. Conserviamo il prefisso \(u=(x_T,\ldots,x_d)\) e rigeneriamo il suffisso \(v'=(x'_{d-1},\ldots,x'_0)\) con Q_d,...,Q_1. Per profondità fissata,

\[
q_d(d\tau'\mid\tau)=\delta_u(du')P(dv'\mid u).
\]

Per due traiettorie con lo stesso prefisso, scrivendo le densità dei suffissi,

\[
\frac{\pi_k(\tau')\rho_D(d)q_d(\tau\mid\tau')}
{\pi_k(\tau)\rho_D(d)q_d(\tau'\mid\tau)}
=\frac{P(u)P(v'\mid u)e^{\beta_k s(\tau')}P(v\mid u)}
{P(u)P(v\mid u)e^{\beta_k s(\tau)}P(v'\mid u)}
=e^{\beta_k[s(\tau')-s(\tau)]}.
\]

La probabilità di accettazione è quindi

\[
\alpha_{k,d}(\tau,\tau')=\min\{1,e^{\beta_k[s(\tau')-s(\tau)]}\}.
\]

Ogni kernel a profondità fissata soddisfa il bilancio dettagliato rispetto a \(\pi_k\). La loro miscela con coefficienti \(\rho_D\) indipendenti dallo stato è anch'essa reversibile. Indichiamo questa miscela con \(M_k\) e la composizione di m sweep con \(K_k=M_k^m\). Pertanto

\[
\int\pi_k(d\tau)K_k(\tau,d\tau')=\pi_k(d\tau').
\]

Al rifiuto si conserva tutta la vecchia traiettoria. I pesi già assegnati non vengono ricalcolati sullo stato proposto o accettato. Non si assume equilibrio dopo un numero finito di sweep.

## 3. Legge forward dei percorsi delle particelle

Il pedice k seguente indica un livello SMC, non il tempo t del modello. Ogni \(\tau_k\) è ancora un'intera traiettoria con T transizioni. La storia di una particella è

\[
H_R=(\tau_0,\ldots,\tau_R).
\]

L'algoritmo genera \(\tau_0\sim P\), poi \(\tau_k\mid\tau_{k-1}\sim K_k\). La sua legge forward, inclusi implicitamente profondità, proposal e rifiuti, è

\[
\boxed{\eta_R(dH_R)=P(d\tau_0)\prod_{k=1}^R K_k(\tau_{k-1},d\tau_k).}
\]

Si tratta di una formula in termini di kernel di probabilità: i rifiuti MH introducono masse atomiche, quindi non si presume una densità rispetto alla sola misura di Lebesgue.

Le N storie sono IID sotto \(\eta_R\). Gli stati entro una storia sono dipendenti. Il marginale forward di \(\tau_R\), \(P K_1\cdots K_R\), non è generalmente \(\pi_R\).

Il potenziale incrementale e il peso cumulativo sono

\[
G_k(\tau)=\frac{d\gamma_k}{d\gamma_{k-1}}(\tau)
=e^{(\beta_k-\beta_{k-1})s(\tau)},
\]
\[
\boxed{W_R(H_R)=\prod_{k=1}^R G_k(\tau_{k-1})
=\exp\left\{\sum_{k=1}^R(\beta_k-\beta_{k-1})s(\tau_{k-1})\right\}.}
\]

Ogni fattore è valutato **prima della rispettiva mutazione**. In generale

\[
W_R(H_R)\ne e^{\lambda s(\tau_R)}.
\]

La normale normalizzazione numerica \(w_{k,i}=W_{k,i}/\sum_j W_{k,j}\) non cambia la legge delle storie, dato che non avvengono selezione o adattamento. I pesi normalizzati, però, hanno un denominatore comune e non sono indipendenti.

## 4. Identità Feynman–Kac e correzione inversa

Dimostriamo per induzione, per f integrabile,

\[
\mathbb E_{\eta_k}[W_k f(\tau_k)]=\gamma_k(f).
\]

Per k=0, \(W_0=1\), \(\eta_0=P=\gamma_0\): l'identità è immediata. Supponiamola vera a k-1. Condizionando sulla storia precedente,

\[
\begin{aligned}
\mathbb E_{\eta_k}[W_k f(\tau_k)]
&=\mathbb E_{\eta_{k-1}}[W_{k-1}G_k(\tau_{k-1})(K_k f)(\tau_{k-1})]\\
&=\gamma_{k-1}(G_k K_kf)\\
&=\gamma_k(K_k f)\\
&=Z_k\pi_k(K_k f)\\
&=Z_k\pi_k(f)=\gamma_k(f).
\end{aligned}
\]

La seconda uguaglianza usa l'ipotesi induttiva, la terza la definizione di G_k e la penultima l'invarianza di K_k. In particolare \(\mathbb E_{\eta_R}W_R=Z_R\).

Definiamo

\[
C(H_R)=W_R(H_R)e^{-\lambda s(\tau_R)},\quad
U(H_R)=C(H_R)h(\tau_R),\quad V(H_R)=C(H_R).
\]

Applicando l'identità appena provata a \(f(\tau)=e^{-\lambda s(\tau)}g(\tau)\),

\[
\begin{aligned}
\mathbb E_{\eta_R}[C g(\tau_R)]
&=\gamma_R(e^{-\lambda s}g)\\
&=\int P(d\tau)e^{\lambda s(\tau)}e^{-\lambda s(\tau)}g(\tau)\\
&=P(g).
\end{aligned}
\]

Prendendo g=1 e g=h si ottiene

\[
\boxed{\mathbb E V=1,\qquad \mathbb E U=p.}
\]

La misura estesa \(\nu_R(dH_R)=C(H_R)\eta_R(dH_R)\) è quindi una probabilità, con marginale finale P; C è un vero rapporto di Radon–Nikodym sullo spazio delle storie. Il rapporto finale \(P/\pi_R=Z_R e^{-\lambda s}\), invece, non è da solo il rapporto rispetto alla legge forward effettiva.

## 5. Stimatore IS ordinario

Per N storie indipendenti:

\[
\boxed{\widehat p_{\mathrm{ord}}=\overline U_N
=\frac1N\sum_{i=1}^N W_R(H_i)e^{-\lambda s(\tau_{R,i})}h(\tau_{R,i}).}
\]

È non distorto per ogni N. Non si divide per la somma dei pesi C. Potrebbe superare 1: non è vincolato a essere una proporzione campionaria.

Poiché \(\widehat Z_R=N^{-1}\sum_iW_R(H_i)\), si può riscrivere

\[
\widehat p_{\mathrm{ord}}=\widehat Z_R\sum_i w_{R,i}e^{-\lambda s(\tau_{R,i})}h(\tau_{R,i}).
\]

La stima usuale di Z tramite incrementi coincide qui con questa media. Infatti, posto \(S_k=\sum_iW_{k,i}\),

\[
\sum_i w_{k-1,i}G_k(\tau_{k-1,i})=S_k/S_{k-1},\quad
\prod_{k=1}^R S_k/S_{k-1}=S_R/N.
\]

Sostituire \(\widehat Z_R\) con il noto Z_R mantenendo i w normalizzati produrrebbe uno stimatore diverso, che non implementiamo.

Per indipendenza delle storie,

\[
\begin{aligned}
\operatorname{Var}(\widehat p_{\mathrm{ord}})
&=N^{-2}\sum_{i=1}^N\operatorname{Var}(U_i)\\
&=\frac{\sigma_{\mathrm{ord}}^2}{N},\\
\sigma_{\mathrm{ord}}^2
&=\mathbb E_{\eta_R}[U^2]-(\mathbb EU)^2\\
&=\boxed{\mathbb E_{\eta_R}[W_R^2e^{-2\lambda s(\tau_R)}h(\tau_R)]-p^2}.
\end{aligned}
\]

Abbiamo usato \(h^2=h\). Questa formula della varianza della media è **esatta a N finito**. La CLT aggiunge

\[
\sqrt N(\widehat p_{\mathrm{ord}}-p)\Rightarrow\mathcal N(0,\sigma_{\mathrm{ord}}^2).
\]

## 6. Stimatore self-normalized e CLT per il rapporto

Definiamo

\[
\boxed{\widehat p_{\mathrm{SN}}=\overline U_N/\overline V_N
=\frac{\sum_iW_{R,i}e^{-\lambda s_i}h_i}{\sum_iW_{R,i}e^{-\lambda s_i}}.}
\]

Il denominatore è strettamente positivo. Per la legge dei grandi numeri, \((\overline U_N,\overline V_N)\to(p,1)\), quindi il rapporto è consistente. Non è in generale non distorto a N finito.

La CLT bivariata è

\[
\sqrt N\left[\begin{pmatrix}\overline U_N\\\overline V_N\end{pmatrix}
-\begin{pmatrix}p\\1\end{pmatrix}\right]
\Rightarrow\mathcal N(0,\Sigma),\quad
\Sigma=\begin{pmatrix}\operatorname{Var}U&\operatorname{Cov}(U,V)\\
\operatorname{Cov}(U,V)&\operatorname{Var}V\end{pmatrix}.
\]

Per \(g(u,v)=u/v\), \(\nabla g(p,1)=(1,-p)\). In modo equivalente, espandendo esplicitamente il rapporto,

\[
\sqrt N(\widehat p_{\mathrm{SN}}-p)
=\frac{\sqrt N[(\overline U_N-p)-p(\overline V_N-1)]}{\overline V_N}.
\]

Il denominatore converge a 1. Il numeratore ha varianza limite

\[
\begin{aligned}
\sigma_{\mathrm{SN}}^2
&=(1,-p)\Sigma(1,-p)^\top\\
&=\operatorname{Var}U+p^2\operatorname{Var}V-2p\operatorname{Cov}(U,V)\\
&=\operatorname{Var}(U-pV).
\end{aligned}
\]

Poiché \(\mathbb E(U-pV)=p-p=0\), segue

\[
\boxed{\sigma_{\mathrm{SN}}^2
=\mathbb E_{\eta_R}[W_R^2e^{-2\lambda s(\tau_R)}(h(\tau_R)-p)^2].}
\]

La CLT è \(\sqrt N(\widehat p_{\mathrm{SN}}-p)\Rightarrow\mathcal N(0,\sigma_{\mathrm{SN}}^2)\). Il confronto \(\operatorname{Var}(\widehat p_{\mathrm{SN}})\simeq\sigma_{\mathrm{SN}}^2/N\) è asintotico, non un'identità a N finito. Nel modello fissato tutti i momenti necessari sono finiti: le traiettorie accettate sono selezioni da un numero finito di proposte costruite con innovazioni gaussiane, e gli esponenziali coinvolti sono dominabili da esponenziali di somme finite di valori assoluti gaussiani. Anche i momenti inversi di C sono finiti. Ciò consente il controllo dei resti del rapporto; non garantisce che una dimensione N pratica abbia già raggiunto il regime asintotico.

La covarianza U,V non va eliminata. SNIS non ha una varianza universalmente minore dello stimatore ordinario. Entrambi sono zero esattamente quando nessuno stato finale cade in A.

## 7. Come valutare i secondi momenti della vera legge forward

Non sostituiamo \(\eta_R\) con campioni da \(\pi_R\). Per una funzione terminale f,

\[
\mathbb E_{\eta_R}[W_R^2 f(\tau_R)]
=P\bigl(G_1^2K_1[G_2^2K_2[\cdots G_R^2K_Rf]]\bigr).
\]

Questa formula segue integrando la definizione di eta e raccogliendo un fattore G_k^2 prima di ciascun kernel. Equivalentemente, si pone \(b_R=f\), poi

\[
b_{k-1}(\tau)=G_k(\tau)^2\int K_k(\tau,d\tau')b_k(\tau'),
\quad k=R,\ldots,1,
\]

e si calcola P(b_0). Per IS ordinario si usa \(f=e^{-2\lambda s}h\) e si sottrae p²; per SNIS si usa \(f=e^{-2\lambda s}(h-p)^2\).

Nell'esperimento stimiamo questi stessi momenti mediante storie RollBack indipendenti in un pilot separato:

\[
\widehat\sigma^2_{\mathrm{ord,pilot}}=\frac1L\sum_{i=1}^L U_i^2-p^2,
\qquad
\widehat\sigma^2_{\mathrm{SN,pilot}}=\frac1L\sum_{i=1}^L(U_i-pV_i)^2.
\]

Il p noto consente di evitare una seconda stima del centro. L'errore Monte Carlo di ciascun secondo momento è la deviazione standard campionaria dei rispettivi quadrati divisa per sqrt(L). Si riportano anche differenza tra due metà del pilot, quota del massimo contributo e quota dello 0.1% dei contributi più grandi. Questi sono controlli empirici: nessun pilot finito esclude code non ancora osservate.

## 8. Esperimento ideale distinto: campioni IID da pi_lambda

Se, in un altro esperimento, \(\tau_i\stackrel{IID}{\sim}\pi_\lambda\), il rapporto corretto sarebbe \(r(\tau)=Z_\lambda e^{-\lambda s(\tau)}\). Allora

\[
\widehat p_{\mathrm{ord,ideal}}=N^{-1}\sum_i r_i h_i,\qquad
\widehat p_{\mathrm{SN,ideal}}=\frac{\sum_i r_i h_i}{\sum_i r_i}.
\]

Per qualunque f,

\[
\mathbb E_{\pi_\lambda}[r^2 f]
=\int P(d\tau)\frac{e^{\lambda s}}{Z_\lambda}Z_\lambda^2e^{-2\lambda s}f
=Z_\lambda P(e^{-\lambda s}f).
\]

Inoltre, completando il quadrato,

\[
P(e^{-\lambda x_0}h)
=\int_a^\infty\frac{e^{-x^2/2-\lambda x}}{\sqrt{2\pi}}dx
=e^{\lambda^2/2}\int_a^\infty\frac{e^{-(x+\lambda)^2/2}}{\sqrt{2\pi}}dx
=e^{\lambda^2/2}\overline\Phi(a+\lambda).
\]

Si ottiene quindi

\[
\boxed{\sigma^2_{\mathrm{ord,ideal}}=e^{\lambda^2}\overline\Phi(a+\lambda)-p^2.}
\]

Poiché \((h-p)^2=(1-2p)h+p^2\) e \(P(e^{-\lambda x_0})=e^{\lambda^2/2}\),

\[
\boxed{\sigma^2_{\mathrm{SN,ideal}}=e^{\lambda^2}\bigl[(1-2p)\overline\Phi(a+\lambda)+p^2\bigr].}
\]

Queste due curve sono riferimenti analitici per un **altro sampler**, non predizioni del RollBack effettivamente eseguito. Nei confronti empirici non vengono generate particelle ideali in sostituzione di quelle RollBack.

## 9. Crude Monte Carlo e controlli limite

Con \(\tau_i\stackrel{IID}{\sim}P\), h_i è Bernoulli(p). Pertanto

\[
\mathbb Eh_i=p,\quad\mathbb Eh_i^2=p,\quad
\boxed{\sigma^2_{\mathrm{MC}}=\operatorname{Var}(h_i)=p-p^2=p(1-p)=0.000999.}
\]

Per la media di N indicatori,

\[
\operatorname{Var}(\widehat p_{\mathrm{MC}})=p(1-p)/N,
\quad \Pr(\widehat p_{\mathrm{MC}}=0)=(1-p)^N.
\]

Non si deve confondere il coefficiente della CLT p(1-p) con la varianza della stima, che contiene 1/N.

Se tutti i kernel sono identità, la traiettoria non cambia e W_R=e^{lambda s(tau_0)}, quindi C=1: i due stimatori coincidono esattamente con crude MC. Se lambda=0, tutti i pesi sono 1, i kernel conservano P e gli stati finali sono ancora IID da P: vale lo stesso risultato distributivo, pur sostenendo il costo delle mutazioni.

Se D<T, alcuni stati iniziali della traiettoria non possono mai essere modificati. L'invarianza del kernel rimane valida, ma non vi è esplorazione dell'intera traiettoria tramite quel kernel. L'identità pesata e la CLT tra storie indipendenti non richiedono ergodicità della singola catena a livello fissato; efficienza e stabilità dei pesi possono comunque peggiorare.

## 10. Portata rispetto al paper

Lo stimatore finale del paper locale `paper/ai4good_rare_event_estimation.tex` è il rapporto SNIS con pesi terminali. Qui si studia un singolo tilt, si sopprime il resampling e si aggiunge IS ordinario come confronto. Le varianze qui derivate non sono applicabili alla popolazione con resampling, i cui lignaggi sono dipendenti. Non vengono implementati resampling, MBAR, waste-free SMC o altri algoritmi.
