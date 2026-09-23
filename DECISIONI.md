# Decisioni di progetto e perché

Note sulle scelte non ovvie, con le misure che le hanno motivate. Servono a non rifare gli
stessi errori: quasi tutte nascono da un bug che non dava errore, solo un training piatto.

## Osservazione a 6 canali, non 4

I canali 4 e 5 (scarto riga/colonna verso il target, costanti su tutta la mappa) sembrano
ridondanti rispetto ai canali 2 e 3, che già contengono posizione dell'agente e del target.
Non lo sono: lì l'informazione è codificata come **due singoli pixel accesi** in una griglia
20×20, e il tronco convoluzionale scende a 3×3 prima della dense layer. A quella risoluzione i
due pixel cadono quasi sempre nella stessa cella.

Misurato clonando una policy greedy e valutandola su due distribuzioni di stati:

| | 4 canali | 6 canali |
|---|---|---|
| accuratezza su stati visitati dalla greedy | 100,0% | 99,9% |
| accuratezza su stati di una policy casuale | **15,7%** (caso: 25%) | 43,6% |
| training, successo dal primo all'ultimo decile | 17,9% → 21,1% | **38,1% → 82,2%** |

Con 4 canali la rete non usava affatto la posizione dell'agente (azzerando il canale 2
l'accuratezza scendeva solo da 100% a 97,8%): predice l'azione dalla posizione **assoluta** del
target, scorciatoia valida lungo un avvicinamento rettilineo e sotto il livello del caso
ovunque altrove — cioè proprio dove PPO passa il tempo, visto che esplora campionando.

Se cambi l'architettura, ricontrolla questo punto: è il singolo fattore che separa un training
piatto da uno che impara.

## GroupNorm, non BatchNorm

`BatchNorm` rende l'uscita della rete dipendente da *quali altri campioni* stanno nel batch. In
RL on-policy lo stesso stato viene valutato in due batch diversi — 32 ambienti paralleli durante
la raccolta, 512 campioni mescolati durante l'update — quindi le probabilità e i valori salvati
non corrispondono a quelli che l'update ricalcola.

Misurato sugli stessi 32 stati, batch 32 contro batch 512:

| | BatchNorm | GroupNorm |
|---|---|---|
| rapporto di importanza PPO, dev.std | 0,516 | **0,000** |
| valore V | −0,427 vs +0,282 | identico |

Con dev.std 0,516 il rapporto PPO satura il clipping per puro rumore di normalizzazione, e il
critic insegue un bersaglio che cambia segno a seconda del batch.

## Reward: shaping potenziale denso, non ratchet

La versione precedente pagava un bonus solo quando l'agente batteva la sua distanza minima di
sempre. Risultato misurato sui dati di training: **98,9% dei passi riceveva esattamente lo
stesso reward**, e l'ultimo bonus arrivava in media al 71% dell'episodio. Con episodi di ~900
passi contro un orizzonte di credito GAE di ~20 passi (`1/(1-gamma*lambda)`), il segnale non
raggiungeva mai l'inizio dell'episodio.

Sostituito con shaping potenziale `gamma*PHI(s') - PHI(s)` con `PHI = -distanza`, che è denso e
lascia invariata la policy ottima. Correlazione fra "l'azione avvicina al target" e il vantaggio
stimato: da **+0,059 a +0,329**.

## c1 = 0.05, non 0.5

Policy e critic condividono lo stesso tronco convoluzionale. Con i ritorni densi la MSE del
valore parte da ~9,5 contro un `actor_loss` di 0,055: `c1 * critic_loss` pesava **86 volte**
l'attore, e la rete veniva ottimizzata quasi solo per predire il valore. È esposto nella config
del curriculum, per step.

## Maschera delle azioni

L'ambiente espone in `info["action_mask"]` quali delle 8 mosse sono percorribili, e i logit
delle altre vengono messi a `-inf`. È applicata in modo coerente in tre punti — campionamento,
update PPO (la maschera è salvata nell'`ExperienceManager`) e validazione — altrimenti il
rapporto di importanza confronterebbe distribuzioni diverse.

Senza la maschera, una policy deterministica sceglieva una direzione bloccata, restava ferma su
un'osservazione immutata e ripeteva la stessa scelta all'infinito: **89,3% dei passi senza
movimento**, ora 0%.

## Gli stati salvati sono s(t), non s(t+1)

In `Agent.train()` la variabile `observations` viene riassegnata da `environments.step()`. La
versione precedente salvava nel buffer il valore **dopo** lo step, quindi l'update PPO calcolava
`pi_new(a_t | s_t+1) / pi_old(a_t | s_t)`: due distribuzioni su stati diversi. Misurato al primo
minibatch, prima di qualsiasi update: rapporto medio 0,653 con dev.std 0,453 e **52,3% dei
campioni fuori dalla finestra di clipping**, contro 1,001 ± 0,071 dopo la correzione.

## Difficoltà: i limiti del rover, non la quantità di DTM

I DTM marziani a 1 m/px sono quasi tutti pianeggianti rispetto a un rover che scala un metro.
Con `max_step = 1` solo lo 0,8% delle mosse è bloccato e una rete non addestrata risolve già il
~70% dei casi: non c'è nulla da imparare. A **0,3 m** il 15,1% delle mosse è bloccato e il 89,6%
della mappa resta raggiungibile.

Aggiungere DTM non tocca questo: fra i 15 DTM scaricati a settembre 2026 la media di mosse
bloccate era 1,8% contro l'1,0% dei precedenti, e solo 2 su 15 erano davvero accidentati.

## Split train/test stratificato

I DTM variano dall'1,1% al 59,3% di mosse bloccate a 0,3 m: un ordine di grandezza e mezzo. Il
test set è stato scelto prendendo **un DTM per quartile** della distribuzione di training, non i
più vicini alla mediana, così da replicarne la distribuzione invece di misurare solo il caso
tipico. Prima: training mediana 10,1%, testing 59,3% — due problemi diversi. Dopo: 6,5% contro
6,2%.

Se aggiungi DTM e rifai lo split, ripeti la profilazione: non si può giudicare a occhio.

## Metriche di riferimento

Misurate su mappa 20×20, `max_step = 0.3`, pool di training:

| policy | budget 200 | budget 1341 |
|---|---|---|
| casuale | 19,5% in 178,6 passi | 62,0% in 803,3 passi |
| greedy verso il target (scritta a mano) | 80,2% in 47,8 passi | 80,2% in 273,2 passi |
| cammino ottimo (Dijkstra) | ~10 passi | ~10 passi |

Il **tasso di successo da solo è una metrica debole**: con 1341 passi concessi su una mappa
20×20 (3,3 passi per cella) una passeggiata casuale la copre quasi tutta e arriva al 62%.
Guarda la lunghezza media degli episodi rispetto all'ottimo: il caso impiega ~52 volte il
cammino ottimo, quindi è lì che sta tutto il margine.

## Come verificare che il loop di training sia sano

Se un giorno il training torna piatto, il test che isola l'algoritmo dal task è far girare
`Agent.train()` su CartPole-v1: serve solo un wrapper che aggiunga le chiavi di `info` usate dal
logging e una rete MLP con la stessa firma `forward(x, action_mask=None) -> (probs, value)`. Un
PPO corretto porta CartPole da ~24 a diverse centinaia di passi per episodio. Se CartPole sale e
il rover no, il difetto è nel task, non nell'algoritmo — che è esattamente come è stata trovata
la questione dei 6 canali.
