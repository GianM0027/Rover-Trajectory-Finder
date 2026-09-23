# Rover-Trajectory-Finder

Un rover è atterrato su Marte. Conosce il punto di atterraggio e la destinazione, ma percepisce
solo ciò che ha intorno. L'obiettivo è raggiungere il target minimizzando la lunghezza del
percorso.

Il terreno viene da DTM reali [HiRISE](https://www.uahirise.org/dtm/), semplificati in una griglia
dove ogni pixel è 1 m. L'agente è addestrato con PPO su una rete IMPALA.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu129
.venv\Scripts\pip install -r requirements.txt
```

Il progetto gira su **Python 3.10**. La prima riga installa la build CUDA di torch: `pip install
torch` da PyPI su Windows dà quella CPU-only.

## 1. Costruire il tile pool

I `.IMG` non vengono mai caricati a runtime. `build_tile_pool.py` li scorre a finestre con
`rasterio`, ne estrae tile 64×64 filtrate e le impila in un unico `.npy` memory-mapped che tutti
i worker condividono.

```bash
python build_tile_pool.py --dtm-dir DTMs/training --out tile_pools/tiles_training.npy --n-tiles 50000
python build_tile_pool.py --dtm-dir DTMs/testing  --out tile_pools/tiles_testing.npy  --n-tiles 4000
```

> **Ogni volta che aggiungi, togli o sostituisci un `.IMG` devi rilanciare questo comando.**
> Il training legge il `.npy`, non la cartella `DTMs/`: se il pool non viene rigenerato, i DTM
> nuovi non entrano nell'addestramento e la corsa gira sugli stessi dati di prima senza dare
> alcun errore. Sono 40 secondi.

Questo passo è obbligatorio prima di addestrare o validare.

Conseguenza pratica: i `.IMG` sono un **input di build, non una dipendenza di runtime**. Per
aumentare la diversità del dataset scarichi un lotto di DTM, ci costruisci il pool, cancelli i
`.IMG` e ripeti. La memoria a runtime dipende solo da `--n-tiles`, mai da quanti DTM sono stati
scansionati.

Opzioni utili:

| flag | default | effetto |
|---|---|---|
| `--n-tiles` | 30000 | dimensione del pool (30k tile 64×64 = 492 MB) |
| `--tile-size` | 64 | deve coprire `map_size + 2·fov_distance`; 64 regge `map_size` fino a ~46 |
| `--max-nodata` | 0.0 | frazione massima di pixel nodata ammessa in una tile |
| `--min-relief` | 0.05 | scarta le tile con deviazione standard di quota sotto questa soglia, in metri |

Le tile sono prese da un reticolo non sovrapposto con offset casuale, con quota uguale per DTM
così che un raster grande non domini il pool. `tiles_*_meta.json` registra la provenienza di ogni
tile, quindi il contenuto di un training è riproducibile e ispezionabile a posteriori.

## 2. Addestrare

```bash
python curriculum_learning_training.py
```

La configurazione del curriculum è il dizionario in testa al file: un passo per riga, ognuno con
`map_size`, learning rate, `c2` (peso dell'entropia), se congelare la CNN e quali pesi ricaricare
dal passo precedente. `single_training.py` fa un singolo training senza curriculum.

## 3. Validare

```bash
python validate.py
```

Gira sul pool di testing, costruito da DTM che non compaiono mai nel training.

Lo split non è casuale. I DTM sono stati profilati misurando la frazione di mosse che il
terreno blocca a `max_step = 0,3`, e per il testing ne è stato preso **uno per quartile** della
distribuzione di training: un test set centrato sulla mediana misurerebbe solo il caso tipico,
mentre uno che copre i quartili ha la stessa distribuzione di difficoltà su cui l'agente si
allena. Risultato, mediana di mosse bloccate **6,5% nel training contro 6,2% nel testing**
(prima: 10,1% contro 59,3%, cioè due problemi diversi).

Se aggiungi DTM e vuoi rifare lo split, la profilazione va ripetuta: un DTM marziano può
passare dall'1% al 59% di mosse bloccate a parità di tutto il resto.

I flag in testa allo script: con `SAVE_RESULTS = True` salva un JSON per episodio in
`validation_info/`, con `False` apre una simulazione renderizzata in pygame;
`RANDOM_POLICY = True` dà la baseline casuale con cui confrontarsi.

I risultati si analizzano in `training_results.ipynb`. `how_things_work.ipynb` spiega ambiente,
DTM e osservazioni.

## Struttura

| file | ruolo |
|---|---|
| `build_tile_pool.py` | estrae le tile dai DTM (offline) |
| `tile_pool.py` | serve le tile a runtime, memory-mapped e condivise fra worker |
| `hirise_dtm.py` | lettura dei `.IMG` e geometria del terreno (FOV, mosse ammesse, adiacenze) |
| `custom_environment.py` | l'ambiente Gymnasium |
| `impala.py` | la rete IMPALA (policy + value head); usa GroupNorm, non BatchNorm |
| `agent.py` | PPO: raccolta esperienza, training, curriculum, validazione |
| `experience_manager.py` | buffer delle traiettorie e GAE |
| `constants.py` | path e iperparametri derivati da `map_size` |

## Osservazione

6 canali `(6, map_size, map_size)`, il numero sta in `OBSERVATION_CHANNELS` dentro
`constants.py` e la rete lo legge da lì:

| canale | contenuto |
|---|---|
| 0 | quote relative all'agente, normalizzate sui limiti di superabilità (±1 = limite esatto) e clippate a ±3 |
| 1 | maschera di validità: 1 dove l'agente ha già osservato la quota, 0 altrove |
| 2 | posizione corrente (1.0) e scia delle precedenti (valori crescenti verso 1) |
| 3 | posizione del target |
| 4 | scarto di riga verso il target, normalizzato, costante su tutta la mappa |
| 5 | scarto di colonna verso il target, normalizzato, costante su tutta la mappa |

I canali 4 e 5 sembrano ridondanti — il vettore agente-target è già deducibile da 2 e 3 — ma
non lo sono. In 2 e 3 quell'informazione è codificata come due singoli pixel accesi in una
griglia 20×20, e il tronco convoluzionale scende a 3×3 prima della dense layer: a quel punto i
due pixel cadono quasi sempre nella stessa cella e la posizione *relativa* non è più
recuperabile. Senza i canali 4 e 5 la rete ripiega sulla posizione **assoluta** del target, una
scorciatoia che regge lungo un avvicinamento rettilineo e crolla ovunque altrove, e
l'addestramento resta piatto. Misurato: con 4 canali il successo va da 17,9% a 21,1% in 600k
step, con 6 canali da 38,1% a 82,2%.

Alla policy viene inoltre passata una **maschera delle azioni**: l'ambiente espone in
`info["action_mask"]` quali delle 8 mosse sono effettivamente percorribili, e le probabilità
delle altre vengono azzerate prima della scelta. Senza la maschera una policy deterministica
può scegliere una direzione bloccata, restare ferma su un'osservazione immutata e ripetere la
stessa scelta all'infinito.

## Difficoltà del terreno

La leva sulla difficoltà sono `max_step_height` e `max_drop_height` negli script, non la
quantità di DTM: i DTM marziani a 1 m/px sono quasi tutti pianeggianti rispetto a un rover che
scala un metro. Misurato sul pool di training, mappa 20×20:

| `max_step`/`max_drop` | mosse bloccate dal terreno | mappa raggiungibile |
|---|---|---|
| 1,0 m | 2,0% | 99,3% |
| 0,5 m | 7,1% | 96,1% |
| **0,3 m** (valore attuale) | **15,1%** | **89,6%** |
| 0,2 m | 24,3% | 79,0% |

A 1 m il problema è "cammina verso il target in campo aperto" e una rete non addestrata lo
risolve già nel ~70% dei casi: non c'è nulla da imparare.
