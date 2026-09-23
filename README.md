# Rover-Trajectory-Finder

A rover has landed on Mars. It knows its landing point and its destination, but it can only
perceive what is immediately around it. The goal is to reach the target while keeping the path
short.

The terrain comes from real [HiRISE](https://www.uahirise.org/dtm/) digital terrain models,
reduced to a grid where one pixel is one metre. The agent is trained with PPO on an IMPALA
network.

![The rover solving three episodes](assets/rover_demo.gif)

*Blue cells are where the rover has been, amber dots are the shortest path found by Dijkstra,
the red square is the target. The rover only sees its own neighbourhood, not the whole map.*

## Results

On 40×40 maps, evaluated on held-out terrain the agent never trained on (300 episodes each).
Dijkstra has full knowledge of the map and gives the ceiling; the rover only ever sees its own
field of view.

| | reached the target | steps to target (median) | reward per step |
|---|---|---|---|
| Dijkstra (full map, optimal) | 95.0% | 19 | — |
| **trained policy** | **90.0%** | **68** | **+0.543** |
| random policy | 50.7% | 1506 | +0.092 |

Dijkstra's 95% is not a failure to solve the rest: in 5% of episodes no path exists at all, so
95% is the ceiling any policy can reach.

![Reward per step](assets/reward_40x40.png)

![Steps to reach the target](assets/steps_to_target_40x40.png)

## Setup

Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu129
.venv\Scripts\pip install -r requirements.txt
```

The first line installs the CUDA build of torch: plain `pip install torch` on Windows gives the
CPU-only one.

## Running it

Download some DTMs from the [HiRISE DTM catalogue](https://www.uahirise.org/hiwish/maps/dtms.jsp)
into `DTMs/training` and `DTMs/testing`, then:

```bash
python build_tile_pool.py --dtm-dir DTMs/training --out tile_pools/tiles_training.npy --n-tiles 50000
python build_tile_pool.py --dtm-dir DTMs/testing  --out tile_pools/tiles_testing.npy  --n-tiles 4000
python curriculum_learning_training.py
python validate.py
```

> **Rebuild the pools whenever you add, remove or replace an `.IMG`.** Training reads the `.npy`,
> not the `DTMs/` folder, so otherwise the run silently uses the old data without raising
> anything.

The curriculum is the dictionary at the top of `curriculum_learning_training.py`: one step per
entry, with map size, learning rate, loss coefficients and which weights to carry over. The
shipped configuration learns on 20×20 maps and then moves the same policy to 40×40.
`single_training.py` runs a single stage without the curriculum.

`validate.py` has three flags at the top: `SAVE_RESULTS` writes one JSON per episode into
`validation_info/` or else opens a rendered pygame simulation, `RANDOM_POLICY` produces the
baseline, and `SAMPLE_ACTION` chooses how actions are drawn. Results are analysed in
`training_results.ipynb`; `how_things_work.ipynb` explains the environment.

## How the terrain is fed to the agent

The `.IMG` files are never loaded at runtime. `build_tile_pool.py` scans them with windowed
`rasterio` reads and stacks filtered 64×64 tiles into a single memory-mapped `.npy` that every
worker process shares. Loading the DTMs directly instead costs 2.63 GB per worker — 84 GB across
32 workers, all of it the same data.

So the `.IMG` files are a **build-time input, not a runtime dependency**: to grow the dataset,
download a batch of DTMs, fold their tiles into the pool, delete the `.IMG` files and repeat.
Runtime memory depends only on `--n-tiles`, never on how much terrain was scanned.

Tiles come from a non-overlapping lattice with a random offset, with an equal quota per DTM so
one large raster cannot dominate the pool. `tiles_*_meta.json` records where every tile came
from, so a training run is reproducible and can be inspected afterwards.

The train/test split is not random either. The DTMs were profiled by measuring the fraction of
moves the terrain blocks, and one DTM per quartile of the training distribution was held out, so
the test set spans the same difficulty range rather than clustering at its median. A Mars DTM can
range from 1% to 59% blocked moves, so this is worth redoing if you change the dataset.

## Observation

6 channels, `(6, map_size, map_size)`. The count lives in `OBSERVATION_CHANNELS` in
`constants.py` and the network reads it from there.

| channel | contents |
|---|---|
| 0 | altitudes relative to the agent, normalised against its traversability limits (±1 is exactly the limit) and clipped at ±3 |
| 1 | validity mask: 1 where the agent has already observed the altitude, 0 elsewhere |
| 2 | current position (1.0) and the trail of previous ones (values rising towards 1) |
| 3 | target position |
| 4 | row offset to the target, normalised, constant over the whole map |
| 5 | column offset to the target, normalised, constant over the whole map |

Channels 4 and 5 look redundant — the agent-to-target vector is already implied by 2 and 3 — but
they are not. There it is encoded as two single lit pixels in a 20×20 grid, and the convolutional
trunk reduces that to 3×3 before the dense layer, by which point both pixels usually fall in the
same cell and their *relative* position is no longer recoverable. Without channels 4 and 5 the
network falls back on the target's **absolute** position, a shortcut that holds along a straight
approach and collapses everywhere else, and training stays flat: success goes from 17.9% to 21.1%
over 600k steps, against 38.1% to 82.2% with the extra channels.

The policy is also given an **action mask**: the environment exposes in `info["action_mask"]`
which of the 8 moves are traversable, and the others have their probability zeroed before the
choice is made.

## Terrain difficulty

The lever on difficulty is `max_step_height` and `max_drop_height` in the scripts, not the number
of DTMs: Mars DTMs at 1 m/px are almost all flat relative to a rover that can climb a metre.
Measured on the training pool, 20×20 maps:

| `max_step`/`max_drop` | moves blocked by terrain | map reachable |
|---|---|---|
| 1.0 m | 2.0% | 99.3% |
| 0.5 m | 7.1% | 96.1% |
| **0.3 m** (current value) | **15.1%** | **89.6%** |
| 0.2 m | 24.3% | 79.0% |

At 1 m the problem is "walk to the target across an open field" and an untrained network already
solves about 70% of it: there is nothing to learn.

## Project layout

| file | role |
|---|---|
| `build_tile_pool.py` | extracts tiles from the DTMs (offline) |
| `tile_pool.py` | serves tiles at runtime, memory-mapped and shared across workers |
| `hirise_dtm.py` | reads `.IMG` files and the terrain geometry (FOV, legal moves, adjacency) |
| `custom_environment.py` | the Gymnasium environment |
| `impala.py` | the IMPALA network (policy + value head); uses GroupNorm, not BatchNorm |
| `agent.py` | PPO: experience collection, training, curriculum, validation |
| `experience_manager.py` | trajectory buffer and GAE |
| `constants.py` | paths and hyperparameters derived from `map_size` |

`DECISIONS.md` records the non-obvious design decisions together with the measurements that
motivated them. Most of them come from bugs that raised no error and only produced a flat
training curve, so it is worth reading before changing anything.
