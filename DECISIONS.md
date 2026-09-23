# Design decisions, and why

Notes on the non-obvious choices, with the measurements that motivated them. They exist so the
same mistakes are not made twice: nearly all of them came from a bug that raised no error and
only produced a flat training curve.

## 6 observation channels, not 4

Channels 4 and 5 (row and column offset to the target, constant over the whole map) look
redundant against channels 2 and 3, which already hold the agent and target positions. They are
not: there the information is encoded as **two single lit pixels** in a 20×20 grid, and the
convolutional trunk reduces that to 3×3 before the dense layer. At that resolution both pixels
usually fall in the same cell.

Measured by cloning a greedy policy and evaluating it on two state distributions:

| | 4 channels | 6 channels |
|---|---|---|
| accuracy on states visited by the greedy policy | 100.0% | 99.9% |
| accuracy on states visited by a random policy | **15.7%** (chance: 25%) | 43.6% |
| training success, first to last decile | 17.9% → 21.1% | **38.1% → 82.2%** |

With 4 channels the network was not using the agent's position at all: zeroing channel 2 only
dropped accuracy from 100% to 97.8%. It predicts the action from the target's **absolute**
position, a shortcut that holds along a straight approach and falls below chance everywhere else
— which is exactly where PPO spends its time, since it explores by sampling.

If you change the architecture, re-check this point: it is the single factor separating a flat
training run from one that learns.

## GroupNorm, not BatchNorm

`BatchNorm` makes the network's output depend on *which other samples* are in the batch. In
on-policy RL the same state is evaluated in two different batches — 32 parallel environments
while collecting, 512 shuffled samples while updating — so the stored action probabilities and
values disagree with the ones the update recomputes.

Measured on the same 32 states, batch 32 against batch 512:

| | BatchNorm | GroupNorm |
|---|---|---|
| PPO importance ratio, standard deviation | 0.516 | **0.000** |
| value V | −0.427 vs +0.282 | identical |

At a standard deviation of 0.516 the PPO ratio saturates the clipping range on normalisation
noise alone, and the critic chases a target that changes sign depending on the batch.

## Reward: dense potential shaping, not a ratchet

The previous version paid a bonus only when the agent beat its best-ever distance. Measured on
the training data: **98.9% of steps received exactly the same reward**, and the last bonus
arrived on average at 71% of the episode. With episodes around 900 steps against a GAE credit
horizon of ~20 steps (`1/(1-gamma*lambda)`), the signal never reached the start of the episode.

Replaced with potential-based shaping `gamma*PHI(s') - PHI(s)` with `PHI = -distance`, which is
dense and provably leaves the optimal policy unchanged. Correlation between "this action moves
closer to the target" and the estimated advantage went from **+0.059 to +0.329**.

One consequence when reporting results: do not compare policies by *total* episode reward. For
`gamma < 1` the undiscounted sum keeps a term `(1-gamma) * sum(distances)` that grows with
episode length, so a policy that wanders far from the target for thousands of steps collects
more raw reward than one that arrives quickly. The optimal policy is unchanged, but the
undiscounted sum is not the objective. Use reward per step, success rate and path length.

## c1 = 0.05, not 0.5

Policy and critic share the same convolutional trunk. With dense returns the value MSE starts
around 9.5 against an `actor_loss` of 0.055: `c1 * critic_loss` weighed **86 times** the actor,
and the network was optimised almost entirely for value prediction. It is exposed per step in
the curriculum config.

## Action mask

The environment exposes in `info["action_mask"]` which of the 8 moves are traversable, and the
others have their logits set to `-inf`. It is applied consistently in three places — sampling,
the PPO update (the mask is stored in the `ExperienceManager`) and validation — otherwise the
importance ratio would compare different distributions.

Without the mask a deterministic policy picked a blocked direction, stayed in place on an
unchanged observation and repeated the same choice forever: **89.3% of steps without movement**,
now 0%.

## Stored states are s(t), not s(t+1)

In `Agent.train()` the variable `observations` is reassigned by `environments.step()`. The
previous version stored the value from **after** the step, so the PPO update computed
`pi_new(a_t | s_t+1) / pi_old(a_t | s_t)`: two distributions over different states. Measured on
the first minibatch, before any update: mean ratio 0.653 with standard deviation 0.453 and
**52.3% of samples outside the clipping range**, against 1.001 ± 0.071 after the fix.

## Difficulty comes from the rover's limits, not the number of DTMs

Mars DTMs at 1 m/px are almost all flat relative to a rover that can climb a metre. With
`max_step = 1` only 0.8% of moves are blocked and an untrained network already solves ~70% of
episodes: there is nothing to learn. At **0.3 m** 15.1% of moves are blocked and 89.6% of the map
is still reachable.

Adding DTMs does not change this: among the 15 DTMs downloaded in September 2026 the average
blocked-move rate was 1.8% against 1.0% for the earlier ones, and only 2 of 15 were genuinely
rugged.

## Stratified train/test split

DTMs range from 1.1% to 59.3% blocked moves at 0.3 m — an order of magnitude and a half. The test
set was built by taking **one DTM per quartile** of the training distribution rather than the
ones closest to the median, so that it reproduces the training difficulty distribution instead of
measuring only the typical case. Before: training median 10.1%, testing 59.3% — effectively two
different problems. After: 6.5% against 6.2%.

If you add DTMs and redo the split, repeat the profiling; it cannot be judged by eye.

## Sampled or greedy actions at validation

Both numbers are meaningful and they measure different things. **Sampled** is the performance of
the policy PPO actually optimised, since the objective is the expected return of the stochastic
policy π(a|s), and the entropy bonus explicitly keeps it stochastic. **Argmax** is the
performance of its greedy projection, which is what you would deploy for deterministic behaviour.

The gap between them is itself a diagnostic. On the trained policy it is large — 60.5% against
92.5% — because the agent is purely reactive: when its preferred move is blocked and the
observation does not change, a deterministic policy cannot know it has already been there, and
loops. Measured over failed argmax episodes: the full step budget spent visiting about 15 of 400
cells.

The principled way to close that gap is to give the policy what it needs to break loops
deliberately. The environment already tracks `visited_locations` but does not put it in the
observation.

## Reference numbers

Measured on 20×20 maps, `max_step = 0.3`, training pool:

| policy | budget 200 | budget 1341 |
|---|---|---|
| random | 19.5% in 178.6 steps | 62.0% in 803.3 steps |
| greedy towards the target (hand-written) | 80.2% in 47.8 steps | 80.2% in 273.2 steps |
| optimal path (Dijkstra) | ~10 steps | ~10 steps |

**Success rate alone is a weak metric**: with 1341 steps allowed on a 20×20 map (3.3 steps per
cell) a random walk covers most of the grid and reaches 62%. Look at mean episode length against
the optimum instead — random takes ~52 times the optimal path, so that is where the headroom is.

## How to check the training loop is sound

If training ever goes flat again, the test that isolates the algorithm from the task is running
`Agent.train()` on CartPole-v1: it only needs a wrapper adding the `info` keys the episode
logging reads, and an MLP with the same `forward(x, action_mask=None) -> (probs, value)`
signature. A correct PPO takes CartPole from ~24 to several hundred steps per episode. If
CartPole climbs and the rover does not, the defect is in the task rather than the algorithm —
which is exactly how the 6-channel problem was found.
