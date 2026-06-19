# BDQN UAV Search-and-Track

Educational repository for learning **Bayesian Deep Q-Networks (BDQN)** on a small UAV search-and-track problem.

The environment is intentionally simple:

- one drone;
- one `20 x 20` grid map;
- four hidden targets randomly placed at reset;
- three targets have value `1`;
- one target has value `2`;
- the drone must decide whether to **search** for unknown targets or **track** already detected targets.

The goal is not to build a perfect UAV simulator. The goal is to understand how a BDQN can use posterior uncertainty over Q-values to explore more intelligently than a standard DQN.

---

## Core idea

A standard DQN outputs deterministic Q-values:

```text
Q(s, a; theta)
```

A BDQN keeps the neural network as a feature extractor:

```text
phi(s; theta)
```

Then the last layer is replaced by a Bayesian linear model:

```text
Q(s, a) = w_a^T phi(s)
```

where each action has a posterior over its weights:

```text
w_a ~ N(mu_a, Sigma_a)
```

At decision time, the agent samples weights from the posterior and acts greedily with respect to the sampled Q-function. This is Thompson sampling.

---

## Environment

### State

The observation is a tensor with shape:

```text
(5, 20, 20)
```

Channels:

1. drone position;
2. current belief probability map;
3. detected target value map;
4. completed target map;
5. visited-cell map.

### Actions

The action space has `10` discrete actions:

```text
mode in {SEARCH, TRACK} x move in {STAY, UP, DOWN, LEFT, RIGHT}
```

So for example:

```text
0 = SEARCH + STAY
1 = SEARCH + UP
...
5 = TRACK + STAY
6 = TRACK + UP
...
```

### Target mechanics

At each episode reset:

- three targets of value `1` are placed randomly;
- one target of value `2` is placed randomly;
- the agent does not know their positions;
- the belief map starts uniform.

When the drone searches, it can detect hidden targets within the sensor radius.
When the drone tracks, it gains progress on a detected target if it is close enough.
Once a target is completed, it gives no further completion reward.

---

## Reward design

The reward is shaped to encourage a meaningful search-track balance:

```text
r_t = r_step
    + r_new_cell
    + r_detection
    + r_tracking_progress
    + r_completion
    + r_invalid
    + r_redundant
```

Recommended default values:

| Term | Value | Meaning |
|---|---:|---|
| `step_penalty` | `-0.01` | encourages shorter missions |
| `new_cell_bonus` | `+0.01` | weak exploration shaping |
| `revisit_penalty` | `-0.005` | discourages looping |
| `detect_bonus * target_value` | `+0.5 * value` | rewards discovering new targets |
| `track_progress_bonus * value` | `+0.05 * value` | rewards useful tracking before completion |
| `complete_bonus * value` | `+2.0 * value` | main task reward |
| `invalid_track_penalty` | `-0.05` | penalizes tracking when no detected target is available |
| `completed_track_penalty` | `-0.03` | discourages staying on completed targets |
| `all_targets_bonus` | `+3.0` | encourages completing the whole map |

This makes the value-2 target attractive, but not enough to justify staying there forever because completion reward is one-shot.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train

```bash
python scripts/train.py --episodes 500
```

Useful options:

```bash
python scripts/train.py \
  --episodes 1000 \
  --grid-size 20 \
  --sensor-radius 2 \
  --max-steps 150 \
  --device auto
```

---

## Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/latest.pt --episodes 10
```

---

## What to look at first

Start with these files:

```text
src/uav_bdqn/envs/search_track_env.py   # environment and reward
src/uav_bdqn/agents/bdqn_agent.py       # BDQN + Thompson sampling
src/uav_bdqn/models/networks.py         # feature extractor phi(s)
scripts/train.py                        # training loop
```

---

## Expected learning behavior

Early training:

- the drone moves almost randomly;
- it discovers targets inconsistently;
- tracking is often invalid or badly timed.

Middle training:

- the drone starts sweeping unexplored areas;
- it tracks detected targets more often;
- the value-2 target becomes more attractive.

Later training:

- the drone should learn to search until it finds targets;
- track them until completion;
- then leave and search elsewhere.

