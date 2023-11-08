# MinAtar Seaquest

<p align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-seaquest.gif" width="30%">
</p>


## Usage

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```
pip install pgx-minatar
```

Then, you can use the environment as follows:

```py
import pgx

env = pgx.make("minatar-seaquest")
```

## Description

MinAtar is originally proposed by `[Young&Tian+19]`. 
The Pgx implementation is intended to be the *exact* copy of the original MinAtar implementation in JAX. The Seaquest environment is described as follows:

> The player controls a submarine consisting of two cells, front and back, to allow direction to be determined. The
player can also fire bullets from the front of the submarine. Enemies consist of submarines and fish, distinguished
by the fact that submarines shoot bullets and fish do not. A reward of +1 is given each time an enemy is struck by
one of the player's bullets, at which point the enemy is also removed. There are also divers which the player can
move onto to pick up, doing so increments a bar indicated by another channel along the bottom of the screen. The
player also has a limited supply of oxygen indicated by another bar in another channel. Oxygen degrades over time,
and is replenished whenever the player moves to the top of the screen as long as the player has at least one rescued
diver on board. The player can carry a maximum of 6 divers. When surfacing with less than 6, one diver is removed.
When surfacing with 6, all divers are removed and a reward is given for each active cell in the oxygen bar. Each
time the player surfaces the difficulty is increased by increasing the spawn rate and movement speed of enemies.
Termination occurs when the player is hit by an enemy fish, sub or bullet; or when oxygen reached 0; or when the
player attempts to surface with no rescued divers. Enemy and diver directions are indicated by a trail channel
active in their previous location to reduce partial observability.
> 
> [github.com/kenjyoung/MinAtar - seaquest.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/seaquest.py)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `1` |
| Number of actions | `6` |
| Observation shape | `(10, 10, 10)` |
| Observation type | `bool` |
| Rewards | `{0, 1, ..., 10}` |

## Observation

| Index | Channel |
|:---:|:----|
| `[:, :, 0]` | Player submarine (front) |
| `[:, :, 1]` | Player submarine (back) |
| `[:, :, 2]` | Friendly bullet |
| `[:, :, 3]` | Trail |
| `[:, :, 4]` | Enemy bullet |
| `[:, :, 5]` | Enemy fish |
| `[:, :, 6]` | Enemy submarine |
| `[:, :, 7]` | Oxygen guage |
| `[:, :, 8]` | Diver guage |
| `[:, :, 9]` | Diver |

## Action

No-op (0), up (1), down (2), left (3), right (4), or fire (5).

## Version History

- `v1`: Specify rng key explicitly (API v2) by [@sotetsuk](https://github.com/sotetsuk) in [#1058](https://github.com/sotetsuk/pgx/pull/1058) (v2.0.0)
- `v0` : Initial release (v1.0.0)

## Training example

For MinAtar environments, we provide a [PPO training example](https://github.com/sotetsuk/pgx/tree/main/examples/minatar-ppo), which takes only 1 min to train on a single GPU.


## Baseline models

We provide a baseline model for the MinAtar Seaquest environment, which reasonably plays the game.

```py
model = pgx.make_baseline("minatar-seaquest_v0")

logits, value = model(state.observation)
```

We trained the model with PPO for 20M steps. 
See [wandb report](https://api.wandb.ai/links/sotetsuk/k5cfwe17) for the details of the training.

## Reference

- `[Young&Tian+19]` "Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments" [arXiv:1903.03176](https://arxiv.org/abs/1903.03176)

## LICENSE

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.