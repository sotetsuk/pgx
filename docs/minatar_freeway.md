# MinAtar Freeway

<p align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-freeway.gif" width="30%">
</p>


## Usage

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```
pip install pgx-minatar
```

Then, you can use the environment as follows:

```py
import pgx

env = pgx.make("minatar-freeway")
```

## Description

MinAtar is originally proposed by `[Young&Tian+19]`. 
The Pgx implementation is intended to be the *exact* copy of the original MinAtar implementation in JAX. The Freeway environment is described as follows:

> The player begins at the bottom of the screen and motion is restricted to traveling up and down. Player speed is
also restricted such that the player can only move every 3 frames. A reward of +1 is given when the player reaches
the top of the screen, at which point the player is returned to the bottom. Cars travel horizontally on the screen
and teleport to the other side when the edge is reached. When hit by a car, the player is returned to the bottom of
the screen. Car direction and speed is indicated by 5 trail channels, the location of the trail gives direction
while the specific channel indicates how frequently the car moves (from once every frame to once every 5 frames).
Each time the player successfully reaches the top of the screen, the car speeds are randomized. Termination occurs
after 2500 frames have elapsed.
> 
> [github.com/kenjyoung/MinAtar - freeway.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `1` |
| Number of actions | `3` |
| Observation shape | `(10, 10, 7)` |
| Observation type | `bool` |
| Rewards | `{0, 1}` |

## Observation

| Index | Channel |
|:---:|:----|
| `[:, :, 0]` | Chicken |
| `[:, :, 1]` | Car |
| `[:, :, 2]` | Speed 1 |
| `[:, :, 3]` | Speed 2 |
| `[:, :, 4]` | Speed 3 |
| `[:, :, 5]` | Speed 4 |

## Action
No-op (0), up (1), or down (2).

## Version History

- `v1`: Specify rng key explicitly (API v2) by [@sotetsuk](https://github.com/sotetsuk) in [#1058](https://github.com/sotetsuk/pgx/pull/1058) (v2.0.0)
- `v0` : Initial release (v1.0.0)

## Training example

For MinAtar environments, we provide a [PPO training example](https://github.com/sotetsuk/pgx/tree/main/examples/minatar-ppo), which takes only 1 min to train on a single GPU.


## Baseline models

We provide a baseline model for the MinAtar Freeway environment, which reasonably plays the game.

```py
model = pgx.make_baseline("minatar-freeway_v0")

logits, value = model(state.observation)
```

We trained the model with PPO for 20M steps. 
See [wandb report](https://api.wandb.ai/links/sotetsuk/k5cfwe17) for the details of the training.

## Reference

- `[Young&Tian+19]` "Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments" [arXiv:1903.03176](https://arxiv.org/abs/1903.03176)

## LICENSE

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.