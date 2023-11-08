# MinAtar Asterix

<p align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-asterix.gif" width="30%">
</p>


## Usage

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```
pip install pgx-minatar
```

Then, you can use the environment as follows:

```py
import pgx

env = pgx.make("minatar-asterix")
```

## Description

MinAtar is originally proposed by `[Young&Tian+19]`. 
The Pgx implementation is intended to be the *exact* copy of the original MinAtar implementation in JAX. The Asterix environment is described as follows:

> The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of
+1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and
treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed
and spawn rate of enemies and treasure. 
>
> [github.com/kenjyoung/MinAtar - asterix.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/asterix.py)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `1` |
| Number of actions | `5` |
| Observation shape | `(10, 10, 4)` |
| Observation type | `bool` |
| Rewards | `{0, 1}` |

## Observation

| Index | Channel |
|:---:|:----|
| `[:, :, 0]` | Player |
| `[:, :, 1]` | Enemy |
| `[:, :, 2]` | Trail |
| `[:, :, 3]` | Gold |

## Action
No-op (0), left (1), right (2), up (3), or down (4).

## Version History

- `v1`: Specify rng key explicitly (API v2) by [@sotetsuk](https://github.com/sotetsuk) in [#1058](https://github.com/sotetsuk/pgx/pull/1058) (v2.0.0)
- `v0` : Initial release (v1.0.0)

## Training example

For MinAtar environments, we provide a [PPO training example](https://github.com/sotetsuk/pgx/tree/main/examples/minatar-ppo), which takes only 1 min to train on a single GPU.


## Baseline models

We provide a baseline model for the MinAtar Asterix environment, which reasonably plays the game.

```py
model = pgx.make_baseline("minatar-asterix_v0")

logits, value = model(state.observation)
```

We trained the model with PPO for 20M steps. 
See [wandb report](https://api.wandb.ai/links/sotetsuk/k5cfwe17) for the details of the training.

## Reference

- `[Young&Tian+19]` "Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments" [arXiv:1903.03176](https://arxiv.org/abs/1903.03176)

## License

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.