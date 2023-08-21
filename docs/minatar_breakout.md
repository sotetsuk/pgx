# MinAtar Breakout

<p align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-breakout.gif" width="30%">
</p>


## Usage

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```
pip install pgx-minatar
```

Then, you can use the environment as follows:

```py
import pgx

env = pgx.make("minatar-breakout")
```

## Description

MinAtar is originally proposed by `[Young&Tian+19]`. 
The Pgx implementation is intended to be the *exact* copy of the original MinAtar implementation in JAX. The Breakout environment is described as follows:

> The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the
top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
>
> [github.com/kenjyoung/MinAtar - breakout.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `1` |
| Number of actions | `3` |
| Observation shape | `(10, 10, 4)` |
| Observation type | `bool` |
| Rewards | `{0, 1}` |

## Observation

| Index | Channel |
|:---:|:----|
| `[:, :, 0]` | Paddle |
| `[:, :, 1]` | Ball |
| `[:, :, 2]` | Trail |
| `[:, :, 3]` | Brick |

## Action

No-op (0), left (1), or right (2).

## Version History

- `v0` : Initial release (v1.0.0)

## Reference

- `[Young&Tian+19]` "Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments" [arXiv:1903.03176](https://arxiv.org/abs/1903.03176)


## License

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.