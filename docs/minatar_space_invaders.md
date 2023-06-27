# MinAtar Space Invaders

<p align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-space_invaders.gif" width="30%">
</p>


## Usage

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```
pip install pgx-minatar
```

Then, you can use the environment as follows:

```py
import pgx

env = pgx.make("minatar-space_invaders")
```

## Description

MinAtar is originally proposed by `[Young&Tian+19]`. 
The Pgx implementation is intended to be the *exact* copy of the original MinAtar implementation in JAX. The Space Invaders environment is described as follows:

> The player controls a cannon at the bottom of the screen and can shoot bullets upward at a cluster of aliens above.
The aliens move across the screen until one of them hits the edge, at which point they all move down and switch
directions. The current alien direction is indicated by 2 channels (one for left and one for right) one of which is
active at the location of each alien. A reward of +1 is given each time an alien is shot, and that alien is also
removed. The aliens will also shoot bullets back at the player. When few aliens are left, alien speed will begin to
increase. When only one alien is left, it will move at one cell per frame. When a wave of aliens is fully cleared a
new one will spawn which moves at a slightly faster speed than the last. Termination occurs when an alien or bullet
hits the player.
> 
> [github.com/kenjyoung/MinAtar - space_invaders.py](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/space_invaders.py)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `1` |
| Number of actions | `4` |
| Observation shape | `(10, 10, 6)` |
| Observation type | `bool` |
| Rewards | `{0, 1}` |

## Observation

| Index | Channel |
|:---:|:----|
| `[:, :, 0]` | Cannon |
| `[:, :, 1]` | Alien |
| `[:, :, 2]` | Alien left |
| `[:, :, 3]` | Alien right |
| `[:, :, 4]` | Friendly bullet |
| `[:, :, 5]` | Enemy bullet |

## Action

TBA

## Version History

- `v0` : Initial release (v1.0.0)

## Reference

- `[Young&Tian+19]` "Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments" [arXiv:1903.03176](https://arxiv.org/abs/1903.03176)


## LICENSE

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.