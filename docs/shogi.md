# Shogi

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif" width="30%">
    </p>

## Usage

```py
import pgx

env = pgx.make("shogi")
```

or you can directly load `Shogi` class

```py
from pgx.shogi import Shogi

env = Shogi()
```

## Description

TBA


## Rules

TBA

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `2187` |
| Observation shape | `(9, 9, 119)` |
| Observation type | `float` |
| Rewards | `{-1, 0, 1}` |

## Observation
We follow the observation design of [dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi), a open-source shogi AI.

| Index | Description |
|:---:|:----|
| TBA | TBA |

## Action

TBA

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |
| Draw | `0` |

## Termination

TBA

## Version History

- `v0` : Initial release (v1.0.0)
