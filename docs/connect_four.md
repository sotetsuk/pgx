# Connect four

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("connect_four")
```

or you can directly load `ConnectFour` class

```py
from pgx.connect_four import ConnectFour

env = ConnectFour()
```

## Description

> Connect Four is a two-player connection rack game, in which the players choose a color and then take turns dropping colored tokens into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own tokens.
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Connect_Four)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `7` |
| Observation shape | `(6, 7, 2)` |
| Observation type | `bool` |
| Rewards | `{-1, 0, 1}` |

## Observation

| Index | Description |
|:---:|:----|
| `[:, :, 0]` | represents `(6, 7)` squares filled by the current player |
| `[:, :, 1]` | represents `(6, 7)` squares filled by the opponent player of current player |

## Action
Each action represents the column index the player drops the token to.

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |
| Draw | `0` |

## Termination

Termination happens when 

1. either one player places four of their tokens in a row (horizontally, vertically, or diagonally), or 
2. all `42 (= 6 x 7)` squares are filled.


## Version History

- `v0` : Initial release (v1.0.0)
