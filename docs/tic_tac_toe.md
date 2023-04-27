# Tic-tac-toe

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("tic_tac_toe")
```

or you can directly load `TicTacToe` class

```py
from pgx.tic_tac_toe import TicTacToe

env = TicTacToe("tic_tac_toe")
```

## Description

> Tic-tac-toe is a paper-and-pencil game for two players who take turns marking the spaces in a three-by-three grid with X or O. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner. 
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Tic-tac-toe)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `9` |
| Observation shape | `(3, 3, 2)` |
| Observation type | `bool` |
| Rewards | `{-1, 0, 1}` |

## Observation


| Index | Description |
|:---:|:----|
| `[:, :, 0]` | represents `(3, 3)` squares filled by the current player |
| `[:, :, 1]` | represents `(3, 3)` squares filled by the opponent player of current player |

## Action
Each action represents the square index to be filled.

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

1. either one player places three of their symbols in a row (horizontally, vertically, or diagonally), or 
2. all nine squares are filled.


## Version History

- `v0` : Initial release (v1.0.0)
