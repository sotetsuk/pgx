# Hex

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_dark.gif" width="50%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_light.gif" width="50%">
    </p>


## Usage

```py
import pgx

env = pgx.make("hex")
```

or you can directly load `Hex` class

```py
from pgx.hex import Hex

env = Hex()
```

## Description

> Hex is a two player abstract strategy board game in which players attempt to connect opposite sides of a rhombus-shaped board made of hexagonal cells. Hex was invented by mathematician and poet Piet Hein in 1942 and later rediscovered and popularized by John Nash.
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Hex_(board_game))

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `121 (= 11 x 11)` |
| Observation shape | `(11, 11, 3)` |
| Observation type | `bool` |
| Rewards | `{-1, 1}` |

## Observation


| Index | Description |
|:---:|:----|
| `[:, :, 0]` | represents `(11, 11)` cells filled by `player_ix` |
| `[:, :, 1]` | represents `(11, 11)` cells filled by the opponent player of `player_id` |
| `[:, :, 2]` | represents whether `player_id` is black or white|

## Action
Each action represents the cell index to be filled.

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |

## Termination

Termination happens when 

1. either one player connect opposite sides of the board, or 
2. all `121 (= 11 x 11)` cells are filled.


## Version History

- `v0` : Initial release (v1.0.0)
