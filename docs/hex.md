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

## Rules

As the first player to move has a distinct advantage, the swap rule is used to compensate for this.
The detailed swap rule used in Pgx follows *swap pieces*:

> **"Swap pieces":** The players perform the swap by switching pieces. This means the initial red piece is replaced by a blue piece in the mirror image position, where the mirroring takes place with respect to the board's long diagonal. For example, a red piece at a3 becomes a blue piece at c1. The players do not switch colours: Red stays Red and Blue stays Blue. After the swap, it is Red's turn.
> 
> [Hex Wiki - Swap rule](https://www.hexwiki.net/index.php/Swap_rule)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `122 (= 11 x 11) + 1` |
| Observation shape | `(11, 11, 4)` |
| Observation type | `bool` |
| Rewards | `{-1, 1}` |

## Observation


| Index | Description |
|:---:|:----|
| `[:, :, 0]` | represents `(11, 11)` cells filled by `player_ix` |
| `[:, :, 1]` | represents `(11, 11)` cells filled by the opponent player of `player_id` |
| `[:, :, 2]` | represents whether `player_id` is black or white |
| `[:, :, 3]` | represents whether swap is legal or not |

## Action
Each action (`{0, ... 120}`) represents the cell index to be filled.
The final action `121` is the swap action available only at the second turn.

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |

Note that there is no draw in Hex.

## Termination

Termination happens when either one player connect opposite sides of the board.


## Version History

- `v0` : Initial release (v1.0.0)
