# Gardner chess

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/gardner_chess_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/gardner_chess_light.gif" width="30%">
    </p>

## Usage

```py
import pgx

env = pgx.make("gardner_chess")
```

or you can directly load `GardnerChess` class

```py
from pgx.gardner_chess import GardnerChess

env = GardnerChess()
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
| Number of actions | `1225` |
| Observation shape | `(5, 5, 115)` |
| Observation type | `float` |
| Rewards | `{-1, 0, 1}` |

## Observation
We follow the observation design of AlphaZero `[Silver+18]`.

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

Termination occurs when one of the following conditions are satisfied:

1. checkmate
2. stalemate
3. no sufficient pieces to checkmate
4. `50` halfmoves are elapsed without any captures or pawn moves
4. `256` steps are elapsed (`512` in full-size chess experiments in AlphaZero `[Silver+18]`)


## Version History

- `v0` : Initial release (v1.0.0)

## Reference

- `[Silver+18]` "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" Science 