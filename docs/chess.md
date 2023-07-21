# Chess

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_light.gif" width="30%">
    </p>

## Usage

```py
import pgx

env = pgx.make("chess")
```

or you can directly load `Chess` class

```py
from pgx.chess import Chess

env = Chess()
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
| Number of actions | `4672` |
| Observation shape | `(8, 8, 119)` |
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
4. `512` steps are elapsed (from AlphaZero `[Silver+18]`)


## Version History

- `v1` : Bug fix when castling by [@HongruiTang](https://github.com/HongruiTang) in [#983](https://github.com/sotetsuk/pgx/pull/983) (v1.1.0) 
- `v0` : Initial release (v1.0.0)

!!! success "Well-tested"

    Pgx chess implementation is well-tested. You can confirm that its behavior is identical to the throughly-tested [OpenSpiel](https://github.com/deepmind/open_spiel) implementation by [this colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/check_chess.ipynb).

## Reference

- `[Silver+18]` "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" Science
