# Othello

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("othello")
```

or you can directly load `Othello` class

```py
from pgx.othello import Othello

env = Othello()
```

## Description

> **Othello**, or differing in not having a defined starting position, Reversi, is a two-player zero-sum and perfect information abstract strategy board game, usually played on a board with 8 rows and 8 columns and a set of light and a dark turnable pieces for each side. The player's goal is to have a majority of their colored pieces showing at the end of the game, turning over as many of their opponent's pieces as possible. The dark player makes the first move from the starting position, alternating with the light player. Each player has to place a piece on the board such that there exists at least one straight (horizontal, vertical, or diagonal) occupied line of opponent pieces between the new piece and another own piece. After placing the piece, the side turns over (flips, captures) all opponent pieces lying on any straight lines between the new piece and any anchoring own pieces.
> 
> [Chess Programming Wiki](https://www.chessprogramming.org/Othello)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `65 (= 8 x 8 + 1)` |
| Observation shape | `(8, 8, 2)` |
| Observation type | `bool` |
| Rewards | `{-1, 0, 1}` |

## Observation


| Index | Description |
|:---:|:----|
| `[:, :, 0]` | represents `(8, 8)` squares colored by the current player |
| `[:, :, 1]` | represents `(8, 8)` squares colored by the opponent player of current player |

## Action
Each action (`{0, ..., 63}`) represents the square index to be filled. The last `64`-th action represents pass action.

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |
| Draw | `0` |

## Termination

Termination happens when all `64 (= 8 x 8)` playable squares are filled.

## Version History

- `v0` : Initial release (v1.0.0)

## Baseline models

Pgx offers a baseline model for Othello. Users can use it for an anchor opponent in evaluation.
See [our paper](https://arxiv.org/abs/2303.17503) for more details. See [this colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/baselines.ipynb) for how to use it.

| Model ID | Description |
|:---:|:----|
| `othello_v0`| See [our paper](https://arxiv.org/abs/2303.17503) for the training details. |