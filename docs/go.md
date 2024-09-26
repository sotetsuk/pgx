# Go

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("go_19x19")  # or "go_9x9"
```

or you can directly load `Go` class

```py
from pgx.go import Go

env = Go(size=19, komi=7.5)
```

## Description

> Go is an abstract strategy board game for two players in which the aim is to surround more territory than the opponent. The game was invented in China more than 2,500 years ago and is believed to be the oldest board game continuously played to the present day.
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Go_(game))

## Rules

The rule implemented in Pgx follows [Tromp-Taylor Rules](https://tromp.github.io/go.html).

!!! note Komi

    By default, we use `7.5`. Users can set different `komi` at `Go` class constructor.


!!! note Superko rule

    The Tromp-Taylor rule enforces PSK (Positional Superko). However, strictly implementing PSK to determine legal moves is inefficient, as it requires computing the hash for all possible subsequent board states. Since PSK rarely occurs—based on our best knowledge—most implementations compromise. For example:

    OpenSpiel uses SSK (Situational Superko) instead of PSK to compute legal moves, and if a PSK move occurs, the game ends in a tie. PettingZoo also uses SSK for legal moves but ignores PSK moves altogether. The strict rule is: "PSK for legal moves, and any PSK move results in an immediate loss." Like others, we also compromise. Our approach is similar to OpenSpiel:
    
    Pgx uses SSK for legal moves, but any PSK move results in an immediate loss. Overall, we believe the impact of this compromise is minimal, especially on a 19x19 board, since PSK scenarios are rare.

    | | Tromp-Taylor | OpenSpiel | PettingZoo | Pgx | 
    |:---|:---:|:---:|:---:|:---:| 
    | legal action | **PSK** | SSK | SSK | SSK |
    | PSK occurrence | **loss** | tie | ignore (SSK) | loss |
    
## Specs

Let `N` be the board size (e.g., `19`).

| Name | Value |
|:---|:----:|
| Version | `v1` |
| Number of players | `2` |
| Number of actions | `N x N + 1` |
| Observation shape | `(N, N, 17)` |
| Observation type | `bool` |
| Rewards | `{-1, 1}` |

## Observation
We follow the observation design of AlphaGo Zero `[Silver+17]`.

| Index | Description |
|:---:|:----|
| `obs[:, :, 0]` | stones of `player_id`          (@ current board) |
| `obs[:, :, 1]` | stones of `player_id`'s opponent (@ current board) |
| `obs[:, :, 2]` | stones of `player_id`          (@ 1-step before) |
| `obs[:, :, 3]` | stones of `player_id`'s opponent (@ 1-step before) |
| ... | ... |
| `obs[:, :, -1]` | color of `player_id` |

!!! note Final observation dimension

    For the final dimension, there are two possible options:

    - Use the color of current player to play
    - Use the color of `player_id`

    This ambiguity happens because `observe` function is available even if `player_id` is different from `state.current_player`.
    In AlphaGo Zero paper `[Silver+17]`, the final dimension C is explained as:

    > The final feature plane, C, represents the colour to play, and has a constant value of either 1 if black
        is to play or 0 if white is to play.

    however, it also describes as

    > the colour feature C is necessary because the komi is not observable.

    So, we use player_id's color to let the agent know komi information.
    As long as it's called when `player_id == state.current_player`, this doesn't matter.

## Action
Each action (`{0, ..., N * N - 1}`) represents the point to be colored.
The final action represents pass action.

## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |

## Termination

Termination happens when 

1. either one plays two consecutive passes, or
2. `N * N * 2` steps are elapsed `[Silver+17]`.


## Version History

- `v1` : Superko rule change in [#1224](https://github.com/sotetsuk/pgx/pull/1224) (v2.4.0)
- `v0` : Initial release (v1.0.0)

## Reference

1. `[Silver+17]` "Mastering the game of go without human knowledge" Nature

## Baseline models

Pgx offers a baseline model for Go (9x9). Users can use it for an anchor opponent in evaluation.
See [our paper](https://arxiv.org/abs/2303.17503) for more details. See [this colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/baselines.ipynb) for how to use it.

| Model ID | Description |
|:---:|:----|
| `go_9x9_v0`| See [our paper](https://arxiv.org/abs/2303.17503) for the training details. |
