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
from pgx.chess import chess

env = chess()
```

## Description

TBA


## Rules

TBA

## Specs

Let `N` be the board size (e.g., `19`).

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `N x N + 1` |
| Observation shape | `(N, N, 17)` |
| Observation type | `bool` |
| Rewards | `{-1, 0, 1}` |

## Observation
We follow the observation design of Alphachess Zero `[Silver+17]`.

| Index | Description |
|:---:|:----|
| `obs[:, :, 0]` | stones of `player_id`          (@ current board) |
| `obs[:, :, 1]` | stones of `player_id`'s opponent (@ current board) |
| `obs[:, :, 2]` | stones of `player_id`          (@ 1-step before) |
| `obs[:, :, 3]` | stones of `player_id`'s opponent (@ 1-step before) |
| ... | ... |
| `obs[:, :, -1]` | color of `player_id` |

!!! note "Final observation dimension"

    For the final dimension, there are two possible options:

    - Use the color of current player to play
    - Use the color of `player_id`

    This ambiguity happens because `observe` function is available even if `player_id` is different from `state.current_player`.
    In Alphachess Zero paper `[Silver+17]`, the final dimension C is explained as:

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
| Draw | `0` |

## Termination

Termination happens when 

1. either one plays two consecutive passes, or
2. `N * N * 2` steps are elapsed


## Version History

- `v0` : Initial release (v1.0.0)

## Reference
