# AnimalShogi

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_dark.gif" width="50%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_light.gif" width="50%">
    </p>


## Usage

```py
import pgx

env = pgx.make("animal_shogi")
```

or you can directly load `AnimalShogi` class

```py
from pgx.animal_shogi import AnimalShogi

env = AnimalShogi()
```

## Description

Animal Shogi (Dōbutsu shōgi) is a variant of shogi primarily developed for children. It consists of a 3x4 board and four types of pieces (five including promoted pieces). One of the rule differences from regular shogi is the *Try* Rule, where entering the opponent's territory with the king leads to victory.

See also [Wikipedia](https://en.wikipedia.org/wiki/D%C5%8Dbutsu_sh%C5%8Dgi)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v2` |
| Number of players | `2` |
| Number of actions | `132` |
| Observation shape | `(4, 3, 194)` |
| Observation type | `float` |
| Rewards | `{-1, 0, 1}` |

## Observation


| Index | Description |
|:---:|:----|
| `[:, :, 0:5]` | my pieces on board |
| `[:, :, 5:10]` | opponent's pieces on board |
| `[:, :, 10:16]` | my hands |
| `[:, :, 16:22]` | opponent's hands |
| `[:, :, 22:24]` | repetitions |
| ... | ... |
| `[:, :, 193]` | `player_id`'s turn' |
| `[:, :, 194]` | Elapsed timesteps (normalized to `1`) |


## Action

Uses AlphaZero like action label:

- `132` labels
- Move: `8 x 12` (direction) x (source square)
- Drop: `3 x 12` (drop piece type) x (destination square)

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

1. If either player's king is checkmated, or
2. if either king enters the opponent's territory (farthest rank)
3. If the same position occurs three times.
4. If 250 moves have passed (a unique rule in Pgx).

In cases 3 and 4, the game is declared a draw.

## Version History

- `v2` : Fixed a bug in Pawn drop [#1218](https://github.com/sotetsuk/pgx/pull/1218) by [@KazukiOhta](https://github.com/KazukiOhta) (v2.3.0)
- `v1` : Fixed visualization [#1208](https://github.com/sotetsuk/pgx/pull/1208) and bug in Gold's move [#1209](https://github.com/sotetsuk/pgx/pull/1209) by [@KazukiOhta](https://github.com/KazukiOhta) (v2.2.0)
- `v0` : Initial release (v1.0.0)

## Baseline models

Pgx offers a baseline model for Animal Shogi. Users can use it for an anchor opponent in evaluation.
See [our paper](https://arxiv.org/abs/2303.17503) for more details. See [this colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/baselines.ipynb) for how to use it.

> [!WARNING]
> Curren latest model (`animal_shogi_v0`) is trained with `v0` environment. It may perform significantly worse in `v1` environment.

| Model ID | Description |
|:---:|:----|
| `animal_shogi_v0`| See [our paper](https://arxiv.org/abs/2303.17503) for the training details. |