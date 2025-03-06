# Shogi

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif" width="30%">
    </p>

## Usage

```py
import pgx

env = pgx.make("shogi")
```

or you can directly load `Shogi` class

```py
from pgx.shogi import Shogi

env = Shogi()
```

## Description

TBA


## Specs

| Name | Value |
|:---|:----:|
| Version | `v1` |
| Number of players | `2` |
| Number of actions | `2187` |
| Observation shape | `(9, 9, 119)` |
| Observation type | `bool` |
| Rewards | `{-1, 0, 1}` |

## Observation
We follow the observation design of [dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi), an open-source shogi AI. 
Ther original dlshogi implementations are [here](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/1053b87e73c5eb69b66c9937c984d02afe8d1a3b/cppshogi/cppshogi.cpp#L39-L123).
Pgx implementation has `[9, 9, 119]` shape and `[:, :, x]` denotes:

| `x` | Description |
|---:|:----|
| `0:14` | Where my piece `x` exists |
| `14:28` | Where my pieces `x` are attacking |
| `28:31` | Where the number of my attacking pieces are `>= 1,2,3` respectively|
| `31:45` | Where opponent's piece `x` exists |
| `45:59` | Where opponent's pieces `x` are attacking |
| `59:62` | Where the number of opponent's attacking pieces are `>= 1,2,3` respectively|

The following planes are all ones ore zeros

| `x` | Description |
|---:|:----|
| `62:70`   | My hand has `>= 1, ..., 8` Pawn |
| `70:74`   | My hand has `>= 1, 2, 3, 4` Lance |
| `74:78`   | My hand has `>= 1, 2, 3, 4` Knight |
| `78:82`   | My hand has `>= 1, 2, 3, 4` Silver |
| `82:86`   | My hand has `>= 1, 2, 3, 4` Gold |
| `86:88`   | My hand has `>= 1, 2` Bishop |
| `88:90`   | My hand has `>= 1, 2` Rook |
| `90:98`   | Oppnent's hand has `>= 1, ..., 8` Pawn |
| `98:102`  | Oppnent's hand has `>= 1, 2, 3, 4` Lance |
| `102:106` | Oppnent's hand has `>= 1, 2, 3, 4` Knight |
| `106:110` | Oppnent's hand has `>= 1, 2, 3, 4` Silver |
| `110:114` | Oppnent's hand has `>= 1, 2, 3, 4` Gold |
| `114:116` | Oppnent's hand has `>= 1, 2` Bishop |
| `116:118` | Oppnent's hand has `>= 1, 2` Rook |
| `118` | Ones if checked |

Note that piece ids are

| Piece | Id |
| :--- | ---: |
| 歩　 `PAWN`  | `0` |
| 香　 `LANCE`  | `1` |
| 桂　 `KNIGHT`  | `2` |
| 銀　 `SILVER`  | `3` |
| 角　 `BISHOP`  | `4` |
| 飛　 `ROOK` | `5` | 
| 金　 `GOLD` | `6` | 
| 玉　 `KING` | `7` | 
| と　 `PRO_PAWN` | `8` |
| 成香 `PRO_LANCE` | `9` | 
| 成桂 `PRO_KNIGHT` | `10` | 
| 成銀 `PRO_SILVER` | `11` | 
| 馬　 `HORSE` | `12` | 
| 龍　 `DRAGON` | `13` | 

## Action

The design of action also follows that of dlshogi.
There are `2187 = 81 x 27` distinct actions.
The action can be decomposed into 

- `direction` from which the piece moves and
- `destination` to which the piece moves

by `direction, destination = action // 81, action % 81`.
The `direction` is encoded by

| id | direction |
| ---: | :--- |
|  0 | Up |
|  1 | Up left |
|  2 | Up right |
|  3 | Left |
|  4 | Right |
|  5 | Down |
|  6 | Down left |
|  7 | Down right |
|  8 | Up2 left |
|  9 | Up2 right |
| 10 | Promote + Up |
| 11 | Promote + Up left |
| 12 | Promote + Up right |
| 13 | Promote + Left |
| 14 | Promote + Right |
| 15 | Promote + Down |
| 16 | Promote + Down left |
| 17 | Promote + Down right |
| 18 | Promote + Up2 left |
| 19 | Promote + Up2 right |
| 20 | Drop Pawn |
| 21 | Drop Lance |
| 22 | Drop Knight |
| 23 | Drop Silver |
| 24 | Drop Bishop |
| 25 | Drop Rook |
| 26 | Drop Gold |


## Rewards
Non-zero rewards are given only at the terminal states.
The reward at terminal state is described in this table:

| | Reward |
|:---|:----:|
| Win | `+1` |
| Lose | `-1` |
| Draw | `0` |

## Termination

Termination occurs when 

1. either player checkmates the opponent, or
2. `512` steps are elapsed (from AlphaZero `[Silver+18]`)

Fourfold repetition is not implemented in `v0`.

## Version History

- `v1` : Bug fix in current player by [@KazukiOta](https://github.com/KazukiOta) in [#1298](https://github.com/sotetsuk/pgx/pull/1298) (v2.6.0) 
- `v0` : Initial release (v1.0.0)
