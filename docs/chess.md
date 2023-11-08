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

> Chess is a board game for two players, called White and Black, each controlling an army of chess pieces in their color, with the objective to checkmate the opponent's king. It is sometimes called international chess or Western chess to distinguish it from related games such as xiangqi (Chinese chess) and shogi (Japanese chess). The recorded history of chess goes back at least to the emergence of a similar game, chaturanga, in seventh century India. The rules of chess as they are known today emerged in Europe at the end of the 15th century, with standardization and universal acceptance by the end of the 19th century. Today, chess is one of the world's most popular games played by millions of people worldwide.
> 
> Chess is an abstract strategy game that involves no hidden information and no elements of chance. It is played on a chessboard with 64 squares arranged in an 8×8 grid. At the start, each player controls sixteen pieces: one king, one queen, two rooks, two bishops, two knights, and eight pawns. White moves first, followed by Black. The game is won by checkmating the opponent's king, i.e. threatening it with inescapable capture. There are also several ways a game can end in a draw.
> 
> [Chess - Wikipedia](https://en.wikipedia.org/wiki/Chess)


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
P1 denotes the current player, and P2 denotes the opponent.

| Index | Description |
|:---:|:----|
| `[0:6]` | P1 board @ 0-steps before |
| `[6:12]` | P2 board @ 0-steps before |
| `[12:14]` | Repetitions @ 0-steps before |
| ... | (@ 1-7 steps before) |
| `[112]` | Color | 
| `[113]` | Total move count | 
| `[114:116]` | P1 castling | 
| `[116:118]` | P2 castling | 
| `[118]` | No progress count| 

## Action
We also follow the action design of AlphaZero `[Silver+18]`.
There are `4672 = 64 x 73` possible actions.
Each action represents

- 64 source position (`action // 73`), and
- 73 moves (`action % 73`)

Moves are defined by 56 queen moves, 8 knight moves, and 9 underpromotions.

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
4. Threefold repetition
5. `50` halfmoves are elapsed without any captures or pawn moves
6. `512` steps are elapsed (from AlphaZero `[Silver+18]`)

## Version History

- `v2` : Bug fix of wrong zobrist hash by [@sotetsuk](https://github.com/sotetsuk) in [#1078](https://github.com/sotetsuk/pgx/pull/1078) (v2.0.0) 
- `v1` : Bug fix when castling by [@HongruiTang](https://github.com/HongruiTang) in [#983](https://github.com/sotetsuk/pgx/pull/983) (v1.1.0) 
- `v0` : Initial release (v1.0.0)

!!! success "Well-tested"

    Pgx chess implementation is well-tested. You can confirm that its behavior is identical to the throughly-tested [OpenSpiel](https://github.com/deepmind/open_spiel) implementation by [this colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/check_chess.ipynb).

## Reference

- `[Silver+18]` "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" Science
