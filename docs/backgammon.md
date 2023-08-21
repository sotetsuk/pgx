# Backgammon

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("backgammon")
```

or you can directly load `Backgammon` class

```py
from pgx.backgammon import Backgammon

env = Backgammon()
```

## Description

> Backgammon is a two-player board game played with counters and dice on tables boards. It is the most widespread Western member of the large family of tables games, whose ancestors date back nearly 5,000 years to the regions of Mesopotamia and Persia. The earliest record of backgammon itself dates to 17th-century England, being descended from the 16th-century game of Irish.
> 
> Backgammon is a two-player game of contrary movement in which each player has fifteen pieces known traditionally as men (short for 'tablemen'), but increasingly known as 'checkers' in the US in recent decades, analogous to the other board game of Checkers. The backgammon table pieces move along twenty-four 'points' according to the roll of two dice. The objective of the game is to move the fifteen pieces around the board and be first to bear off, i.e., remove them from the board. The achievement of this while the opponent is still a long way behind results in a triple win known as a backgammon, hence the name of the game.
> 
> Backgammon involves a combination of strategy and luck (from rolling dice). While the dice may determine the outcome of a single game, the better player will accumulate the better record over a series of many games. With each roll of the dice, players must choose from numerous options for moving their pieces and anticipate possible counter-moves by the opponent. The optional use of a doubling cube allows players to raise the stakes during the game.
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Backgammon)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v1` |
| Number of players | `2` |
| Number of actions | `156 (= 6 * 26)` |
| Observation shape | `(34,)` |
| Observation type | `int` |
| Rewards | `{-3, -2, -1, 0, 1, 2, 3}` |

## Observation

The first `28` observation dimensions follow `[Antonoglou+22]`:

> In our backgammon experiments, the board was represented using a vector of size 28, with the first
24 positions representing the number of chips for each player in the 24 possible points on the board,
and the last four representing the number of hit chips and born off chips for each of the two players.
We used positive numbers for the current player’s chips and negative ones for her opponent.

| Index | Description |
|:---:|:----|
| `[:24]` | Number of checkers on each position |
| `[24:26]` | Number of checkers on bar |
| `[26:28]` | Number of checkers brone off |
| `[28:34]` | One-hot vector of playable dice |

## Action

Action design is same as micro-action in `[Antonoglou+22]`:

> An action in our implementation consists of 4 micro-actions, the same as the maximum number of dice a player can play at each turn.  Each micro-action encodes the source position of a chip along with the value of the die used. We consider 26 possible source positions, with the 0th position corresponding to a no-op, the 1st to retrieving a chip from the hit pile, and the remaining to selecting a chip in one of the 24 possible points.   Each micro-action is encoded as a single integer with micro-action = src · 6 + die.

The difference is that after every micro-action, the state transition happens.
The turn of the same player can continue up to 4 times.

## Rewards
The game payoff is rewarded.

## Termination
Continues until either player wins.

## Version History

- `v1`: Remove redundant actions (From `162` to `156`) by [@nissymori](https://github.com/nissymori) in [#1004](https://github.com/sotetsuk/pgx/pull/1004)
- `v0` : Initial release (v1.0.0)

## Reference

1. `[Antonoglou+22]` "Planning in Stochastic Environments with a Learned Model", ICLR
