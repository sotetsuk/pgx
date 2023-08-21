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

> Backgammon ...
> 
> [Wikipedia](https://en.wikipedia.org/wiki/Backgammon)

## Specs

| Name | Value |
|:---|:----:|
| Version | `v1` |
| Number of players | `2` |
| Number of actions | `162 (= 6 * 26 + 6)` |
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
| `[:24]` | represents  |
| `[24:28]` | represents  |
| `[28:34]` | is one-hot vector of playable dice |

## Action
...

## Rewards
...

## Termination
...


## Version History

- `v0` : Initial release (v1.0.0)

## Reference

1. `[Antonoglou+22]` "Planning in Stochastic Environments with a Learned Model", ICLR
