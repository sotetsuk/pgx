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
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `162 (= 6 * 26 + 6)` |
| Observation shape | `(34,)` |
| Observation type | `int` |
| Rewards | `{-3, -2, -1, 0, 1, 2, 3}` |

## Observation

The first `28` observation dimensions follow `[Antonoglou+22]`:

> An action in our implementation consists of 4 micro-actions, the same as the maximum number
of dice a player can play at each turn. Each micro-action encodes the source position of a chip
along with the value of the die used. We consider 26 possible source positions, with the 0-th position corresponding to a no-op, the 1st to retrieving a chip from the hit pile, and the remaining to selecting a chip in one of the 24 possible points. Each micro-action is encoded as a single integer with micro-action = `src · 6 + die`.

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

1. `[Antonoglou+22]` "Planning in Stochastic Environments with a Learned Modell", ICLR