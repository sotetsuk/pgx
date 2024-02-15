# Kuhn poker

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_light.gif" width="30%">
    </p>
    
## Description

Kuhn poker is a simplified poker with three cards: J, Q, and K.

## Rules

Each player is dealt one card and the remaining card is unused.
There are two actions: *bet* and *pass* and five possible scenarios.

1. `bet (1st) - bet (2nd)` : *Showdown* and the winner takes `+2`
2. `bet (1st) - pass (2nd)` : 1st player takes `+1`
3. `pass (1st) - pass (2nd)` : *Showdown* and the winner takes `+1`
4. `pass (1st) - bet (2nd) - bet (1st)` : *Showdown* and the winner takes `+2`
5. `pass (1st) - bet (2nd) - pass (1st)` : 2nd takes `+1`

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `2` |
| Observation shape | `(7,)` |
| Observation type | `bool` |
| Rewards | `{-2, -1, +1, +2}` |

## Observation

| Index | Description |
|:---:|:----|
| `[0]` | One if J in my hand |
| `[1]` | One if Q in my hand |
| `[2]` | One if K in my hand |
| `[3]` | One if 0 chip is bet by me |
| `[4]` | One if 1 chip is bet by me |
| `[5]` | One if 0 chip of the opponent |
| `[6]` | One if 1 chip of the opponent |

## Action
There are four distinct actions.

| Action | Index |
|:---|----:|
| Bet | 0|
| Pass | 1|


## Rewards
The winner takes `+2` or `+1` depending on the game payoff.
As Kuhn poker is zero-sum game, the loser takes `-2` or `-1` respectively.

## Termination
Follows the rules above.

## Version History

- `v0` : Initial release (v1.0.0)
