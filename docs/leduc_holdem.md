# Leduc hold’em

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_light.gif" width="30%">
    </p>

## Description
Leduc hold’em is a simplified poker proposed in [Souhty+05].

## Rules
We quote the description in  [Souhty+05]:

> **Leduc Hold ’Em.** We have also constructed a smaller
version of hold ’em, which seeks to retain the strategic elements of the large game while keeping the size of the game
tractable. In Leduc hold ’em, the deck consists of two suits
with three cards in each suit. There are two rounds. In the
first round a single private card is dealt to each player. In
the second round a single board card is revealed. There is
a two-bet maximum, with raise amounts of 2 and 4 in the
first and second round, respectively. Both players start the
first round with 1 already in the pot.


> ![](assets/leduc_holdem_tree.png)
> 
> Figure 1: An example decision tree for a single betting
round in poker with a two-bet maximum. Leaf nodes with
open boxes continue to the next round, while closed boxes
end the hand.

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `4` |
| Observation shape | `(7,)` |
| Observation type | `bool` |
| Rewards | `{-13, -12, ... 0, ..., 12, 13}` |

## Observation

| Index | Description |
|:---:|:----|
| `[0]`  | True if J in hand |
| `[1]`  | True if Q in hand |
| `[2]`  | True if K in hand |
| `[3]`  | True if J is the public card | 
| `[4]`  | True if Q is the public card | 
| `[5]`  | True if K is the public card | 
| `[6:19]` | represent my chip count (0, ..., 13) |
| `[20:33]`| represent opponent's chip count (0, ..., 13) |

## Action

There are four distinct actions.

| Index | Action | 
|:---|----:|
| 0 | Call  |
| 1 | Raise |
| 2 | Fold  |

## Rewards
The reward is the payoff of the game.

## Termination
Follows the rules above.

## Version History

- `v0` : Initial release (v1.0.0)

## References

- [Souhty+05] [Bayes' Bluff: Opponent Modelling in Poker](https://arxiv.org/abs/1207.1411) UAI2005
