# Bridge bidding

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/bridge_bidding_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/bridge_bidding_light.gif" width="30%">
    </p>

## Usage

!!! warning "Bridge bidding requires domain knowledge"

    To appropriately use bridge bidding environment, you need to understand the rules of contract bridge well.
    To avoid wrong usage, we do not provide `pgx.make("bridge_bidding")`.
    Instead, you have to directly load `BridgeBidding` class.

```py
from pgx.bridge_bidding import BridgeBidding

env = BridgeBidding()
```

In Pgx, we follow `[Tian+20]` and use pre-computed Double Dummy Solver (DDS) dataset for each hand.
So, `BrdigeBidding` environment requires to load pre-computed DDS dataset by `env = BridgeBidding("<path_to_dataset>")`.
Please run the following command to download the DDS results provided by Pgx.

```py
from pgx.bridge_bidding import download_dds_results
download_dds_results()
```

You can specify which pre-coumpted DDS dataset to use by passing argument to `BridgeBidding` constructor.
Typically, you have to use different DDS datasets for training and testing (evaluation).
To make your own DDS datasets, follow the instruction at [sotetsuk/make-dds-dataset](https://github.com/sotetsuk/make-dds-dataset).

## Description

> Contract bridge, or simply bridge, is a trick-taking card game using a standard 52-card deck. In its basic format, it is played by four players in two competing partnerships,[1] with partners sitting opposite each other around a table. Millions of people play bridge worldwide in clubs, tournaments, online and with friends at home, making it one of the world's most popular card games, particularly among seniors.
> 
> ...
> 
The game consists of a number of deals, each progressing through four phases. The cards are dealt to the players; then the players call (or bid) in an auction seeking to take the contract, specifying how many tricks the partnership receiving the contract (the declaring side) needs to take to receive points for the deal. During the auction, partners use their bids to exchange information about their hands, including overall strength and distribution of the suits; no other means of conveying or implying any information is permitted. The cards are then played, the declaring side trying to fulfill the contract, and the defenders trying to stop the declaring side from achieving its goal. The deal is scored based on the number of tricks taken, the contract, and various other factors which depend to some extent on the variation of the game being played.
> 
> [Contract bridge](https://en.wikipedia.org/wiki/Contract_bridge)

We follow the previous works `[Rong+19,Gong+19,Tian+20,Lockhart+20]` and focus only on the bidding phase of contract bridge.
Therefore, we approximate the playing phase of bridge by using the results of DDS (Double Dummy Solver).

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `4` |
| Number of actions | `38` |
| Observation shape | `(480,)` |
| Observation type | `bool` |
| Rewards | Game payoff |

## Observation
We follow the observation design of `[Lockhart+20]`, OpenSpiel.

| Index | Description |
|:---:|:----|
| `obs[0:4]` | Vulnerability |
| `obs[4:8]` | Per player, did this player pass before the opening bid? |
| `obs[8:20]` | Per player played bid, double, redouble against 1♧ |
| ... | ... |
| `obs[416:428]` | Per player played bid, double, redouble against 7NT |
| `obs[428:480]` | 13-hot vector indicating the cards we hold |

## Action
Each action `(0, ..., 37)` corresponds to `(Pass, Double, Redouble, 1♧, 1♢, 1♡, 1♤, 1NT, ..., 7♧, 7♢, 7♡, 7♤, 7NT)`, respectively.

| Index | Description |
|:---:|:----|
| `0` | `Pass` |
| `1` | `Double` |
| `2` | `Redouble` |
| `3, ..., 7`  | `1♧, 1♢, 1♡, 1♤, 1NT` |
| ... | ... |
| `33, ..., 37`  | `7♧, 7♢, 7♡, 7♤, 7NT` |

## Rewards
Players get the game payoff at the end of the game.

## Termination
Terminates by three consecutive passes after the last bid.

## Version History

- `v0` : Initial release (v1.0.0)

## Reference

- `[Rong+19]` "Competitive Bridge Bidding with Deep Neural Networks"
- `[Gong+19]` "Simple is Better: Training an End-to-end Contract Bridge Bidding Agent without Human Knowledge"
- `[Tian+20]` "Joint Policy Search for Multi-agent Collaboration with Imperfect Information"
- `[Lockhart+20]` "Human-agent cooperation in bridge bidding"
- `Double Dummy Solver` http://privat.bahnhof.se/wb758135/
- `PBN format` https://www.tistis.nl/pbn/ 
- `IMP` https://en.wikipedia.org/wiki/International_Match_Points