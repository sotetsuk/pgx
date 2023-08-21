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

Note that `BrdigeBidding` environment requires pre-computed Double Dummy Solver (DDS) results.
Please run the following command to download the DDS results provided by Pgx.

```py
from pgx.bridge_bidding import download_dds_results
download_dds_results()
```

You can specify which pre-coumpted DDS results to use by passing argument to `BridgeBidding` constructor.
Typically, you have to use different DDS results for training and testing (evaluation).

## Description

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

## Termination

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