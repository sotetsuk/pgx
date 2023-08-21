# Sparrow mahjong

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_dark.svg" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_light.svg" width="30%">
    </p>

## Description

Sparrow Mahjong ([すずめ雀](https://sugorokuya.jp/p/suzume-jong)) is a simplified version of Japanese Mahjong (Riichi Mahjong).
It was developed for those unfamiliar with Mahjong,
and requires similar strategic thinking to standard Japanese Mahjong.


### Rules of Sparrow Mahjong

<!---
すずめ雀のルールの概略は以下のようなものです。

  * 2-6人用
  * 牌はソウズと發中のみの11種44枚
  * 手牌は基本5枚、ツモ直後6枚
  * 順子か刻子を2つ完成で手牌完成
  * チーポンカンリーチはなし
  * ドラは表示牌がそのままドラ
  * 中はすべて赤ドラ、發は赤ドラなし、各牌ひとつ赤ドラがある
  * フリテンは自分が捨てた牌はあがれないが、他の牌ではあがれる
--->

The original rules of Sparrow Mahjong ([すずめ雀](https://sugorokuya.jp/p/suzume-jong)) are summarized as follows:

* For 2-6 players
* 11 types of tiles (44 tiles): 
    * 1-9 of bamboo, 1-9 of characters (x 4)
    * red dragon (x 4)
    * green dragon (x 4)
* Basically 5 tiles in hand, 6 tiles after drawing
* 2 sets of sequences or triplets to complete hand
* No chi (chow), pon (pong/pung), kan (kong), or riichi
* Dora is the same as the revealed tile
* Red doras: 
    * all red doragons are red dora (4 tiles)
    * all green doragons are NOT dora
    * one red dora for each banboo tile type (9 tiles)
* Furiten: you cannot *ron* with a tile you have discarded, but you can ron with other tiles

### Specifications in Pgx

Pgx implementation is simplified as follows:

* Only for 3 players
* Actions are only for discarding tiles (11 discrete actions)
  * If players can win, they automatically win
  * Players always keep red doras in their hands (i.e., red doras are not discarded if they have same but non-dora tiles)
* No [Heavenly hand](https://riichi.wiki/Tenhou_and_chiihou) (Tenhou/天和) to avoid the game ends without any action from players

## Specs

| Name | Value |
|:---|:----:|
| Version | `v1` |
| Number of players | `3` |
| Number of actions | `11` |
| Observation shape | `(11, 15)` |
| Observation type | `bool` |
| Rewards | `[-1, 1]` |

## Observation
There are 15 planes in the observation and each plane consists of 11 tiles.

| Planes | Description |
|:---:|:----|
| 4 | P1 hand | 
| 1 | Red dora in P1 hand | 
| 1 | Dora | 
| 1 | All discarded tiles by P1 |
| 1 | All discarded tiles by P2 |
| 1 | All discarded tiles by P3 | 
| 3 | Discarded tiles in the last 3 steps by P2 | 
| 3 | Discarded tiles in the last 3 steps by P3 |

## Action
Tile to discard.

## Rewards
Game payoff normalized to `[-1, 1]`

## Termination
Terminates when either player wins or the wall becomes empty.

## Version History

- `v1` : Change observation shape from `(15, 11)` to `(11, 15)` by [@sotetsuk](https://github.com/sotetsuk) in [#1010](https://github.com/sotetsuk/pgx/pull/1010) (v1.3.0)
- `v0` : Initial release (v1.0.0)