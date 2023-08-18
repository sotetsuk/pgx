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


### Rules

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

### Modifications in Pgx

Pgx implementation is simplified as follows:

* Only for 3 players
* If players can win, they automatically win
* Players always keep red doras in their hands (i.e., red doras are not discarded if they have same but non-dora tiles)
* No [Heavenly hand](https://riichi.wiki/Tenhou_and_chiihou) (Tenhou/天和) to avoid the game ends without any action from players