import numpy as np
import jax
import jax.numpy as jnp

from pgx.mahjong._full_mahjong import State, Deck, Hand, Action, Yaku, Meld, Tile, step
from pgx.mahjong import full_mahjong

import tenhou_wall_reproducer

def test_tile():
    Tile.to_str(Tile.from_str("1m")) == "1m"
    Tile.to_str(Tile.from_str("2m")) == "2m"
    Tile.to_str(Tile.from_str("9m")) == "9m"
    Tile.to_str(Tile.from_str("0m")) == "0m"
    Tile.to_str(Tile.from_str("5m")) == "5m"
    Tile.to_str(Tile.from_str("1p")) == "1p"
    Tile.to_str(Tile.from_str("1s")) == "1s"
    Tile.to_str(Tile.from_str("1z")) == "1z"

def test_meld():
    Meld.to_str(Meld.from_str("[1]11m")) == "[1]11m"
    Meld.to_str(Meld.from_str("2[2]2p")) == "2[2]2p"
    Meld.to_str(Meld.from_str("[3]45m")) == "[3]45m"
    Meld.to_str(Meld.from_str("[3]24m")) == "[3]24m"
    Meld.to_str(Meld.from_str("[3]12m")) == "[3]12m"
    Meld.to_str(Meld.from_str("[3]40m")) == "[3]40m"
    Meld.to_str(Meld.from_str("[0]46p")) == "[0]46p"
    Meld.to_str(Meld.from_str("5[0]55m")) == "5[0]55m"
    Meld.to_str(Meld.from_str("0[5]55m")) == "0[5]55m"
    Meld.to_str(Meld.from_str("0[55]5m")) == "0[55]5m"
    Meld.to_str(Meld.from_str("5[50]5m")) == "5[50]5m"
    Meld.to_str(Meld.from_str("5[05]5m")) == "5[05]5m"
    Meld.to_str(Meld.from_str("5555m")) == "5555m"
    Meld.to_str(Meld.from_str("2[2]22z")) == "2[2]22z"
    Meld.to_str(Meld.from_str("2222z")) == "2222z"


def test_deck():
    deck = Deck.init()
    deck, tile1 = Deck.draw(deck)
    deck, tile2 = Deck.draw(deck)
    deck, tile3 = Deck.draw(deck)
    deck, tile4 = Deck.draw(deck)
    deck, tile5 = Deck.draw(deck)
    assert tile1 != tile2 or tile2 != tile3 or tile3 != tile4 or tile4 != tile5


def test_hand():
    hand = np.zeros(34, dtype=np.uint8)
    red = np.full(3, False)
    hand, red = Hand.add(hand, red, 0)
    assert Hand.can_ron(hand, 0)
    assert not Hand.can_ron(hand, 1)
    hand, red = Hand.add(hand, red, 0)
    assert Hand.can_tsumo(hand)
    hand, red = Hand.add(hand, red, tile=1, x=2)
    assert Hand.can_steal(hand, red, tile=0, action=Action.PON)

    assert Hand.can_steal(hand, red, tile=2, action=Action.CHI_R)
    assert Hand.can_steal(hand, red, tile=1, action=Action.CHI_M) == False
    hand, red = Hand.add(hand, red, tile=2, x=3)  # 1122333m
    assert Hand.can_steal(hand, red, tile=1, action=Action.CHI_M)

    hand, red = Hand.steal(hand, red, tile=1, action=Action.CHI_M)
    # 12233m
    hand, red = Hand.sub(hand, red, tile=1)
    # 1233m
    hand = Hand.steal(hand, red, tile=2, action=Action.PON)
    # 12m

    hand = np.array([
        3,1,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.can_tsumo(hand)

    hand = np.array([
        3,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.is_tenpai(hand)

    hand = np.array([
        3,0,1,1,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.is_tenpai(hand)

    hand = np.array([
        3,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0
        ])
    assert Hand.can_riichi(hand)

    hand = np.array([
        3,0,1,1,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0
        ])
    assert not Hand.can_riichi(hand)

    hand = np.array([
        2,0,0,0,0,0,1,1,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.can_steal(hand, red, tile=5, action=Action.CHI_L)
    assert not Hand.can_steal(hand, red, tile=5, action=Action.CHI_M)
    assert not Hand.can_steal(hand, red, tile=5, action=Action.CHI_R)

    assert np.all(
            np.array([
                    2,0,0,0,0,0,0,0,1,
                    1,0,0,0,1,2,1,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,3,0
                ]) ==
            Hand.from_str("119m15667p666z")[0]
            )

    hand, red = Hand.from_str("117788m2233s6677z")
    assert Hand.can_tsumo(hand)

    hand, red = Hand.from_str("19m19p199s1234567z")
    assert Hand.can_tsumo(hand)

def fu(
        hand_s: str,
        melds_s: str,
        last_s: str,
        is_ron: bool = False
        ) -> int:
    hand, _ = Hand.from_str(hand_s)
    melds = np.zeros(4, dtype=np.int32)
    n_meld=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[n_meld] = Meld.from_str(s)
        melds
        n_meld += 1
    last = Tile.from_str(last_s)
    return Yaku.judge(hand, melds, n_meld, last, False, is_ron)[2]

def test_yaku_fu():
    # リャンメン(平和)
    assert fu("112233567m34588p", "", "1m") == 20
    assert fu("112233567m34588p", "", "1m", is_ron=True) == 30

    # 単騎
    assert fu("112233567m34588p", "", "8p") == 30
    assert fu("112233567m34588p", "", "8p", is_ron=True) == 40

    # ペンチャン
    assert fu("112233567m34588p", "", "3m") == 30
    assert fu("112233567m34588p", "", "3m", is_ron=True) == 40

    # カンチャン
    assert fu("112233567m34588p", "", "2m") == 30
    assert fu("112233567m34588p", "", "2m", is_ron=True) == 40

    assert fu("111222333567m88p", "", "1m") == 40
    # 平和一盃口 == 三暗刻. 符で三暗刻が優先される.

    assert fu("233445m23456799p", "", "4m") == 20
    # 平和がつくのでリャンメンにとる
    assert fu("233445m22234599p", "", "4m") == 30
    # 平和がつかないのでカンチャンにとる

    assert fu("11123456m111234p", "", "1m") == 40
    # 単騎 > リャンメン

    assert fu("111123456m11222p", "", "1m", is_ron=True) == 50
    # リャンメン > シャンポン

    assert fu("123456789m77p", "[3]45p", "1m", is_ron=True) == 30
    # 鳴き平和は例外的に30符扱い

    assert fu("123456789m66677z", "", "1m") == 40
    # 役牌の雀頭は2符

    assert fu("123456789m11666z", "", "1m", is_ron=True) == 50
    # 連風牌の雀頭は4符

    assert fu("123m11666z", "9999m,1111s", "6z", is_ron=True) == 110
    # 1翻110符


def has_yaku(
        yaku: int,
        hand_s: str,
        melds_s: str,
        last_s: str,
        is_ron: bool = False
        ) -> bool:
    hand, _ = Hand.from_str(hand_s)
    melds = np.zeros(4, dtype=np.int32)
    n_meld=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[n_meld] = Meld.from_str(s)
        n_meld += 1
    last = Tile.from_str(last_s)
    riichi = False
    return Yaku.judge(hand, melds, n_meld, last, riichi, is_ron)[0][yaku]

def test_yaku_tanyao():
    assert has_yaku(Yaku.断么九, "23456777m678p", "[4]23m", "2m")

    assert has_yaku(Yaku.断么九, "23456777m678p", "[3]12m", "2m") == False

    assert has_yaku(Yaku.断么九, "23456777m678p", "[5]55m", "2m")

    assert has_yaku(Yaku.断么九, "23456777m678p", "[6]66z", "2m") == False

def test_yaku_flush():
    assert has_yaku(Yaku.混一色, "11223355577789m", "", "1m") == False
    assert has_yaku(Yaku.清一色, "11223355577789m", "", "1m")

    assert has_yaku(Yaku.混一色, "11223355577m789p", "", "1m") == False
    assert has_yaku(Yaku.清一色, "11223355577m789p", "", "1m") == False

    assert has_yaku(Yaku.混一色, "11223355577m777z", "", "1m")
    assert has_yaku(Yaku.清一色, "11223355577m777z", "", "1m") == False

    assert has_yaku(Yaku.混一色, "11223355577m", "[7]89m", "1m") == False
    assert has_yaku(Yaku.清一色, "11223355577m", "[7]89m", "1m")

    assert has_yaku(Yaku.混一色, "11223355577m", "[7]89p", "1m") == False
    assert has_yaku(Yaku.清一色, "11223355577m", "[7]89p", "1m") == False

    assert has_yaku(Yaku.混一色, "11223355577m", "7[7]7z", "1m")
    assert has_yaku(Yaku.清一色, "11223355577m", "7[7]7z", "1m") == False


def test_yaku_without_tanyao():
    assert has_yaku(Yaku.混老頭, "111999m11p", "6[6]66z,7777z", "1m")
    assert has_yaku(Yaku.混全帯么九, "111999m11p", "6[6]66z,7777z", "1m") == False

    assert has_yaku(Yaku.混老頭, "123999m11p", "6[6]66z,7777z", "1m") == False
 
 
def test_yaku_dragons():
    assert has_yaku(Yaku.小三元, "123m456p55566677z", "", "1m")
    assert has_yaku(Yaku.白, "123m456p55566677z", "", "1m")
    assert has_yaku(Yaku.發, "123m456p55566677z", "", "1m")
    assert has_yaku(Yaku.中, "123m456p55566677z", "", "1m") == False

    assert has_yaku(Yaku.大三元, "11m456p555666z", "7[77]7z", "1m")
    assert has_yaku(Yaku.白, "11m456p555666z", "7[77]7z", "1m") == False
    assert has_yaku(Yaku.發, "11m456p555666z", "7[77]7z", "1m") == False
    assert has_yaku(Yaku.中, "11m456p555666z", "7[77]7z", "1m") == False

def test_yaku_winds():
    assert has_yaku(Yaku.場風, "12345666m789p111z", "", "1m")
    assert has_yaku(Yaku.自風, "12345666m789p111z", "", "1m")

    assert has_yaku(Yaku.大四喜, "11m222333444z", "11[1]z", "1m")
    assert has_yaku(Yaku.小四喜, "11m222333444z", "11[1]z", "1m") == False

    assert has_yaku(Yaku.大四喜, "111m22333444z", "11[1]z", "1m") == False
    assert has_yaku(Yaku.小四喜, "111m22333444z", "11[1]z", "1m")

def test_yaku_tsumo():
    assert has_yaku(Yaku.門前清自摸和, "12345666m778899p", "", "1m")
    assert has_yaku(Yaku.門前清自摸和, "12345666m778899p", "", "1m", is_ron=True) == False

    assert has_yaku(Yaku.門前清自摸和, "12345666m789p", "[9]78p", "1m") == False

def test_yaku_nine_gates():
    assert has_yaku(Yaku.九蓮宝燈, "11123455678999m", "", "1m")
    assert has_yaku(Yaku.九蓮宝燈, "11123455678m", "[9]99m", "1m") == False

def test_yaku_thirteen_orphans():
    assert has_yaku(Yaku.国士無双, "19m19p19s12345677z", "", "1m")
    assert has_yaku(Yaku.国士無双, "19m19p179s1234567z", "", "1m") == False

def test_yaku_one_and_nines():
    assert has_yaku(Yaku.清老頭, "111999m111999p11s", "", "1m")
    assert has_yaku(Yaku.混老頭, "111999m111999p11s", "", "1m") == False

    assert has_yaku(Yaku.清老頭, "111999m111p11s", "7[7]7z", "1m") == False
    assert has_yaku(Yaku.混老頭, "111999m111p11s", "7[7]7z", "1m")

    assert has_yaku(Yaku.混老頭, "111999m111p11s", "[1]23m", "1m") == False
    assert has_yaku(Yaku.純全帯么九, "111999m111p11s", "[1]23m", "1m")

def test_yaku_all_honors():
    assert has_yaku(Yaku.字一色, "11122233366677z", "", "7z")

    assert has_yaku(Yaku.字一色, "111m22233366677z", "", "7z") == False

def test_yaku_all_green():
    assert has_yaku(Yaku.緑一色, "223344666s666z", "[8]88s", "6z")
    assert has_yaku(Yaku.緑一色, "223344666s666z", "[7]77s", "6z") == False

def test_yaku_four_concealed_pungs():
    assert has_yaku(Yaku.四暗刻, "11122233344455m", "", "1m")
    assert has_yaku(Yaku.四暗刻, "11122233344455m", "", "1m", is_ron=True) == False
    assert has_yaku(Yaku.四暗刻, "11122233344455m", "", "5m", is_ron=True)
 
 
def test_yaku_pinfu():
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "1m")
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "2m") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "3m") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "4m")
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "5m") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "6m")
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "7p") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "8p") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "9p")
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "1s")
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "2s") == False
    assert has_yaku(Yaku.平和, "12345666m789p123s", "", "3s") == False

    assert has_yaku(Yaku.平和, "11223344556677m", "", "1m")
    assert has_yaku(Yaku.平和, "11223344556677m", "", "2m")
    assert has_yaku(Yaku.平和, "11223344556677m", "", "3m") == False
    assert has_yaku(Yaku.平和, "11223344556677m", "", "4m")
    assert has_yaku(Yaku.平和, "11223344556677m", "", "5m")
    assert has_yaku(Yaku.平和, "11223344556677m", "", "6m")
    assert has_yaku(Yaku.平和, "11223344556677m", "", "7m")

    assert has_yaku(Yaku.平和, "22334455667788m", "", "2m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "3m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "4m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "5m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "6m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "7m")
    assert has_yaku(Yaku.平和, "22334455667788m", "", "8m")


def test_yaku_outside():
    assert has_yaku(Yaku.純全帯么九, "11223399m789p123s", "", "1m")
    assert has_yaku(Yaku.混全帯么九, "11223399m789p123s", "", "1m") == False

    assert has_yaku(Yaku.純全帯么九, "11123m789p123s", "33[3]z", "1m") == False
    assert has_yaku(Yaku.混全帯么九, "11123m789p123s", "33[3]z", "1m")

    assert has_yaku(Yaku.純全帯么九, "11223399m789p", "[2]13s", "1m")
    assert has_yaku(Yaku.混全帯么九, "11223399m789p", "[2]13s", "1m") == False

    assert has_yaku(Yaku.純全帯么九, "11223399m789p", "[2]34s", "1m") == False
    assert has_yaku(Yaku.混全帯么九, "11223399m789p", "[2]34s", "1m") == False

    assert has_yaku(Yaku.純全帯么九, "11223399m789p", "[2]22z", "1m") == False
    assert has_yaku(Yaku.混全帯么九, "11223399m789p", "[2]22z", "1m")

def test_yaku_double_chows():
    assert has_yaku(Yaku.一盃口, "11223388m789p123s", "", "1m")
    assert has_yaku(Yaku.一盃口, "11223388m789p", "[3]12s", "1m") == False

    assert has_yaku(Yaku.一盃口, "11223388m778899p", "", "1m") == False
    assert has_yaku(Yaku.二盃口, "11223388m778899p", "", "1m")
    assert has_yaku(Yaku.一盃口, "11223388m", "[7]89p,[8]79p", "1m") == False
    assert has_yaku(Yaku.二盃口, "11223388m", "[7]89p,[8]79p", "1m") == False

def test_is_pure_straight():
    assert has_yaku(Yaku.一気通貫, "12345678999m345p", "", "1m")

    assert has_yaku(Yaku.一気通貫, "12345699m345789p", "", "1m") == False

    assert has_yaku(Yaku.一気通貫, "99m123456p345s", "[9]78p", "9m")
    assert has_yaku(Yaku.一気通貫, "99m123456p345s", "[9]78s", "9m") == False


def test_yaku_triple_chow():
    assert has_yaku(Yaku.三色同順, "112233m123p12388s", "", "1m")
    assert has_yaku(Yaku.三色同順, "112233m123p23488s", "", "1m") == False

    assert has_yaku(Yaku.三色同順, "112233m12388s", "[1]23p", "1m")
    assert has_yaku(Yaku.三色同順, "112233m23488s", "[1]23p", "1m") == False


def test_all_pungs():
    assert has_yaku(Yaku.対々和, "11133m111p", "2[2]2s,7777z", "1m")

    assert has_yaku(Yaku.対々和, "11133m111p", "2[2]2s,[2]34s", "1m") == False

    assert has_yaku(Yaku.対々和, "11133m111789p", "2[2]2s", "1m") == False
    assert has_yaku(Yaku.対々和, "11133m111777p", "2[2]2s", "1m")

def test_yaku_triple_pung():
    assert has_yaku(Yaku.三色同刻, "111333m111p11188s", "", "1m", is_ron=True)
    assert has_yaku(Yaku.三色同刻, "111333m222p11188s", "", "1m", is_ron=True) == False

    assert has_yaku(Yaku.三色同刻, "222333m22288s", "[2]22p", "2m")
    assert has_yaku(Yaku.三色同刻, "222333m22288s", "[3]33p", "2m") == False

def test_yaku_three_concealed_pungs():
    assert has_yaku(Yaku.三暗刻, "11122233344455m", "", "1m", is_ron=True)
    assert has_yaku(Yaku.三暗刻, "11122233344455m", "", "5m", is_ron=True) == False
    assert has_yaku(Yaku.三暗刻, "11122233344455m", "", "1m") == False

    assert has_yaku(Yaku.三暗刻, "111333m66677z", "[3]33p", "1m")
    assert has_yaku(Yaku.三暗刻, "111333m66677z", "[3]33p", "1m", is_ron=True) == False
    assert has_yaku(Yaku.三暗刻, "111333m66677z", "[3]33p", "7z")
    assert has_yaku(Yaku.三暗刻, "111333m66677z", "[3]33p", "7z", is_ron=True)

    assert has_yaku(Yaku.三暗刻, "111123m11166677z", "", "1m", is_ron=True)
    assert has_yaku(Yaku.三暗刻, "111234m11166677z", "", "1m", is_ron=True) == False

def test_yaku_seven_pairs():
    assert has_yaku(Yaku.七対子, "113355m778899p33z", "", "1m")
    assert has_yaku(Yaku.七対子, "12355m778899p333z", "", "1m") == False
    assert has_yaku(Yaku.七対子, "112233445566m88p", "", "1m") == False
    # 二盃口 < 七対子

    assert has_yaku(Yaku.七対子, "1122445566m3344z", "", "1m")
    assert has_yaku(Yaku.混一色, "1122445566m3344z", "", "1m")

def test_yaku_coner_cases():
    assert has_yaku(Yaku.三暗刻, "111222333789m22z", "", "2m") == False
    assert has_yaku(Yaku.一盃口, "111222333789m22z", "", "2m")
    assert has_yaku(Yaku.混全帯么九, "111222333789m22z", "", "2m")
    # 面前チャンタ一盃口 > 三暗刻

    assert has_yaku(Yaku.三暗刻, "111222333m11p", "[8]79m", "1m")
    assert has_yaku(Yaku.純全帯么九, "111222333m11p", "[8]79m", "1m") == False
    # 副露純チャン < 三暗刻 (符で有利)

    assert has_yaku(Yaku.平和, "11222333444456m", "", "1m") == False
    assert has_yaku(Yaku.一盃口, "11222333444456m", "", "1m") == False
    assert has_yaku(Yaku.三暗刻, "11222333444456m", "", "1m")
    # 平和一盃口 < 三暗刻 (符で有利)

    assert has_yaku(Yaku.平和, "22334455m234p234s", "", "3m") == False
    assert has_yaku(Yaku.一盃口, "22334455m234p234s", "", "3m")
    assert has_yaku(Yaku.三色同順, "22334455m234p234s", "", "3m")
    # 平和と三色が両立しないケース

    # 234m, 234m, 234m, 456m, 66m
    assert has_yaku(Yaku.平和, "22233344445666m", "", "2m", is_ron=True)
    assert has_yaku(Yaku.三暗刻, "22233344445666m", "", "2m", is_ron=True) == False

    # 222m, 333m, 444m, 456m, 66m
    assert has_yaku(Yaku.平和, "22233344445666m", "", "2m") == False
    assert has_yaku(Yaku.三暗刻, "22233344445666m", "", "2m")


def score(
        hand_s: str,
        melds_s: str,
        last_s: str,
        riichi: bool = False,
        is_ron: bool = False
        ) -> int:
    hand, red = Hand.from_str(hand_s)
    dora = np.zeros(34, dtype=np.uint8)
    melds = np.zeros(4, dtype=np.int32)
    n_meld=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[n_meld] = Meld.from_str(s)
        n_meld += 1
    last = Tile.from_str(last_s)
    return Yaku.score(hand, red, melds, n_meld, last, riichi, is_ron, dora)

def roundup(n):
    return n + (-n % 100)

def test_yaku_score():
    assert roundup(score("11123456789999m", "", "5m", riichi=True, is_ron=True)) == 8000
    # 九蓮宝燈
    assert roundup(score("11122233344455m", "", "1m", riichi=True)) == 8000
    # 四暗刻
    assert roundup(score("11122233344455m", "", "1m", riichi=True, is_ron=True)) == 6000
    # 清一色,三暗刻,対々和,立直=11翻
    assert roundup(score("11122233344455m", "", "1m", is_ron=True)) == 4000
    # 清一色,三暗刻,対々和=10翻

    s = score("112233m123p12333s", "", "2m")
    assert roundup(s) == 2000
    assert roundup(s * 2) == 3900
    # ツモ,一盃口,三色=4翻,30符 => 2000,3900 / 3900オール

    s = score("112233m7788p3344s", "", "1m", riichi=True)
    assert roundup(s) == 1600
    assert roundup(s * 2) == 3200
    # ツモ,七対子,立直=4翻,25符 => 1600,3200 / 3200オール

SEED = "zmsk28otF+PUz4E7hyyzUN0fvvn3BO6Ec3fZfvoKX1ATIhkPO8iNs9yH6pWp+lvKcYsXccz1oEJxJDbuPL6qFpPKrjOe/PCBMq1pQdW2c2JsWpNSRdOCA6NABD+6Ty4pUZkOKbWDrWtGxKPUGnKFH2NH5VRMqlbo463I6frEgWrCkW3lpazhuVT1ScqAI8/eCxUJrY095I56NKsw5bGgYPARsE4Sibrk44sAv3F42/Q3ohmb/iXFCilBdfE5tNSg55DMu512CoOwd2bwV7U0LctLgl9rj6Tv6K3hOtcysivTjiz+UGvJPT6R/VTRX/u1bw6rr/SuLqOAx0Dbl2CC1sjKFaLRAudKnr3NAS755ctPhGPIO5Olf9nJZiDCRpwlyzCdb8l7Jh3VddtqG9GjhSrqGE0MqlR2tyi+R3f1FkoVe8+ZIBNt1A1XigJeVT//FsdEQYQ2bi4kG8jwdlICgY2T0Uo2BakfFVIskFUKRNbFgTLqKXWPTB7KAAH/P4zBW1Qtqs9XuzZIrDrak9EXt/4nO0PYVTCjC1B+DE/ZlqgO8SoGeJRz/NbAp6gxe0H1G7UQ+tr2QfZUA1jDUInylosQDufKpr0gPQMQepVI6XjpWkNrVu6zFwedN1W8gUSd6uDKb83QS49/pXSBWmEXSDC8dWs0a1SopdbroqZxoVfg2QUuwdMa7LHQ71fg63yYMXErIa9mci58CEMQnqsgczMaVyNClb7uWdR3e4i5DRgaF2rENuM0wT8Ihm49Z1HLbmqkiHJLQ9t7RaQP+M51GMBc53ygBsgA2TCEsXCBYMM1nhO5IVuZ0+Xu2iJvl2TeBM5UZD7NYECo6WqfRlsy1+/pNCFOBucFuChWqITn9bwAsVu1Th+2r2DHoN+/JO1b2cRcr4vzG5ci5r0n6BObhPtSAYif4fhbqAsOiEAWHQWJRuAZfS2XbIu7Ormi0LxIhRoX5zZwU26MJud1yVsf6ZQD0GQF2TqZkHrqbr9ey2QojNHernYv0JA1pqIIfEuxddQwYh5FJgcmdwbKUzIubGUn/FnbWPQiJuAoGU/3qiC6Y5VbEUazRvRufbABgbmmJHZghyxO4yDuECfNWDYNyY7G+T6aGXLpysywgZxIdPxTbyYJ8DbyE9Ir5foQIBpXby+ULVTrOQNbuUlt4iYY0QcAzlK2HRm/ek46r8Sip+3axzebvXy43QJ/XqMF2FTph0qQyIQeqXrjGixjgYQ+gRiVRuS06TWBIMjToG4H5G5UebBNoAir7B0AQzDNgHJt8Jrr2k5AHkr7/nIoiYOwkav7Yo5+FCVWBhr8NT7++qgtqK8CFpHRD5wkWEYAUCFQysYf1F8SRYkeRPbIpYBjhQzGbqbJ6KlF1eETp8oAeXC672L5kiC4PMMmqo/wOINpB//pHNPEsVaMOKuYiEN3fGD6e38zAXeddchn2J9s6QSnjcl33ZHDO9vyoKKHfVYmW/skE2TljaxiS+1zuCjhCMT60QYqBRSUFsIh6aHXxSj2IEgmc64kqErgyOJKS80nDGz0HVVdCVHJXsQadZrrJB1+itIW4H7xlquVHW0/tnTibnRyzK5P6u15Z3JAk4ls86hUEC6lbGK7lJ+Haalcot9QuKRZ7iPMsYlODLOI93A1Tz1E4ahy7uInECaa8fSCLY0ccv1Wx0VM8E77yZbcDn55rH9zeYz7cg6S8a6aD3Pvx+8khN8fKCX5CJj4PBPJKbH71QIhfgjUATJROL144wr3KkeYnzt1ScqGAqfzDu/5bV1B1tkF6rm5SvsOBcdYZW7Tq4oPxYyExbiBMkXzRw0UbCDrV1cCblw43wLEpZtpIkR0P3pf/iD6IvU+hdplSfp62Qvj4HeyuVfZZMgM59O7sPqqHvIxPoJb9T2TSfE/B5/EYr9rDB8qCCWaJxfwmzv6n/xF3RfHqJbWDZY0iPMHczaminOFEjrcrTa2cpCUAc1qGxj+PnAbTppjwmsMkKFCIaL9GwY2W+I4Io3dp3YMoGqRoHAlWLPVL/jh3fvcm6SluMAeuXltXorczpglslG1YAudgyfhIcZF/LIevQgiAKdFln+yVApmObVJ3gSEj2u1T0f7Jy2/PVTGbZrt9RaLyd4u2gm6dTWJO6jADJKGe43Vk1ec5dpOsCfl8mwtpeHZ8DMoSf0L63iNqvETCZe6DQzIPjX57NKBYg2wDLzVObz+fJF3IJWOxvgF6q7J1q2Gnpwm7IXibAzUS3EohgFQy6x6gersbv72kvZAhRDiexovVP6euh3oAgJpMMN4vCrJvNbFOB5cEC2ZTWaYs+qqQZvsh6I36W2UBbbpCgRyNR2Jfm0ffZW76ybjqmyn8Tnmyam+shdSn5bS5z2ew86hImOhv9aqfRL3JQuKJZictnKfNY6195Gz6DD9EyvxVTN+qzzpjLTM3nYuH1zXN9bZz+jKvOc3DygPkGPRAcFRewfQY9v8jACCbojc9QYTKqACJXPvzIwwggAOxZTPwU8sKxM8nq8zpd9d+H3VXQ7hHjTaLlQP4ocKiu0sxRFUuuCWx5mGkTSFt9yOrvAinnZFckMZx2UQkzatZk5c5tKaZdDpkv4WB/wshRBAlJl4SzN+GVY0qdAjIwTLH15IJZxj+p1nUgTBd19SK4WHL2WC1KNIQ2YIqCFUe+baCTPIW9XZtEIQ4wJwpItkbD1i+cs6LPQejapmIcTY1EjMFL7OrwT82FB7ac7gWnv3QIGcUyn2GQoDuBftpxnYzKvKvEz1JBD64os3hjbkGLxpJAJzhft91bCyp/LjeVmCXjmj8X6cMGkJEALjBPuB6htqRXdWNmVbD9qVsOsmWyy3USqPMPTLXzqUNytMuGHaP4YAT0tsE5m5s/ANHnhaQK8rowD8fEuSI8VjQYaKt7YEDd5jT0ljwf3aC2mB+hCxK7W7myTTU6GsJnWy7wFbGHi7DQC+0OQyAVuBw26PmecxOsdMQ0mA7EEemFO46uFT0w8bM86NoebI9KC5FDQh7DiDDiUWYSbZa/E+AKW6C9ADaYlMIg2Fi9tfptqeL0euFQCTo/QDk/Dv2AqGs5xTIk2+I50UfIT7x1SEOXErodN6C+qxpcGMLH5C/7rLo1lgMLGHRNSPKCBmqrrKiOt1eGtWHbE42kcZStPtSvj+ElQ9vIrHEYKITiwXaPuu3JggpaJOqKbDHnDlmosuECzXeVlRDaJyhnQ0FlmtUYOwEJ/X+QRgp84c0MCK/ZwKOq4OWQYzT4/nh4kjJEL0Jqmzx3tDCcKGUruzi+bXVwNQVEZusjlIM+20ul0Ed/NQirkyiMPTiVAjTXNuYKg4hIFvQq+h"

WALL = tenhou_wall_reproducer.reproduce(SEED, 1)[0][0]
for i in range(136):
    if WALL[i] == 4 * 4:  # 0m
        WALL[i] = 4 * 34
    if WALL[i] == 4 * 13:  # 0p
        WALL[i] = 4 * 35
    if WALL[i] == 4 * 22:  # 0s
        WALL[i] = 4 * 36
WALL = np.array(WALL) // 4

HISTORY = np.array([
    [31, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 30, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 28, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 27],
    [Action.TSUMOGIRI, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 8, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 16, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 30],
    [27, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.TSUMOGIRI, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 9, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 14],
    [Action.PASS, Action.NONE, Action.NONE, Action.NONE],
    [0, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.PASS, Action.NONE],
    [Action.NONE, Action.TSUMOGIRI, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 3, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.PASS],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [Action.PASS, Action.NONE, Action.NONE, Action.NONE],
    [1, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.PASS, Action.NONE, Action.NONE],
    [Action.NONE, 33, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.PON, Action.NONE],
    [Action.NONE, Action.NONE, 13, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 8],
    [29, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 17, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 19, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 7],
    [Action.TSUMOGIRI, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.TSUMOGIRI, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, 26],
    [Action.TSUMOGIRI, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 32, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.kakan(33), Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [Action.TSUMOGIRI, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 2, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.PASS],
    [Action.NONE, Action.NONE, Action.NONE, Action.RIICHI],
    [Action.NONE, Action.NONE, Action.NONE, 11],
    [Action.PASS, Action.NONE, Action.NONE, Action.NONE],
    [18, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.PASS, Action.NONE, Action.NONE],
    [Action.NONE, Action.TSUMOGIRI, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.PASS, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [18, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.PASS, Action.NONE, Action.NONE],
    [Action.NONE, 19, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [16, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 1, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [Action.PASS, Action.NONE, Action.NONE, Action.NONE],
    [19, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, Action.PASS, Action.NONE, Action.NONE],
    [Action.NONE, 18, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, 7, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [26, Action.NONE, Action.NONE, Action.NONE],
    [Action.NONE, 28, Action.NONE, Action.NONE],
    [Action.NONE, Action.NONE, Action.TSUMOGIRI, Action.NONE],
    [Action.NONE, Action.NONE, Action.NONE, Action.TSUMOGIRI],
    [Action.NONE, Action.NONE, Action.RON, Action.NONE],
        ])


def test_state():
    state = State.init_with_deck_arr(WALL)

    reward = np.zeros(4, dtype=np.int32)
    done = False
    for i in range(len(HISTORY)):
        state, reward, done = step(state, HISTORY[i])

    # 中,赤2,ドラ1 = 4翻40符で8000点.
    # リー棒も合わせて9000点の収支.
    assert np.all(reward == np.array([0, 0, 9000, -9000]))
    assert done

#def test_jax():
#    wall = tenhou_wall_reproducer.reproduce(SEED, 1)[0][0]
#    state = full_mahjong.State.init_with_deck_arr(jnp.array(wall) // 4)
#
#    reward = np.zeros(4, dtype=jnp.int32)
#    done = False
#    for i in range(len(HISTORY)):
#        state, reward, done = full_mahjong.step(state, HISTORY[i])
#
#    # 中,赤2,ドラ1 = 4翻40符で8000点だが, ドラ未実装のため1300点.
#    # リー棒も合わせて2300点の収支.
#    assert jnp.all(reward == np.array([0, 0, 2300, -2300]))
#    assert done
#
#def test_jax_compatibility():
#    wall = tenhou_wall_reproducer.reproduce(SEED, 1)[0][0]
#
#    state = State.init_with_deck_arr(np.array(wall) // 4)
#
#    for i in range(len(HISTORY)):
#
#        jnp_state = full_mahjong.State(
#            deck=full_mahjong.Deck(
#                arr=jnp.array(state.deck.arr.copy()),
#                idx=state.deck.idx,
#                end=state.deck.end,
#                n_dora=state.deck.n_dora),
#            hand=jnp.array(state.hand.copy()),
#            turn=state.turn,
#            target=state.target,
#            last_draw=state.last_draw,
#            riichi_declared=state.riichi_declared,
#            riichi=jnp.array(state.riichi.copy()),
#            n_meld=jnp.array(state.n_meld.copy()),
#            melds=jnp.array(state.melds.copy()),
#            is_menzen=jnp.array(state.is_menzen.copy()),
#            pon=jnp.array(state.pon.copy()),
#            )
#
#        state, reward, done = step(state, HISTORY[i])
#        jnp_state, jnp_reward, jnp_done = full_mahjong.step(
#                jnp_state, jnp.array(HISTORY[i])
#                )
#
#        assert done == jnp_done
#        assert np.all(reward == jnp_reward)
#        assert np.all(state.deck.arr == jnp_state.deck.arr)
#        assert state.deck.idx == jnp_state.deck.idx
#        assert state.deck.end == jnp_state.deck.end
#        assert state.deck.n_dora == jnp_state.deck.n_dora
#        assert np.all(state.hand == jnp_state.hand)
#        assert state.turn == jnp_state.turn
#        assert state.target == jnp_state.target
#        assert state.last_draw == jnp_state.last_draw
#        assert state.riichi_declared == jnp_state.riichi_declared
#        assert np.all(state.riichi == jnp_state.riichi)
#        assert np.all(state.n_meld == jnp_state.n_meld)
#        assert np.all(state.melds == jnp_state.melds)
#        assert np.all(state.is_menzen == jnp_state.is_menzen)
#        assert np.all(state.pon == jnp_state.pon)
