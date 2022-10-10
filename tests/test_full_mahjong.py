import numpy as np

from pgx.mahjong._full_mahjong import Deck, Hand, Action, Yaku, Meld, Tile, init, step


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
    hand = Hand.add(hand, 0)
    assert Hand.can_ron(hand, 0)
    assert not Hand.can_ron(hand, 1)
    hand = Hand.add(hand, 0)
    assert Hand.can_tsumo(hand)
    hand = Hand.add(hand, 1, 2)
    assert Hand.can_pon(hand, 0)

    assert Hand.can_chi(hand, 2, Action.CHI_R)
    assert Hand.can_chi(hand, 1, Action.CHI_M) == False
    hand = Hand.add(hand, 2, 3)  # 1122333m
    assert Hand.can_chi(hand, 1, Action.CHI_M)

    hand = Hand.chi(hand, 1, Action.CHI_M)
    # 12233m
    hand = Hand.sub(hand, 1)
    # 1233m
    hand = Hand.pon(hand, 2)
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
        2,0,0,0,0,0,0,1,0,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 8, Action.CHI_M)

    hand = np.array([
        2,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 7, Action.CHI_L)

    assert np.all(
            np.array([
                    2,0,0,0,0,0,0,0,1,
                    1,0,0,0,1,2,1,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,3,0
                ]) ==
            Hand.from_str("119m15667p666z")
            )

    hand = Hand.from_str("117788m2233s6677z")
    assert Hand.can_tsumo(hand)

    hand = Hand.from_str("19m19p199s1234567z")
    assert Hand.can_tsumo(hand)

def fu(
        hand_s: str,
        melds_s: str,
        last_s: str,
        is_ron: bool = False
        ) -> bool:
    hand = Hand.from_str(hand_s)
    melds = np.zeros(4, dtype=np.int32)
    meld_num=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[meld_num] = Meld.from_str(s)
        melds
        meld_num += 1
    last = Tile.from_str(last_s)
    return Yaku.judge(hand, melds, meld_num, last, False, is_ron)[2]

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
    hand = Hand.from_str(hand_s)
    melds = np.zeros(4, dtype=np.int32)
    meld_num=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[meld_num] = Meld.from_str(s)
        meld_num += 1
    last = Tile.from_str(last_s)
    return Yaku.judge(hand, melds, meld_num, last, False, is_ron)[0][yaku]

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


def score(
        hand_s: str,
        melds_s: str,
        last_s: str,
        riichi: bool = False,
        is_ron: bool = False
        ) -> int:
    hand = Hand.from_str(hand_s)
    melds = np.zeros(4, dtype=np.int32)
    meld_num=0
    for s in melds_s.split(","):
        if not s:
            continue
        melds[meld_num] = Meld.from_str(s)
        meld_num += 1
    last = Tile.from_str(last_s)
    return Yaku.score(hand, melds, meld_num, last, riichi, is_ron)

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


def test_state():
    state = init()

    #MANZU_6 = 5
    #state, _, _ = step(
    #    state, np.array([MANZU_6, Action.NONE, Action.NONE, Action.NONE])
    #)

    #state, _, _ = step(
    #    state, np.array([Action.NONE, Action.CHI_R, Action.NONE, Action.NONE])
    #)

    #SOUZU_7 = 24
    #state, _, _ = step(
    #    state, np.array([Action.NONE, SOUZU_7, Action.NONE, Action.NONE])
    #)

    #state, _, _ = step(
    #    state, np.array([Action.NONE, Action.NONE, Action.NONE, Action.PON])
    #)

    #DRAGON_R = 33
    #state, _, _ = step(
    #    state, np.array([Action.NONE, Action.NONE, Action.NONE, DRAGON_R])
    #)

    #MANZU_1 = 0
    #state, _, _ = step(
    #    state, np.array([MANZU_1, Action.NONE, Action.NONE, Action.NONE])
    #)

    #state, _, _ = step(
    #    state, np.array([Action.NONE, Action.PASS, Action.NONE, Action.NONE])
    #)
