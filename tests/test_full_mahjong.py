import jax
import jax.numpy as jnp

from pgx.mahjong.full_mahjong import Deck, Hand, Action, Yaku, Meld, init, step


def test_deck():
    deck = Deck.init(jax.random.PRNGKey(seed=0))
    assert deck.idx == 0
    deck, tile1 = Deck.draw(deck)
    assert deck.idx == 1
    deck = Deck.init(jax.random.PRNGKey(seed=1))
    assert deck.idx == 0
    deck, tile2 = Deck.draw(deck)
    assert tile1 != tile2


def test_hand():
    hand = jnp.zeros(34, dtype=jnp.uint8)
    hand = Hand.add(hand, 0)
    assert Hand.can_ron(hand, 0)
    assert not Hand.can_ron(hand, 1)
    hand = Hand.add(hand, 0)
    assert Hand.can_tsumo(hand)
    assert Hand.can_pon(hand, 0)

    assert not Hand.can_chi(hand, 2, Action.CHI_R)
    hand = Hand.add(hand, 1)
    assert Hand.can_chi(hand, 2, Action.CHI_R)
    assert not Hand.can_chi(hand, 1, Action.CHI_M)
    hand = Hand.add(hand, 2)
    assert Hand.can_chi(hand, 1, Action.CHI_M)
    assert Hand.can_chi(hand, 0, Action.CHI_L)
    assert not Hand.can_chi(hand, 1, Action.CHI_L)

    hand = Hand.chi(hand, 0, Action.CHI_L)
    assert hand[0] == 2
    assert hand[1] == 0
    assert hand[2] == 0
    hand = Hand.pon(hand, 0)
    assert hand[0] == 0

    hand = jnp.array([
        3,1,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.can_tsumo(hand)

    hand = jnp.array([
        3,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.is_tenpai(hand)

    hand = jnp.array([
        3,0,1,1,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.is_tenpai(hand)

    hand = jnp.array([
        3,0,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0
        ])
    assert Hand.can_riichi(hand)

    hand = jnp.array([
        3,0,1,1,0,1,1,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0
        ])
    assert not Hand.can_riichi(hand)

    hand = jnp.array([
        2,0,0,0,0,0,0,1,0,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 8, Action.CHI_M)

    hand = jnp.array([
        2,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 7, Action.CHI_L)

    assert jnp.all(
            jnp.array([
                    2,0,0,0,0,0,0,0,1,
                    1,0,0,0,1,2,1,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,3,0
                ]) ==
            Hand.from_str("119m15667p666z")
            )

def fu(
    hand: jnp.ndarray,
    melds: jnp.ndarray,
    meld_num: int,
    last: int,
    is_ron: bool = False,
) -> bool:
    return Yaku.judge(hand, melds, meld_num, last, is_ron)[1]

def test_yaku_fu():
    hand = Hand.from_str("112233567m34588p")
    melds = jnp.zeros(4, dtype=jnp.int32)
    # リャンメン(平和)
    assert fu(hand, melds, meld_num=0, last=0) == 20
    assert fu(hand, melds, meld_num=0, last=0, is_ron=True) == 30

    # 単騎
    assert fu(hand, melds, meld_num=0, last=16) == 30
    assert fu(hand, melds, meld_num=0, last=16, is_ron=True) == 40

    # ペンチャン
    assert fu(hand, melds, meld_num=0, last=2) == 30
    assert fu(hand, melds, meld_num=0, last=2, is_ron=True) == 40

    # カンチャン
    assert fu(hand, melds, meld_num=0, last=1) == 30
    assert fu(hand, melds, meld_num=0, last=1, is_ron=True) == 40

    hand = Hand.from_str("111222333567m88p")
    assert fu(hand, melds, meld_num=0, last=0) == 40
    # 平和一盃口 == 三暗刻. 符で三暗刻が優先される.

    hand = Hand.from_str("233445m23456799p")
    assert fu(hand, melds, meld_num=0, last=3) == 20
    # 平和がつくのでリャンメンにとる
    hand = Hand.from_str("233445m22234599p")
    assert fu(hand, melds, meld_num=0, last=3) == 30
    # 平和がつかないのでカンチャンにとる

    hand = Hand.from_str("11123456m111234p")
    assert fu(hand, melds, meld_num=0, last=0) == 40
    # 単騎 > リャンメン

    hand = Hand.from_str("111123456m11222p")
    assert fu(hand, melds, meld_num=0, last=0, is_ron=True) == 50
    # リャンメン > シャンポン


def has_yaku(
    yaku: int,
    hand: jnp.ndarray,
    melds: jnp.ndarray,
    meld_num: int,
    last: int,
    is_ron: bool = False,
) -> bool:
    return Yaku.judge(hand, melds, meld_num, last, is_ron)[0][yaku]

def test_yaku_tanyao():
    hand = jnp.zeros(34, dtype=jnp.uint8)
    hand = Hand.add(hand, 1, 2)
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.断么九, hand, melds, meld_num=0, last=1)

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 3, 0))
    assert has_yaku(Yaku.断么九, hand, melds, meld_num=1, last=1)

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 2, 0))
    assert not has_yaku(Yaku.断么九, hand, melds, meld_num=1, last=1)

    melds = melds.at[0].set(Meld.init(Action.PON, 6, 0))
    assert has_yaku(Yaku.断么九, hand, melds, meld_num=1, last=1)

    melds = melds.at[0].set(Meld.init(Action.PON, 33, 0))
    assert not has_yaku(Yaku.断么九, hand, melds, meld_num=1, last=1)

def test_yaku_flush():
    hand = Hand.from_str("11223355577789m")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert not has_yaku(Yaku.混一色, hand, melds, meld_num=0, last=0)
    assert has_yaku(Yaku.清一色, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11223355577m789p")
    assert not has_yaku(Yaku.混一色, hand, melds, meld_num=0, last=0)
    assert not has_yaku(Yaku.清一色, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11223355577m777z")
    assert has_yaku(Yaku.混一色, hand, melds, meld_num=0, last=0)
    assert not has_yaku(Yaku.清一色, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11223355577m")
    melds = melds.at[0].set(Meld.init(Action.CHI_L, 6, 0))  # [7]89m
    assert not has_yaku(Yaku.混一色, hand, melds, meld_num=1, last=0)
    assert has_yaku(Yaku.清一色, hand, melds, meld_num=1, last=0)

    melds = melds.at[0].set(Meld.init(Action.CHI_L, 15, 0))  # [7]89p
    assert not has_yaku(Yaku.混一色, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.清一色, hand, melds, meld_num=1, last=0)

    melds = melds.at[0].set(Meld.init(Action.PON, 33, 0))  # 777z
    assert has_yaku(Yaku.混一色, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.清一色, hand, melds, meld_num=1, last=0)


def test_yaku_pinfu():
    hand = Hand.from_str("12345666m789p123s")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=0)
    assert not has_yaku(Yaku.平和, hand, melds, meld_num=0, last=2)
    assert not has_yaku(Yaku.平和, hand, melds, meld_num=0, last=15)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=26)

    hand = Hand.from_str("11223344556677m")
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=0)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=1)
    assert not has_yaku(Yaku.平和, hand, melds, meld_num=0, last=2)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=3)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=4)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=5)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=6)

    hand = Hand.from_str("22334455667788m")
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=2)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=3)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=4)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=5)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=6)
    assert has_yaku(Yaku.平和, hand, melds, meld_num=0, last=7)


def test_yaku_outside():
    hand = Hand.from_str("11223399m789p123s")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.純全帯么九, hand, melds, meld_num=0, last=0)
    assert not has_yaku(Yaku.混全帯么九, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11123m789p123s333z")
    assert not has_yaku(Yaku.純全帯么九, hand, melds, meld_num=0, last=0)
    assert has_yaku(Yaku.混全帯么九, hand, melds, meld_num=0, last=0)

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 2, src=0))
    hand = Hand.from_str("11223399m789p")
    assert has_yaku(Yaku.純全帯么九, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.混全帯么九, hand, melds, meld_num=1, last=0)

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 4, src=0))
    hand = Hand.from_str("11223399m789p")
    assert not has_yaku(Yaku.純全帯么九, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.混全帯么九, hand, melds, meld_num=1, last=0)

    melds = melds.at[0].set(Meld.init(Action.PON, 33, src=0))
    hand = Hand.from_str("11223399m789p")
    assert not has_yaku(Yaku.純全帯么九, hand, melds, meld_num=1, last=0)
    assert has_yaku(Yaku.混全帯么九, hand, melds, meld_num=1, last=0)


def test_yaku_double_chows():
    hand = Hand.from_str("11223388m789p123s")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.一盃口, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11223388m778899p")
    assert not has_yaku(Yaku.一盃口, hand, melds, meld_num=0, last=0)
    assert has_yaku(Yaku.二盃口, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("11122223333444m")
    assert not has_yaku(Yaku.二盃口, hand, melds, meld_num=0, last=0)


def test_is_pure_straight():
    hand = Hand.from_str("12345678999m345p")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.一気通貫, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("123456999m33345p")
    assert not has_yaku(Yaku.一気通貫, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("99m123456p345s")
    melds = melds.at[0].set(Meld.init(Action.CHI_M, 16, src=0))  # 7[8]9p
    assert has_yaku(Yaku.一気通貫, hand, melds, meld_num=1, last=0)

    melds = melds.at[0].set(Meld.init(Action.CHI_M, 75, src=0))  # 7[8]9s
    assert not has_yaku(Yaku.一気通貫, hand, melds, meld_num=1, last=0)


def test_yaku_triple_chow():
    hand = Hand.from_str("112233m123p12388s")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.三色同順, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("112233m123p23488s")
    assert not has_yaku(Yaku.三色同順, hand, melds, meld_num=0, last=0)

    melds = melds.at[0].set(Meld.init(Action.CHI_L, 9, src=0))  # [1]23p
    hand = Hand.from_str("112233m12388s")
    assert has_yaku(Yaku.三色同順, hand, melds, meld_num=1, last=0)

    hand = Hand.from_str("112233m123p23488s")
    assert not has_yaku(Yaku.三色同順, hand, melds, meld_num=1, last=0)


def test_all_pungs():
    hand = Hand.from_str("11133m111p")
    melds = jnp.zeros(4, dtype=jnp.int32)
    melds = melds.at[0].set(Meld.init(Action.PON, 10, src=0))  # 222s
    melds = melds.at[1].set(Meld.init(Action.PON, 33, src=0))  # 777z
    assert has_yaku(Yaku.対々和, hand, melds, meld_num=2, last=0)

    melds = melds.at[1].set(Meld.init(Action.CHI_L, 10, src=0))  # [2]34s
    assert not has_yaku(Yaku.対々和, hand, melds, meld_num=2, last=0)

    hand = Hand.from_str("11133m111789p")
    assert not has_yaku(Yaku.対々和, hand, melds, meld_num=1, last=0)

    hand = Hand.from_str("11133m111777p")
    assert has_yaku(Yaku.対々和, hand, melds, meld_num=1, last=0)
    

def test_yaku_triple_pung():
    hand = Hand.from_str("111333m111p11188s")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.三色同刻, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("111333m222p11188s")
    assert not has_yaku(Yaku.三色同刻, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("222333m22288s")
    melds = melds.at[0].set(Meld.init(Action.PON, 10, src=0))  # 222p
    assert has_yaku(Yaku.三色同刻, hand, melds, meld_num=1, last=1)

    melds = melds.at[0].set(Meld.init(Action.PON, 11, src=0))  # 333p
    assert not has_yaku(Yaku.三色同刻, hand, melds, meld_num=1, last=1)


def test_yaku_three_concealed_pungs():
    hand = Hand.from_str("11122233344455m")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=0, is_ron=True)
    assert not has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=4, is_ron=True)
    assert not has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=0)

    hand = Hand.from_str("111333m66677z")
    melds = melds.at[0].set(Meld.init(Action.PON, 11, src=0))  # 333p
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.三暗刻, hand, melds, meld_num=1, last=0, is_ron=True)
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=1, last=33)
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=1, last=33, is_ron=True)

    hand = Hand.from_str("111123m11166677z")
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=0, is_ron=True)

    hand = Hand.from_str("111234m11166677z")
    assert not has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=0, is_ron=True)


def test_yaku_coner_cases():
    hand = Hand.from_str("111222333789m22z")
    melds = jnp.zeros(4, dtype=jnp.int32)
    assert not has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=1)
    assert has_yaku(Yaku.一盃口, hand, melds, meld_num=0, last=1)
    assert has_yaku(Yaku.混全帯么九, hand, melds, meld_num=0, last=1)
    # 面前チャンタ一盃口 > 三暗刻

    # TODO
    hand = Hand.from_str("111222333m11p")
    melds = melds.at[0].set(Meld.init(Action.CHI_M, 7, src=0))  # 7[8]9m
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=1, last=0)
    assert not has_yaku(Yaku.純全帯么九, hand, melds, meld_num=1, last=0)
    # 副露純チャン < 三暗刻 (符で有利)

    # TODO
    hand = Hand.from_str("11222333444456m")
    assert not has_yaku(Yaku.平和, hand, melds, meld_num=0, last=0)
    assert not has_yaku(Yaku.一盃口, hand, melds, meld_num=0, last=0)
    assert has_yaku(Yaku.三暗刻, hand, melds, meld_num=0, last=0)
    # 平和一盃口 < 三暗刻 (符で有利)


def test_state():
    state = init(jax.random.PRNGKey(seed=0))

    MANZU_6 = 5
    state, _, _ = step(
        state, jnp.array([MANZU_6, Action.NONE, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.CHI_R, Action.NONE, Action.NONE])
    )

    SOUZU_7 = 24
    state, _, _ = step(
        state, jnp.array([Action.NONE, SOUZU_7, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.NONE, Action.NONE, Action.PON])
    )

    DRAGON_R = 33
    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.NONE, Action.NONE, DRAGON_R])
    )

    MANZU_1 = 0
    state, _, _ = step(
        state, jnp.array([MANZU_1, Action.NONE, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.PASS, Action.NONE, Action.NONE])
    )
