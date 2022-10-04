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


def test_yaku_tanyao():
    hand = jnp.zeros(34, dtype=jnp.uint8)
    hand = Hand.add(hand, 1, 2)
    melds = jnp.zeros(4, dtype=jnp.uint32)
    assert Yaku.judge(hand, melds, meld_num=0, last=1)[Yaku.断么九]

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 3, 0))
    assert Yaku.judge(hand, melds, meld_num=1, last=1)[Yaku.断么九]

    melds = melds.at[0].set(Meld.init(Action.CHI_R, 2, 0))
    assert not Yaku.judge(hand, melds, meld_num=1, last=1)[Yaku.断么九]

    melds = melds.at[0].set(Meld.init(Action.PON, 6, 0))
    assert Yaku.judge(hand, melds, meld_num=1, last=1)[Yaku.断么九]

    melds = melds.at[0].set(Meld.init(Action.PON, 33, 0))
    assert not Yaku.judge(hand, melds, meld_num=1, last=1)[Yaku.断么九]

def test_yaku_pinfu():
    hand = Hand.from_str("12345666m789p123s")
    melds = jnp.zeros(4, dtype=jnp.uint32)
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=2)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=15)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=26)[Yaku.平和]

    hand = Hand.from_str("11122233344456m")
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=1)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=2)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=3)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=4)[Yaku.平和]

    hand = Hand.from_str("11223344556677m")
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=1)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=2)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=3)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=4)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=5)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=6)[Yaku.平和]

    hand = Hand.from_str("22334455667788m")
    assert Yaku.judge(hand, melds, meld_num=0, last=2)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=3)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=4)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=5)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=6)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=7)[Yaku.平和]

    hand = Hand.from_str("11222333444456m")
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=1)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=2)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=3)[Yaku.平和]
    assert not Yaku.judge(hand, melds, meld_num=0, last=4)[Yaku.平和]
    assert Yaku.judge(hand, melds, meld_num=0, last=5)[Yaku.平和]


def test_yaku_double_chows():
    hand = Hand.from_str("11223388m789p123s")
    melds = jnp.zeros(4, dtype=jnp.uint32)
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.一盃口]

    hand = Hand.from_str("11223388m778899p")
    assert not Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.一盃口]
    assert Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.二盃口]

    hand = Hand.from_str("11122223333444m")
    assert not Yaku.judge(hand, melds, meld_num=0, last=0)[Yaku.二盃口]


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
