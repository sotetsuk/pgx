from pgx._mahjong._hand import Hand
from pgx._mahjong._yaku import Yaku
from pgx._mahjong._shanten import Shanten
import jax.numpy as jnp


def test_ron():
    # fmt:off
    hand = jnp.int8([
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert Hand.can_ron(hand, 0)
    assert ~Hand.can_ron(hand, 1)

    # 国士無双
    # fmt:off
    hand = jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1
    ])
    # fmt:on

    assert Hand.can_ron(hand, 33)
    assert ~Hand.can_ron(hand, 1)

    # 七対子
    # fmt:off
    hand = jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert Hand.can_ron(hand, 0)
    assert ~Hand.can_ron(hand, 1)


def test_score():
    # 平和ツモ
    # fmt:off
    hand = jnp.int32([
        1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on
    assert (
        Yaku.score(
            hand=hand,
            melds=jnp.zeros(4, dtype=jnp.int32),
            n_meld=0,
            last=0,
            riichi=False,
            is_ron=False,
        )
        == 320
    )
    # 国士無双
    # fmt:off
    hand = jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        2, 1, 1, 1, 1, 1, 1
    ])
    # fmt:on

    assert (
        Yaku.score(
            hand=hand,
            melds=jnp.zeros(4, dtype=jnp.int32),
            n_meld=0,
            last=33,
            riichi=False,
            is_ron=False,
        )
        == 8000
    )

    # 七対子
    # fmt:off
    hand = jnp.int8([
        2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert (
        Yaku.score(
            hand=hand,
            melds=jnp.zeros(4, dtype=jnp.int32),
            n_meld=0,
            last=27,
            riichi=False,
            is_ron=False,
        )
        == 800
    )


def test_shanten():
    # fmt:off
    hand = jnp.int32([
        2, 0, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 0
    ])
    # fmt:on

    assert Shanten.number(hand) == 5

    # fmt:off
    hand = jnp.int32([
        2, 0, 0, 2, 0, 0, 0, 0, 2,
        2, 0, 0, 2, 0, 0, 0, 0, 2,
        1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on
    assert Shanten.number(hand) == 1
