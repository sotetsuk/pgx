from pgx._mahjong._hand import Hand
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
