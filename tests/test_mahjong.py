from pgx._mahjong._hand import Hand
from pgx._mahjong._yaku import Yaku
from pgx._mahjong._shanten import Shanten
from pgx._mahjong._mahjong import Mahjong
import jax.numpy as jnp
from jax import jit
import jax


def test_ron():
    # fmt:off
    hand = jnp.int8([
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert jit(Hand.can_ron)(hand, 0)
    assert ~jit(Hand.can_ron)(hand, 1)

    # 国士無双
    # fmt:off
    hand = jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1
    ])
    # fmt:on

    assert jit(Hand.can_ron)(hand, 33)
    assert ~jit(Hand.can_ron)(hand, 1)

    # 七対子
    # fmt:off
    hand = jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert jit(Hand.can_ron)(hand, 0)
    assert ~jit(Hand.can_ron)(hand, 1)


def test_riichi():
    # fmt:off
    hand = jnp.int8([
        1, 1, 1, 1, 1, 1, 1, 1, 0,
        3, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert jit(Hand.can_riichi)(hand)

    # fmt:off
    hand = jnp.int8([
        1, 1, 1, 1, 1, 1, 1, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert ~jit(Hand.can_riichi)(hand)

    # fmt:off
    hand = jnp.int8([
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on

    assert jit(Hand.can_riichi)(hand)


def test_can_chi():
    from pgx._mahjong._action import Action

    # fmt:off
    hand = jnp.int8([
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on
    assert jit(Hand.can_chi)(hand, 0, Action.CHI_L)


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
        jit(Yaku.score)(
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
        jit(Yaku.score)(
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
        jit(Yaku.score)(
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

    assert jit(Shanten.number)(hand) == 5

    # fmt:off
    hand = jnp.int32([
        2, 0, 0, 2, 0, 0, 0, 0, 2,
        2, 0, 0, 2, 0, 0, 0, 0, 2,
        1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ])
    # fmt:on
    assert jit(Shanten.number)(hand) == 1


def test_discard():

    env = Mahjong()
    key = jax.random.PRNGKey(0)
    state = env.init(key=key)
    assert state.current_player == 0
    assert state.deck[state.next_deck_ix] == 8
    # fmt:off
    assert (state.hand==jnp.int8([
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]])).all()
    # fmt:on

    state = env.step(state, 4)
    assert state.current_player == 1
    assert state.deck[state.next_deck_ix] == 31
    # fmt:off
    assert (state.hand==jnp.int8([
        [[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
         [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]]])).all()
    # fmt:on

    state = env.step(state, 0)
    assert state.current_player == 2
    assert state.deck[state.next_deck_ix] == 16
    # fmt:off
    assert (state.hand==jnp.int8([
        [[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 0],
         [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]]])).all()
    # fmt:on
