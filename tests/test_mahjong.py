from pgx._mahjong._hand import Hand
from pgx._mahjong._yaku import Yaku
from pgx._mahjong._action import Action
from pgx._mahjong._shanten import Shanten
from pgx._mahjong._mahjong2 import Mahjong
import jax.numpy as jnp
from jax import jit
import jax
from pgx.experimental.utils import act_randomly

env = Mahjong()
init = jit(env.init)
step = jit(env.step)


def visualize(state, fname="tests/assets/mahjong/xxx.svg"):
    state.save_svg(fname, color_theme="dark")


def test_hand():
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
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 0
    assert state.target == -1
    assert state.deck[state.next_deck_ix] == 8
    # fmt:off
    assert (state.hand == jnp.int8(
        [[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
         [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]]
    )).all()
    # fmt:on

    state = step(state, 8)
    assert state.current_player == 1
    assert state.target == -1
    assert state.deck[state.next_deck_ix] == 31
    # fmt:off
    assert (state.hand == jnp.int8(
        [[0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
         [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]]
    )).all()
    # fmt:on

    state = step(state, Action.TSUMOGIRI)
    assert state.current_player == 2
    assert state.target == -1
    assert state.deck[state.next_deck_ix] == 16
    # fmt:off
    assert (state.hand == jnp.int8(
        [[0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 0],
         [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0]]
    )).all()
    # fmt:on


def test_chi():
    key = jax.random.PRNGKey(2)
    state = init(key=key)
    """
    current_player 1
    [[0 1 0 1 0 1 0 0 0 1 1 1 0 1 0 0 0 1 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0]
     [2 0 1 1 1 0 1 0 0 1 0 1 3 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 2 0 1 1 2 0 0 0 0 0 1 0 0 1 0 0 1 1 0 0 0 1 1 1 0 0 0 0 0 0]
     [0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 2 0 2 1 0 0 0 1 0 0 0 0 1 1 0 0 2]]
    """
    state = env.step(state, 6)
    assert state.current_player == 2
    assert state.target == 6
    assert state.legal_action_mask[Action.CHI_L]

    state1 = env.step(state, Action.CHI_L)
    assert state1.current_player == 2
    # fmt:off
    assert (state1.hand == jnp.int8([
        [[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], 
         [2, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
         [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 2]]])).all()
    # fmt:on
    assert state1.melds[2, 0] == 25418

    state2 = env.step(state, Action.PASS)
    assert state2.current_player == 2


def test_ankan():
    key = jax.random.PRNGKey(87)
    state = env.init(key=key)
    assert state.current_player == 0
    """
    [[0 0 0 1 0 1 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 4 0 0 0 0 1 1 1 0 0]
     [1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 3 0 0 2 0 0 1 0 0 1 1 0 0 1 0 0 0 1]
     [0 1 0 0 0 0 1 0 2 0 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 1 0 1 0]
     [0 0 2 0 0 0 0 0 0 0 1 2 1 0 1 1 0 0 2 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0]]
    """
    state = env.step(state, 58)

    # fmt:off
    assert (state.hand == jnp.int8(
        [[1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        ,[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
        ,[0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]
        ,[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]])).all()
    # fmt:on
    assert state.melds[0, 0] == 3130


def test_riichi():
    rng = jax.random.PRNGKey(38)
    state = init(key=rng)
    for _ in range(32):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state)
        state = step(state, a)

    visualize(state, "tests/assets/mahjong/before_riichi.svg")

    assert state.legal_action_mask[Action.RIICHI]
    state = step(state, Action.RIICHI)

    for i in range(12):
        visualize(state, f"tests/assets/mahjong/after_riichi_{i}.svg")

        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state)
        state = step(state, a)


def test_random_play():
    for i in range(10):
        rng = jax.random.PRNGKey(i)
        state = init(key=rng)

        for _ in range(70):
            rng, subkey = jax.random.split(rng)
            a = act_randomly(subkey, state)
            state = step(state, a)

            assert state.hand[state.current_player].sum() + jnp.count_nonzero(
                state.melds[state.current_player]
            ) * 3 in [13, 14]
            assert (0 <= state.hand).all()
            assert (state.hand <= 4).all()
            assert (0 <= state.melds).all()
