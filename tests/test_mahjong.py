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
            n_meld=jnp.int8(0),
            last=jnp.int8(0),
            riichi=jnp.bool_(False),
            is_ron=jnp.bool_(False),
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
            n_meld=jnp.int8(0),
            last=jnp.int8(33),
            riichi=jnp.bool_(False),
            is_ron=jnp.bool_(False),
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
            n_meld=jnp.int8(0),
            last=jnp.int8(27),
            riichi=jnp.bool_(False),
            is_ron=jnp.bool_(False),
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
    assert state.current_player == jnp.int8(0)
    assert state.target == jnp.int8(-1)
    assert state.deck[state.next_deck_ix] == jnp.int8(8)
    assert state.hand[0, 8] == jnp.int8(1)

    state = step(state, 8)
    assert state.hand[0, 8] == jnp.int8(0)
    assert state.current_player == jnp.int8(1)
    assert state.target == jnp.int8(-1)
    assert state.deck[state.next_deck_ix] == jnp.int8(31)

    assert state.hand[1, 8] == jnp.int8(2)

    state = step(state, Action.TSUMOGIRI)
    assert state.hand[1, 8] == jnp.int8(1)
    assert state.current_player == jnp.int8(2)
    assert state.target == jnp.int8(-1)


def test_chi():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    """
    current_player 0
    [[0 0 0 0 1 0 1 0 1 1 1 0 1 0 0 0 0 2 1 1 0 0 0 0 0 1 0 1 1 0 1 0 0 0]
     [1 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 3 1 0 0 0]
     [0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 2 1 0 2 0 0 0 1 1 1 0 0 0 0 2 0 0]
     [1 0 2 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 2 0 0]]
    """
    assert state.legal_action_mask[6]
    state = step(state, 6)
    assert state.current_player == 1
    assert state.target == 6
    assert state.legal_action_mask[Action.CHI_R]

    state1 = step(state, Action.CHI_R)
    assert state1.current_player == 1
    assert state1.melds[1, 0] == 25420

    state2 = step(state, Action.PASS)
    assert state2.current_player == 1
    assert state2.melds[1, 0] == 0


def test_ankan():
    key = jax.random.PRNGKey(352)
    state = init(key=key)
    assert state.current_player == 0
    """
    [[1 2 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 2 0 0 0 0 1 0 0 1 0 4 0 0]
     [0 0 1 2 0 1 0 0 0 2 0 0 0 1 1 2 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0]
     [0 0 0 0 1 1 0 2 0 0 0 0 1 0 0 1 0 0 0 3 0 0 0 0 1 0 0 0 0 1 0 1 1 0]
     [0 0 1 0 1 0 0 0 1 0 0 1 1 2 0 0 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 0 0 0]]
    """
    state = step(state, 65)
    assert state.melds[0, 0] == 4033


def test_riichi():
    from pgx._mahjong._mahjong2 import State

    rng = jax.random.PRNGKey(0)
    state = State.from_json("tests/assets/mahjong/riichi_test.json")
    visualize(state, "tests/assets/mahjong/before_riichi.svg")

    assert state.current_player == 0
    state = step(state, 9)

    assert state.legal_action_mask[Action.RIICHI]
    state = step(state, Action.RIICHI)
    assert not state.terminated

    N = 10
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state)
        state = step(state, a)
    visualize(state, f"tests/assets/mahjong/after_riichi_{N}.svg")


def test_ron():
    from pgx._mahjong._mahjong2 import State

    state = State.from_json("tests/assets/mahjong/ron_test.json")
    visualize(state, "tests/assets/mahjong/before_ron.svg")

    assert state.current_player == 0
    state = step(state, 30)  # 北

    assert state.legal_action_mask[Action.RON]

    state = step(state, Action.RON)

    assert state.terminated
    assert (
        state.rewards
        == jnp.array([-500.0, 500.0, 0.0, 0.0], dtype=jnp.float32)
    ).all()
    visualize(state, "tests/assets/mahjong/after_ron.svg")


def test_tsumo():
    from pgx._mahjong._mahjong2 import State

    state = State.from_json("tests/assets/mahjong/tsumo_test.json")
    visualize(state, "tests/assets/mahjong/before_tsumo.svg")
    assert state.current_player == 0
    state = step(state, 30)

    assert state.legal_action_mask[Action.TSUMO]

    state = step(state, Action.TSUMO)

    assert state.terminated
    assert (
        state.rewards
        == jnp.array([-500.0, 1100.0, -300.0, -300.0], dtype=jnp.float32)
    ).all()
    visualize(state, "tests/assets/mahjong/after_tsumo.svg")


def test_transparent():
    rng = jax.random.PRNGKey(31)
    state = init(key=rng)
    for _ in range(65):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state)
        state = step(state, a)

    visualize(state, "tests/assets/mahjong/transparent.svg")


def test_json():
    from pgx._mahjong._mahjong2 import State
    import os

    rng = jax.random.PRNGKey(0)
    state = init(key=rng)
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state)
        state = step(state, a)

    path = "temp.json"
    with open(path, mode="w") as f:
        f.write(state.json)

    state2 = State.from_json(path=path)
    assert state == state2
    os.remove(path)


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
