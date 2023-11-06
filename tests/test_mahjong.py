from pgx.mahjong import Mahjong
from pgx.mahjong._env import State
from pgx.mahjong._hand import Hand
from pgx.mahjong._yaku import Yaku
from pgx.mahjong._shanten import Shanten
from pgx.mahjong._action import Action
import jax.numpy as jnp
from jax import jit
import jax
from pgx.experimental.utils import act_randomly

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

env = Mahjong()
init = jit(env.init)
step = jit(env.step)
act_randomly = jit(act_randomly)


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

    from pgx.mahjong._action import Action

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
    # 平和ツモドラ1
    # 参考:
    # tobakushi.net/mahjang/cgi-bin/keisan.cgi?hai=02,03,04,05,06,11,12,13,14,15,16,21,21&naki=,,,&agari=01&dora=06,,,,,,,,,&tsumoron=0&honba=0&jifu=32&bafu=31&reach=0

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
            dora=jnp.zeros(34, dtype=jnp.bool_).at[5].set(TRUE),
        )
        == 640
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
            dora=jnp.zeros(34, dtype=jnp.bool_).at[5].set(TRUE),
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
            dora=jnp.zeros(34, dtype=jnp.bool_).at[5].set(TRUE),
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
    _, key = jax.random.split(key)  # due to API update
    state = init(key=key)
    assert state.current_player == jnp.int8(0)
    assert state._target == jnp.int8(-1)
    assert state._deck[state._next_deck_ix] == jnp.int8(8)
    assert state._hand[0, 8] == jnp.int8(1)

    state: State = step(state, 8)
    assert state._hand[0, 8] == jnp.int8(0)
    assert state.current_player == jnp.int8(1)
    assert state._target == jnp.int8(-1)
    assert state._deck[state._next_deck_ix] == jnp.int8(31)

    assert state._hand[1, 8] == jnp.int8(2)

    state: State = step(state, Action.TSUMOGIRI)
    assert state._hand[1, 8] == jnp.int8(1)
    assert state.current_player == jnp.int8(2)
    assert state._target == jnp.int8(-1)


def test_chi():
    key = jax.random.PRNGKey(0)
    state: State = init(key=key)
    """
    current_player 0
    [[0 0 0 0 1 0 1 0 1 1 1 0 1 0 0 0 0 2 1 1 0 0 0 0 0 1 0 1 1 0 1 0 0 0]
     [1 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 3 1 0 0 0]
     [0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 2 1 0 2 0 0 0 1 1 1 0 0 0 0 2 0 0]
     [1 0 2 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 2 0 0]]
    """
    assert state.legal_action_mask[6]
    state: State = step(state, 6)
    assert state.current_player == jnp.int8(1)
    assert state._target == jnp.int8(6)
    assert state.legal_action_mask[Action.CHI_R]

    state1 = step(state, Action.CHI_R)
    assert state1.current_player == jnp.int8(1)
    assert state1._melds[1, 0] == jnp.int32(25420)

    state2 = step(state, Action.PASS)
    assert state2.current_player == jnp.int8(1)
    assert state2._melds[1, 0] == jnp.int8(0)


def test_ankan():
    key = jax.random.PRNGKey(352)
    _, key = jax.random.split(key)
    state: State = init(key=key)
    assert state.current_player == jnp.int8(0)
    """
    [[1 2 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 2 0 0 0 0 1 0 0 1 0 4 0 0]
     [0 0 1 2 0 1 0 0 0 2 0 0 0 1 1 2 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0]
     [0 0 0 0 1 1 0 2 0 0 0 0 1 0 0 1 0 0 0 3 0 0 0 0 1 0 0 0 0 1 0 1 1 0]
     [0 0 1 0 1 0 0 0 1 0 0 1 1 2 0 0 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 0 0 0]]
    """
    assert state.legal_action_mask[65]
    assert (state._doras == jnp.int32([28, -1, -1, -1, -1])).all()
    assert state._n_kan == jnp.int8(0)

    state: State = step(state, 65)
    assert state._melds[0, 0] == jnp.int32(4033)
    assert (state._doras == jnp.int32([28, 23, -1, -1, -1])).all()
    assert state._n_kan == jnp.int8(1)


def test_riichi():
    rng = jax.random.PRNGKey(0)
    state = State.from_json("tests/assets/mahjong/riichi_test.json")
    visualize(state, "tests/assets/mahjong/before_riichi.svg")

    assert state.current_player == jnp.int8(0)
    state: State = step(state, 9)

    assert state.legal_action_mask[Action.RIICHI]
    state: State = step(state, Action.RIICHI)
    assert not state.terminated

    N = 10
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state.legal_action_mask)
        state: State = step(state, a)
    visualize(state, f"tests/assets/mahjong/after_riichi_{N}.svg")


def test_ron():
    state = State.from_json("tests/assets/mahjong/ron_test.json")
    visualize(state, "tests/assets/mahjong/before_ron.svg")

    assert state.current_player == jnp.int8(0)
    state: State = step(state, 30)  # 北

    assert state.legal_action_mask[Action.RON]

    state: State = step(state, Action.RON)

    assert state.terminated
    assert (
        state.rewards
        == jnp.array([-500.0, 500.0, 0.0, 0.0], dtype=jnp.float32)
    ).all()
    visualize(state, "tests/assets/mahjong/after_ron.svg")


def test_tsumo():
    state = State.from_json("tests/assets/mahjong/tsumo_test.json")
    visualize(state, "tests/assets/mahjong/before_tsumo.svg")
    assert state.current_player == jnp.int8(0)
    state: State = step(state, 30)

    assert state.legal_action_mask[Action.TSUMO]

    state: State = step(state, Action.TSUMO)

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
        a = act_randomly(subkey, state.legal_action_mask)
        state: State = step(state, a)

    visualize(state, "tests/assets/mahjong/transparent.svg")


def test_json():
    import os

    rng = jax.random.PRNGKey(0)
    state = init(key=rng)
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, state.legal_action_mask)
        state: State = step(state, a)

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
            a = act_randomly(subkey, state.legal_action_mask)
            state: State = step(state, a)

            assert state._hand[state.current_player].sum() + jnp.count_nonzero(
                state._melds[state.current_player]
            ) * 3 in [13, 14]
            assert (0 <= state._hand).all()
            assert (state._hand <= 4).all()
            assert (0 <= state._melds).all()


# def test_api():
#    import pgx
#    env = pgx.make("mahjong")
#    pgx.api_test(env, 3, use_key=False)
#    pgx.api_test(env, 3, use_key=True)
