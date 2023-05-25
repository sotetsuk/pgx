import jax
import jax.numpy as jnp
import pgx
from pgx.gardner_chess import State, Action, GardnerChess, _zobrist_hash
from pgx.experimental.utils import act_randomly

pgx.set_visualization_config(color_theme="dark")


env = GardnerChess()
init = jax.jit(env.init)
step = jax.jit(env.step)


def p(s: str, b=False):
    """
    >>> p("e1")
    20
    >>> p("e1", b=True)
    24
    """
    x = "abcde".index(s[0])
    offset = int(s[1]) - 1 if not b else 5 - int(s[1])
    return x * 5 + offset

def test_zobrist_hash():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    state = init(subkey)
    assert (state._zobrist_hash == jax.jit(_zobrist_hash)(state)).all()
    # for i in range(5):
    while not state.terminated:
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state)
        state = step(state, action)
        print(action)
        state.save_svg("debug.svg")
        assert (state._zobrist_hash == jax.jit(_zobrist_hash)(state)).all()


def test_action():
    state = State._from_fen("k4/5/5/1Q3/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_001.svg")
    action = Action._from_label(jnp.int32(306))
    assert action.from_ == p("b2")
    assert action.to == p("b1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(309))
    assert action.from_ == p("b2")
    assert action.to == p("b5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(314))
    assert action.from_ == p("b2")
    assert action.to == p("a2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(317))
    assert action.from_ == p("b2")
    assert action.to == p("e2")
    assert action.underpromotion == -1
    # fail
    action = Action._from_label(jnp.int32(322))
    assert action.from_ == p("b2")
    assert action.to == p("a1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(325))
    assert action.from_ == p("b2")
    assert action.to == p("e5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(330))
    assert action.from_ == p("b2")
    assert action.to == p("a3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(331))
    assert action.from_ == p("b2")
    assert action.to == p("c1")
    assert action.underpromotion == -1
    # knight moves
    # fail
    state = State._from_fen("k4/5/2N2/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_002.svg")
    action = Action._from_label(jnp.int32(629))
    assert action.from_ == p("c3")
    assert action.to == p("a2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(630))
    assert action.from_ == p("c3")
    assert action.to == p("a4")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(631))
    assert action.from_ == p("c3")
    assert action.to == p("b1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(632))
    assert action.from_ == p("c3")
    assert action.to == p("b5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(633))
    assert action.from_ == p("c3")
    assert action.to == p("e2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(634))
    assert action.from_ == p("c3")
    assert action.to == p("e4")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(635))
    assert action.from_ == p("c3")
    assert action.to == p("d1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(636))
    assert action.from_ == p("c3")
    assert action.to == p("d5")
    assert action.underpromotion == -1
    # underpromotion
    state = State._from_fen("r1r1k/1P3/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_003.svg")
    action = Action._from_label(jnp.int32(392))
    assert action.from_ == p("b4")
    assert action.to == p("b5")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(393))
    assert action.from_ == p("b4")
    assert action.to == p("c5")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(394))
    assert action.from_ == p("b4")
    assert action.to == p("a5")
    assert action.underpromotion == 0  # rook
    # black turn
    state = State._from_fen("k4/3q1/5/5/4K b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_004.svg")
    # 上（上下はそのまま）
    action = Action._from_label(jnp.int32(797))
    assert action.from_ == p("d4", True)
    assert action.to == p("d3", True)
    assert action.underpromotion == -1
    # 左（左右は鏡写し）
    action = Action._from_label(jnp.int32(805))
    assert action.from_ == p("d4", True)
    assert action.to == p("e4", True)
    assert action.underpromotion == -1


def test_observe():
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (5, 5, 115)
    expected = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected).all()
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    assert state._turn == 0
    assert (state.observation[:, :, 112] == 0).all()

    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_001.svg")
    assert state._turn == 1
    assert (state.observation[:, :, 112] == 1).all()

    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_002.svg")
    expected = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected).all()
    expected = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    print(state.observation[:, :, 14])
    print(state.observation[:, :, 20])
    assert (state.observation[:, :, 14] == expected).all()
    assert state._turn == 0
    assert (state.observation[:, :, 112] == 0).all()


def test_api():
    import pgx
    env = pgx.make("gardner_chess")
    pgx.v1_api_test(env, 3)
