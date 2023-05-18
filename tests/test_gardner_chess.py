import jax
import jax.numpy as jnp
import pgx
from pgx.gardner_chess import State, Action, GardnerChess

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
    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_001.svg")
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


def test_step():
    # normal step
    # queen
    state = State._from_fen("k4/5/5/1Q3/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_001.svg")
    assert state._board[p("b1")] == jnp.int8(0)
    assert state._board[p("e5")] == jnp.int8(0)
    state1 = step(state, jnp.int32(306))
    state1.save_svg("tests/assets/gardner_chess/step_002.svg")
    assert state1._board[p("b1", True)] == -jnp.int8(5)
    state2 = step(state, jnp.int32(325))
    state2.save_svg("tests/assets/gardner_chess/step_003.svg")
    assert state2._board[p("e5", True)] == -jnp.int8(5)

    # knight
    state = State._from_fen("k4/5/2N2/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_004.svg")
    assert state._board[p("b1")] == jnp.int8(0)
    assert state._board[p("e4")] == jnp.int8(0)
    state1 = step(state, jnp.int32(631))
    state1.save_svg("tests/assets/gardner_chess/step_005.svg")
    assert state1._board[p("b1", True)] == -jnp.int8(2)
    state2 = step(state, jnp.int32(634))
    state2.save_svg("tests/assets/gardner_chess/step_006.svg")
    assert state2._board[p("e4", True)] == -jnp.int8(2)

    # promotion
    state = State._from_fen("r1r1k/1P3/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_007.svg")
    assert state._board[p("b5")] == jnp.int8(0)
    assert state._board[p("c5")] == -jnp.int8(4)
    # underpromotion
    next_state = step(state, jnp.int32(392))
    next_state.save_svg("tests/assets/gardner_chess/step_008.svg")
    assert next_state._board[p("b5", True)] == -jnp.int8(4)
    # promotion to queen
    next_state = step(state, jnp.int32(421))
    next_state.save_svg("tests/assets/gardner_chess/step_008.svg")
    assert next_state._board[p("b8", True)] == -jnp.int8(5)


def test_api():
    import pgx
    env = pgx.make("gardner_chess")
    pgx.v1_api_test(env, 3)
