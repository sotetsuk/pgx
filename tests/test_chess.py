import jax
import jax.numpy as jnp
import pgx
from pgx._chess import State, Action, KING, _rotate, Chess, QUEEN, EMPTY, ROOK

env = Chess()
init = jax.jit(env.init)
step = jax.jit(env.step)

pgx.set_visualization_config(color_theme="dark")


def p(s: str, b=False):
    """
    >>> p("e3")
    34
    >>> p("e3", b=True)
    37
    """
    x = "abcdefgh".index(s[0])
    offset = int(s[1]) - 1 if not b else 8 - int(s[1])
    return x * 8 + offset


def test_action():
    # See #704
    # queen moves
    state = State._from_fen("k7/8/8/8/8/8/1Q6/7K w - - 0 1")
    state.save_svg("tests/assets/chess/action_001.svg")
    action = Action._from_label(jnp.int32(672))
    assert action.from_ == p("b2")
    assert action.to == p("b1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(673))
    assert action.from_ == p("b2")
    assert action.to == p("b3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(686))
    assert action.from_ == p("b2")
    assert action.to == p("a2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(687))
    assert action.from_ == p("b2")
    assert action.to == p("c2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(700))
    assert action.from_ == p("b2")
    assert action.to == p("a1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(701))
    assert action.from_ == p("b2")
    assert action.to == p("c3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(714))
    assert action.from_ == p("b2")
    assert action.to == p("a3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(715))
    assert action.from_ == p("b2")
    assert action.to == p("c1")
    assert action.underpromotion == -1
    # knight moves
    state = State._from_fen("k7/8/8/8/2N5/8/P7/7K w - - 0 1")
    state.save_svg("tests/assets/chess/action_002.svg")
    action = Action._from_label(jnp.int32(1452))
    assert action.from_ == p("c4")
    assert action.to == p("a3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1453))
    assert action.from_ == p("c4")
    assert action.to == p("a5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1454))
    assert action.from_ == p("c4")
    assert action.to == p("b2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1455))
    assert action.from_ == p("c4")
    assert action.to == p("b6")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1456))
    assert action.from_ == p("c4")
    assert action.to == p("e3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1457))
    assert action.from_ == p("c4")
    assert action.to == p("e5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1458))
    assert action.from_ == p("c4")
    assert action.to == p("d2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(1459))
    assert action.from_ == p("c4")
    assert action.to == p("d6")
    assert action.underpromotion == -1
    # underpromotion
    state = State._from_fen("r1r4k/1P6/8/8/8/8/P7/7K w - - 0 1")
    state.save_svg("tests/assets/chess/action_003.svg")
    action = Action._from_label(jnp.int32(1022))
    assert action.from_ == p("b7")
    assert action.to == p("b8")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(1023))
    assert action.from_ == p("b7")
    assert action.to == p("c8")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(1024))
    assert action.from_ == p("b7")
    assert action.to == p("a8")
    assert action.underpromotion == 0  # rook


def test_step():
    # normal step
    state = State._from_fen("k7/8/8/8/8/8/1Q6/7K w - - 0 1")
    state.save_svg("tests/assets/chess/step_001.svg")
    assert state.board[p("b1")] == EMPTY
    state = step(state, jnp.int32(672))
    state.save_svg("tests/assets/chess/step_002.svg")
    assert state.board[p("b1", True)] == -QUEEN

    # promotion
    state = State._from_fen("r1r4k/1P6/8/8/8/8/P7/7K w - - 0 1")
    state.save_svg("tests/assets/chess/step_002.svg")
    assert state.board[p("b8")] == EMPTY
    # underpromotion
    next_state = step(state, jnp.int32(1022))
    next_state.save_svg("tests/assets/chess/step_003.svg")
    assert next_state.board[p("b8", True)] == -ROOK
    # promotion to queen
    next_state = step(state, jnp.int32(p("b7") * 73 + 16))
    next_state.save_svg("tests/assets/chess/step_004.svg")
    assert next_state.board[p("b8", True)] == -QUEEN

    # castling
    state = State._from_fen("k7/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/step_005.svg")
    # left
    next_state = step(state, jnp.int32(p("e1") * 73 + 28))
    next_state.save_svg("tests/assets/chess/step_006.svg")
    assert next_state.board[p("c1", True)] == -KING
    assert next_state.board[p("d1", True)] == -ROOK  # castling
    assert next_state.board[p("a1", True)] == EMPTY  # castling
    # right
    next_state = step(state, jnp.int32(p("e1") * 73 + 31))
    next_state.save_svg("tests/assets/chess/step_007.svg")
    assert next_state.board[p("g1", True)] == -KING
    assert next_state.board[p("f1", True)] == -ROOK  # castling
    assert next_state.board[p("h1", True)] == EMPTY  # castling

    # en passant
    state = State._from_fen("k7/8/8/8/3pP3/8/8/R3K2R b KQ e3 0 1")
    state.save_svg("tests/assets/chess/step_008.svg")
    next_state = step(state, jnp.int32(p("d4", True) * 73 + 57))  # UP LEFT
    next_state.save_svg("tests/assets/chess/step_009.svg")

    state = State._from_fen("k7/8/8/8/3pP3/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/step_010.svg")
    next_state = step(state, jnp.int32(p("e4") * 73 + 57))  # UP LEFT
    next_state.save_svg("tests/assets/chess/step_011.svg")



