import jax
import jax.numpy as jnp
import pgx
from pgx.chess import State, Action, KING, _rotate, Chess, QUEEN, EMPTY, ROOK, PAWN, _legal_action_mask, CAN_MOVE

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

    # black turn
    state = State._from_fen("k7/8/8/8/3qP3/8/8/R3K2R b KQ e3 0 1")
    state.save_svg("tests/assets/chess/action_004.svg")
    # 上（上下はそのまま）
    action = Action._from_label(jnp.int32(p("d4", True) * 73 + 16))
    assert action.from_ == p("d4", True)
    assert action.to == p("d3", True)
    assert action.underpromotion == -1
    # 左（左右は鏡写し）
    action = Action._from_label(jnp.int32(p("d4", True) * 73 + 29))
    assert action.from_ == p("d4", True)
    assert action.to == p("c4", True)
    assert action.underpromotion == -1

    # TODO: black underpromotion


def test_step():
    # normal step
    state = State._from_fen("1k6/8/8/8/8/8/1Q6/7K w - - 0 1")
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
    state = State._from_fen("1k6/8/8/8/8/8/8/R3K2R w KQ - 0 1")
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
    state = State._from_fen("1k6/8/8/8/3pP3/8/8/R3K2R b KQ e3 0 1")
    state.save_svg("tests/assets/chess/step_008.svg")
    assert state.board[p("e4", True)] == -PAWN
    next_state = step(state, jnp.int32(p("d4", True) * 73 + 44))
    next_state.save_svg("tests/assets/chess/step_009.svg")
    assert next_state.board[p("e3")] == -PAWN
    assert next_state.board[p("e4")] == EMPTY
    state = State._from_fen("1k6/8/8/8/3p4/8/4P3/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/step_010.svg")
    next_state = step(state, jnp.int32(p("e2") * 73 + 17))  # UP 2
    next_state.save_svg("tests/assets/chess/step_011.svg")
    assert next_state.en_passant == p("e3")
    state = State._from_fen("1k6/p7/8/8/3p4/8/4P3/R3K2R b KQ - 0 1")
    state.save_svg("tests/assets/chess/step_012.svg")
    next_state = step(state, jnp.int32(p("a7", True) * 73 + 17))  # UP 2
    next_state.save_svg("tests/assets/chess/step_013.svg")
    assert next_state.en_passant == p("a6")  # en passant is always white view


def test_legal_action_mask():
    # init board
    state = State._from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_001.svg")
    assert state.legal_action_mask.sum() == 20

    # init pawn
    state = State._from_fen("7k/8/8/8/8/8/P7/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_002.svg")
    state = step(state, jnp.int32(4104))  # BKing: h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_003.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4

    # init pawn (blocked)
    state = State._from_fen("7k/8/8/8/p7/8/P7/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_004.svg")
    state = step(state, jnp.int32(4104))  # BKing: h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_005.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # moved pawn
    state = State._from_fen("7k/8/8/8/8/P7/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_006.svg")
    state = step(state, jnp.int32(4104))  # h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_007.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4

    # moved pawn (blocked)
    state = State._from_fen("7k/8/8/8/p7/P7/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_008.svg")
    state = step(state, jnp.int32(4104))  # BKing: h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_009.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # pawn capture
    state = State._from_fen("7k/8/8/8/8/p1p5/1P6/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_010.svg")
    state = step(state, jnp.int32(4104))  # BKing: h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_011.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 6

    # promotion (white)
    state = State._from_fen("b1r4k/1P6/8/8/8/8/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_012.svg")
    state = step(state, jnp.int32(4104))  # BKing: h8 -> h7
    state.save_svg("tests/assets/chess/legal_action_mask_013.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 15

    # promotion (black)
    state = State._from_fen("7k/8/8/8/8/8/6p1/K4R1B w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_014.svg")
    state = step(state, jnp.int32(16))  # a1 -> a2
    state.save_svg("tests/assets/chess/legal_action_mask_015.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 15

    # ignore check
    state = State._from_fen("7k/8/8/8/8/1p6/P7/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_016.svg")
    state = step(state, jnp.int32(965))  #
    state.save_svg("tests/assets/chess/legal_action_mask_017.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 2

    # en passant
    state = State._from_fen("7k/7p/8/6P1/8/8/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_018.svg")
    state = step(state, jnp.int32(4178))  # BPawn: h7 -> h5
    state.save_svg("tests/assets/chess/legal_action_mask_019.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5
