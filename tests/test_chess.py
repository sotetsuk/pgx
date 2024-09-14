import jax
import jax.numpy as jnp
import pgx
from pgx.chess import State, Chess
from pgx._src.games.chess import GameState, Action, KING, QUEEN, EMPTY, ROOK, PAWN, _legal_action_mask, CAN_MOVE, _zobrist_hash, INIT_ZOBRIST_HASH
from pgx.experimental.utils import act_randomly
from pgx.experimental.chess import from_fen, to_fen

env = Chess()
init = jax.jit(env.init)
step = jax.jit(env.step)
act_randomly = jax.jit(act_randomly)

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
    state = from_fen("k7/8/8/8/8/8/1Q6/7K w - - 0 1")
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
    state = from_fen("k7/8/8/8/2N5/8/P7/7K w - - 0 1")
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
    state = from_fen("r1r4k/1P6/8/8/8/8/P7/7K w - - 0 1")
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
    state = from_fen("k7/8/8/8/3qP3/8/8/R3K2R b KQ e3 0 1")
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
    state = from_fen("1k6/8/8/8/8/8/1Q6/7K w - - 0 1")
    state.save_svg("tests/assets/chess/step_001.svg")
    assert state._x.board[p("b1")] == EMPTY
    state = step(state, jnp.int32(672))
    state.save_svg("tests/assets/chess/step_002.svg")
    assert state._x.board[p("b1", True)] == -QUEEN

    # promotion
    state = from_fen("r1r4k/1P6/8/8/8/8/P7/7K w - - 0 1")
    state.save_svg("tests/assets/chess/step_002.svg")
    assert state._x.board[p("b8")] == EMPTY
    # underpromotion
    next_state = step(state, jnp.int32(1022))
    next_state.save_svg("tests/assets/chess/step_003.svg")
    assert next_state._x.board[p("b8", True)] == -ROOK
    # promotion to queen
    next_state = step(state, jnp.int32(p("b7") * 73 + 16))
    next_state.save_svg("tests/assets/chess/step_004.svg")
    assert next_state._x.board[p("b8", True)] == -QUEEN

    # castling
    state = from_fen("1k6/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/step_005.svg")
    # left
    next_state = step(state, jnp.int32(p("e1") * 73 + 28))
    next_state.save_svg("tests/assets/chess/step_006.svg")
    assert next_state._x.board[p("c1", True)] == -KING
    assert next_state._x.board[p("d1", True)] == -ROOK  # castling
    assert next_state._x.board[p("a1", True)] == EMPTY  # castling
    # right
    next_state = step(state, jnp.int32(p("e1") * 73 + 31))
    next_state.save_svg("tests/assets/chess/step_007.svg")
    assert next_state._x.board[p("g1", True)] == -KING
    assert next_state._x.board[p("f1", True)] == -ROOK  # castling
    assert next_state._x.board[p("h1", True)] == EMPTY  # castling

    # en passant
    state = from_fen("1k6/8/8/8/3pP3/8/8/R3K2R b KQ e3 0 1")
    state.save_svg("tests/assets/chess/step_008.svg")
    assert state._x.board[p("e4", True)] == -PAWN
    next_state = step(state, jnp.int32(p("d4", True) * 73 + 44))
    next_state.save_svg("tests/assets/chess/step_009.svg")
    assert next_state._x.board[p("e3")] == -PAWN
    assert next_state._x.board[p("e4")] == EMPTY
    state = from_fen("1k6/8/8/8/3p4/8/4P3/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/step_010.svg")
    next_state = step(state, jnp.int32(p("e2") * 73 + 17))  # UP 2
    next_state.save_svg("tests/assets/chess/step_011.svg")
    assert next_state._x.en_passant == p("e3", True)
    state = from_fen("1k6/p7/8/8/3p4/8/4P3/R3K2R b KQ - 0 1")
    state.save_svg("tests/assets/chess/step_012.svg")
    next_state = step(state, jnp.int32(p("a7", True) * 73 + 17))  # UP 2
    next_state.save_svg("tests/assets/chess/step_013.svg")
    assert next_state._x.en_passant == p("a6")  # en passant is always white view


def test_legal_action_mask():
    # init board
    state = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_001.svg")
    assert state.legal_action_mask.sum() == 20

    # init pawn
    state = from_fen("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_002.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4

    # init pawn (blocked)
    state = from_fen("7k/8/8/8/p7/8/P7/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_003.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # moved pawn
    state = from_fen("7k/8/8/8/8/P7/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_004.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4

    # moved pawn (blocked)
    state = from_fen("7k/8/8/8/p7/P7/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_005.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # pawn capture
    state = from_fen("7k/8/8/8/8/p1p5/1P6/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_006.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 6

    # promotion (white)
    state = from_fen("b1r4k/1P6/8/8/8/8/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_007.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 15

    # promotion (black)
    state = from_fen("7k/8/8/8/8/8/6p1/K4R1B b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_008.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 15

    # ignore check
    state = from_fen("7k/8/8/8/8/8/Pp6/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_009.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 2

    # castling (cannot)
    state = from_fen("8/7k/7p/8/8/8/8/R3K2R w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_010.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 22

    # black
    state = from_fen("r3k2r/8/8/8/8/P7/K7/8 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_011.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 22

    # castling (cannot)
    state = from_fen("8/7k/7p/8/8/8/8/RN2K1NR w KQ - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_012.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 23

    # black
    state = from_fen("rn2k1nr/8/8/8/8/P7/K7/8 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_013.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 23

    # castling (cannot)
    state = from_fen("2r3r1/7k/7p/8/8/8/8/R3K2R w KQ - 1 2")
    state.save_svg("tests/assets/chess/legal_action_mask_014.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 22

    # black
    state = from_fen("r3k2r/8/8/8/8/P7/K7/2R3R1 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_015.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 22

    # castling (cannot)
    # 落ちる
    state = from_fen("8/7k/7p/8/8/8/4p3/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_016.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 20

    # black
    state = from_fen("r3k2r/8/4N3/8/8/P7/K7/8 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_017.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 20

    # castling (cannot)
    state = from_fen("8/7k/7p/8/8/8/3p4/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_018.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5

    # black
    state = from_fen("r3k2r/8/3N4/8/8/P7/K7/8 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_019.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4

    # castling(one side)
    state = from_fen("6r1/7k/7p/8/8/8/8/R3K2R w KQ - 1 2")
    state.save_svg("tests/assets/chess/legal_action_mask_020.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 23

    # black
    state = from_fen("r3k2r/8/8/8/8/P7/K7/2R5 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_021.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 23

    # castling(one side)
    state = from_fen("8/7k/7p/8/8/3b4/8/R3K2R w KQ - 1 2")
    state.save_svg("tests/assets/chess/legal_action_mask_022.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 21

    # black
    state = from_fen("r3k2r/8/3B4/8/8/P7/K7/8 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_023.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 21

    # castling
    state = from_fen("q7/1r5k/7r/8/8/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_024.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 24

    # black
    state = from_fen("r3k2r/8/8/8/8/R7/1R6/6KQ b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_025.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 24

    # castling
    state = from_fen("8/7k/7p/8/8/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_026.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 24

    # black
    state = from_fen("r3k2r/8/8/8/8/P7/K7/8 b kq - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_027.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 24

    # en passant
    state = from_fen("7k/7p/8/6P1/8/8/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_028.svg")
    state = step(state, jnp.int32(4178))  # BPawn: h7 -> h5
    state.save_svg("tests/assets/chess/legal_action_mask_029.svg")
    print(state._x.en_passant)
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5

    # en passant (black)
    state = from_fen("7k/8/8/8/1p6/8/P7/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_030.svg")
    state = step(state, jnp.int32(90))  # WPawn: a2 -> a4
    state.save_svg("tests/assets/chess/legal_action_mask_031.svg")
    print(state._x.en_passant)
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5

    # en passant
    state = from_fen("7k/4p3/8/3P1P2/8/8/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_032.svg")
    state = step(state, jnp.int32(2426))  # BPawn: e7 -> e5
    state.save_svg("tests/assets/chess/legal_action_mask_033.svg")
    print(state._x.en_passant)
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 7

    # en passant (black)
    state = from_fen("7k/8/8/8/2p1p3/8/3P4/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_034.svg")
    state = step(state, jnp.int32(1842))  # WPawn: d2 -> d4
    state.save_svg("tests/assets/chess/legal_action_mask_035.svg")
    print(state._x.en_passant)
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 7

    # pinned
    state = from_fen("7k/8/8/8/8/8/8/KR5q w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_036.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 8

    # pinned by promotion
    state = from_fen("6rk/P7/8/8/8/8/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_037.svg")
    state = step(state, jnp.int32(454))  # WPawn: a7 -> a8 Queen Promotion
    state.save_svg("tests/assets/chess/legal_action_mask_038.svg")
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 8

    # double check
    state = from_fen("1q6/R2N3k/8/8/8/8/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_039.svg")
    state = step(state, jnp.int32(2260))  # WPawn: f7 -> f8 Night Promotion
    state.save_svg("tests/assets/chess/legal_action_mask_040.svg")
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # double check by promotion
    state = from_fen("1q6/R4P1k/8/8/8/8/8/K7 w - - 0 1")
    state.save_svg("tests/assets/chess/legal_action_mask_041.svg")
    state = step(state, jnp.int32(3364))  # WPawn: f7 -> f8 Night Promotion
    state.save_svg("tests/assets/chess/legal_action_mask_042.svg")
    print(to_fen(state))
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3


def test_terminal():
    # checkmate (white win)
    state = from_fen("7k/7R/5N2/8/8/8/8/K7 b - - 0 1")
    state.save_svg("tests/assets/chess/terminal_001.svg")
    print(to_fen(state))
    assert state.terminated
    assert state.rewards[state.current_player] == -1
    assert state.rewards[1 - state.current_player] == 1.

    # stalemate
    state = from_fen("k7/8/1Q6/K7/8/8/8/8 b - - 0 1")
    state.save_svg("tests/assets/chess/terminal_002.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # 50-move draw rule
    # FEN is from https://www.chess.com/terms/fen-chess#halfmove-clock
    state = from_fen("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50")
    state.save_svg("tests/assets/chess/terminal_003.svg")
    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/chess/terminal_004.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # insufficient pieces
    # K vs K
    state = from_fen("k7/8/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_005.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K
    state = from_fen("k7/8/8/8/8/8/8/6BK w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_006.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K vs K+B
    state = from_fen("kb6/8/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_007.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+N vs K
    state = from_fen("k7/8/8/8/8/8/8/6NK w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_008.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K vs K+N
    state = from_fen("kn6/8/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_009.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in Black tile)
    state = from_fen("kb6/8/8/8/8/8/8/6BK w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_010.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in White tile)
    state = from_fen("k1b1B3/8/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_011.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # insufficient cases by underpromotion
    # K+B vs K
    state = from_fen("k7/7P/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_012.svg")
    state = step(state, jnp.int32(4529))
    state.save_svg("tests/assets/chess/terminal_013.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+N vs K
    state = from_fen("k7/7P/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_014.svg")
    state = step(state, jnp.int32(4532))
    state.save_svg("tests/assets/chess/terminal_015.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B(Bishop in Black tile)
    state = from_fen("k1b5/4P3/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_016.svg")
    state = step(state, jnp.int32(2777))
    state.save_svg("tests/assets/chess/terminal_017.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in White tile)
    state = from_fen("kb6/3P4/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_018.svg")
    state = step(state, jnp.int32(2193))
    state.save_svg("tests/assets/chess/terminal_019.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B*2 vs K(Bishop in Black tile)
    state = from_fen("k1B5/4P3/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_020.svg")
    state = step(state, jnp.int32(2777))
    state.save_svg("tests/assets/chess/terminal_021.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B*2 vs K (Bishop in White tile)
    state = from_fen("kB6/3P4/8/8/8/8/8/7K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_022.svg")
    state = step(state, jnp.int32(2193))
    state.save_svg("tests/assets/chess/terminal_023.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # stalemate with pin
    state = from_fen("kbR5/pn6/P1B5/8/8/8/8/7K b - - 0 1")
    state.save_svg("tests/assets/chess/terminal_024.svg")
    print(to_fen(state))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # perpetual check
    state = from_fen("7k/8/4r1pp/8/8/4q3/8/5Q1K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_025.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(2942))
    state.save_svg("tests/assets/chess/terminal_026.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    print(to_fen(state))
    state = step(state, jnp.int32(4104))
    state.save_svg("tests/assets/chess/terminal_027.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(3446))
    state.save_svg("tests/assets/chess/terminal_028.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(4176))
    state.save_svg("tests/assets/chess/terminal_029.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(3374))
    state.save_svg("tests/assets/chess/terminal_030.svg")
    assert (state.observation[:, :, 12] == 0).all()
    assert (state.observation[:, :, 13] == 1).all()
    print(to_fen(state))
    state = step(state, jnp.int32(4104))
    state = step(state, jnp.int32(3446))
    state = step(state, jnp.int32(4176))
    state = step(state, jnp.int32(3374))
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # repetition
    state = from_fen("r6k/8/8/8/8/8/8/R6K w - - 0 1")
    state.save_svg("tests/assets/chess/terminal_031.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/terminal_032.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/terminal_033.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(614))
    state.save_svg("tests/assets/chess/terminal_034.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(614))
    state.save_svg("tests/assets/chess/terminal_035.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(1196))
    state.save_svg("tests/assets/chess/terminal_036.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(1196))
    state.save_svg("tests/assets/chess/terminal_037.svg")
    assert (state.observation[:, :, 12] == 0).all()
    assert (state.observation[:, :, 13] == 1).all()
    state = step(state, jnp.int32(30))
    state = step(state, jnp.int32(30))
    state = step(state, jnp.int32(613))
    state = step(state, jnp.int32(613))
    assert state.terminated
    assert (state.rewards == 0.0).all()


def test_buggy_samples():
    # Found buggy samples by random playing debug

    # Bishop by bishop
    state = from_fen("r1q1k1nr/2p2p1B/p3pb2/1p4pP/P1pBP3/5P1N/R2P2KP/1Nn2Q1R w kq - 4 23")
    state.save_svg("tests/assets/chess/buggy_samples_001.svg")
    expected_legal_actions = [88, 89, 103, 104, 235, 263, 652, 656, 1841, 2012, 2013, 2014, 2015, 2016, 2026, 2027, 2028, 2029, 2030, 2031, 2571, 2936, 2947, 2948, 2949, 2950, 2975, 2976, 2977, 3082, 3592, 3593, 3606, 4117, 4299, 4300, 4301, 4302, 4396, 4568, 4569, 4583]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # wrong castling
    state = from_fen("4r1nr/4k2B/2p5/p3P2P/Ppp3RP/3P1P2/3N4/R4K2 b - - 2 49")
    state.save_svg("tests/assets/chess/buggy_samples_002.svg")
    expected_legal_actions = [892, 1330, 1476, 1504, 2362, 2363, 2364, 2365, 2366, 2425, 2438, 2439, 2452, 2467, 3572, 3576, 4104]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # half-movecount reset when underpromotion happens
    state = from_fen("8/3kr3/2R2R1P/p3P3/Pr3P1P/3P4/1p6/4K3 b - - 2 68")
    state.save_svg("tests/assets/chess/buggy_samples_003.svg")
    state = step(state, 1025)
    state.save_svg("tests/assets/chess/buggy_samples_004.svg")
    assert to_fen(state) == "8/3kr3/2R2R1P/p3P3/Pr3P1P/3P4/8/1b2K3 w - - 0 69"

    # wrong insufficient piece termination
    state = from_fen("7k/8/5K2/8/P7/8/8/8 b - - 0 152")
    state.save_svg("tests/assets/chess/buggy_samples_005.svg")
    assert not state.terminated

    # wrong castling
    state = from_fen("r3k1n1/p1q4r/2pb4/4PP1p/1BQ2p2/1P3NP1/5K1P/R1N2Q1R b q - 0 27")
    state.save_svg("tests/assets/chess/buggy_samples_006.svg")
    expected_legal_actions = [30, 31, 32, 89, 90, 1256, 1270, 1271, 1272, 1273, 1274, 1284, 1297, 1298, 1299, 1330, 1942, 1954, 1955, 1956, 1957, 2352, 2364, 2365, 2366, 2393, 3256, 3570, 3572, 3576, 4176, 4177, 4187, 4188, 4189, 4190, 4323]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # wrong underpromotions
    state = from_fen("b5K1/5r1P/3k2n1/8/4p3/8/8/8 w - - 1 119")
    state.save_svg("tests/assets/chess/buggy_samples_007.svg")
    expected_legal_actions = [4058, 4526, 4529, 4532, 4542]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # wrong underpromotion from h5 (due to wrong en passant)
    state = from_fen("3r2k1/5nbp/1p6/pPpr3P/2bppBp1/2P1NBP1/P1R5/2RK4 w - a6 0 43")
    state.save_svg("tests/assets/chess/buggy_samples_008.svg")
    expected_legal_actions = [89, 90, 933, 1196, 1197, 1270, 1271, 1272, 1273, 1274, 1275, 1358, 1768, 1782, 2548, 2550, 2551, 2552, 2553, 2554, 3109, 3110, 3123, 3124, 3125, 3183, 3184, 3193, 3194, 3195, 3196, 4396]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # wrong castling right
    state = from_fen("rn1qkbr1/2pp1pp1/p3pn1p/PQ6/1P6/6P1/2PP1PbP/R1BNKBNR b KQq - 1 10")
    state.save_svg("tests/assets/chess/buggy_samples_009.svg")
    state = step(state, 3986)
    state.save_svg("tests/assets/chess/buggy_samples_010.svg")
    assert to_fen(state) == "rn1qkbr1/2pp1pp1/p3pn1p/PQ6/1P6/6P1/2PP1P1P/R1BNKBNb w Qq - 0 11"

    # wrong en passant by pinned pawn
    state = from_fen("rn6/1b6/8/p1Br1k1p/PPp1pPpP/NRp3P1/2B4R/2K5 b - f3 0 54")
    print(state._x.en_passant)
    state.save_svg("tests/assets/chess/buggy_samples_011.svg")
    expected_legal_actions = [16, 17, 263, 652, 654, 656, 701, 714, 715, 1517, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 2000, 2001, 3154, 3182, 3197, 3853]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # wrong rook position when castling
    state = from_fen("r1b1k2r/2q2pp1/p1pbpn1p/2p5/2P2P2/1PNP4/P5PP/R1BQNR1K b kq - 0 15")
    state.save_svg("tests/assets/chess/buggy_samples_012.svg")
    state = step(state, jnp.int32(2367))
    state.save_svg("tests/assets/chess/buggy_samples_013.svg")
    state = step(state, jnp.int32(1796))
    state.save_svg("tests/assets/chess/buggy_samples_014.svg")
    expected_legal_actions = [16, 30, 162, 1212, 1225, 1269, 1270, 1271, 1272, 1284, 1297, 1298, 1299, 1942, 1943, 1956, 2498, 2948, 2949, 3131, 3132, 3133, 3134, 3135, 3136, 3138, 3534, 3548, 3593, 3594, 4250]
    assert state.legal_action_mask.sum() == len(expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"


def test_observe():
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (8, 8, 119)

    # my pawn
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 0] == expected).all()
    # my king
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 5] == expected).all()

    # opp pawn
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 6] == expected).all()
    # opp king
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 11] == expected).all()

    # repetitions
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()

    # color
    assert state._x.color == 0
    assert (state.observation[:, :, 112] == 0).all()


    state = step(state, jnp.int32(89))

    # my pawn
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 0] == expected).all()

    # opp pawn
    expected = jnp.float32(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    assert (state.observation[:, :, 6] == expected).all()

    # repetitions
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()

    # color
    assert state._x.color == 1
    assert (state.observation[:, :, 112] == 1).all()

    # repetition
    # not repetition patterns
    # not the same turn
    state = from_fen("r6k/8/8/8/8/8/8/R6K w - - 0 1")
    state.save_svg("tests/assets/chess/observe_001.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_002.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(31))
    state.save_svg("tests/assets/chess/observe_003.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(614))
    state.save_svg("tests/assets/chess/observe_004.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(1196))
    state.save_svg("tests/assets/chess/observe_005.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    state = step(state, jnp.int32(1196))
    state.save_svg("tests/assets/chess/observe_006.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()

    # not the same castling rights
    state = from_fen("r5k1/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    state.save_svg("tests/assets/chess/observe_007.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_008.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_009.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_010.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_011.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()

    # not the same en-passant position
    state = from_fen("r5k1/8/8/8/8/8/P7/R3K3 w KQ - 0 1")
    state.save_svg("tests/assets/chess/observe_012.svg")
    state = step(state, jnp.int32(90))
    state.save_svg("tests/assets/chess/observe_013.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_014.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_015.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_016.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_017.svg")
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()

    # castling rights
    state = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state.save_svg("tests/assets/chess/observe_018.svg")
    assert (state.observation[:, :, 114] == 1).all()
    assert (state.observation[:, :, 115] == 1).all()
    assert (state.observation[:, :, 116] == 1).all()
    assert (state.observation[:, :, 117] == 1).all()
    state = step(state, jnp.int32(16))
    state.save_svg("tests/assets/chess/observe_019.svg")
    assert (state.observation[:, :, 116] == 0).all()
    state = step(state, jnp.int32(16))
    state.save_svg("tests/assets/chess/observe_020.svg")
    assert (state.observation[:, :, 114] == 0).all()
    state = step(state, jnp.int32(4104))
    state.save_svg("tests/assets/chess/observe_021.svg")
    assert (state.observation[:, :, 117] == 0).all()
    state = step(state, jnp.int32(4104))
    state.save_svg("tests/assets/chess/observe_022.svg")
    assert (state.observation[:, :, 115] == 0).all()

    # castling rights
    state = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state.save_svg("tests/assets/chess/observe_023.svg")
    assert (state.observation[:, :, 114] == 1).all()
    assert (state.observation[:, :, 115] == 1).all()
    assert (state.observation[:, :, 116] == 1).all()
    assert (state.observation[:, :, 117] == 1).all()
    state = step(state, jnp.int32(2352))
    state.save_svg("tests/assets/chess/observe_024.svg")
    assert (state.observation[:, :, 116] == 0).all()
    assert (state.observation[:, :, 117] == 0).all()
    state = step(state, jnp.int32(2352))
    state.save_svg("tests/assets/chess/observe_025.svg")
    assert (state.observation[:, :, 114] == 0).all()
    assert (state.observation[:, :, 115] == 0).all()

    # castling rights
    state = from_fen("r3k1nr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state.save_svg("tests/assets/chess/observe_026.svg")
    assert (state.observation[:, :, 114] == 1).all()
    assert (state.observation[:, :, 115] == 1).all()
    assert (state.observation[:, :, 116] == 1).all()
    assert (state.observation[:, :, 117] == 1).all()
    state = step(state, jnp.int32(4110))
    state.save_svg("tests/assets/chess/observe_027.svg")
    assert (state.observation[:, :, 115] == 0).all()
    assert (state.observation[:, :, 117] == 0).all()
    state = step(state, jnp.int32(22))
    state.save_svg("tests/assets/chess/observe_028.svg")
    assert (state.observation[:, :, 114] == 0).all()
    assert (state.observation[:, :, 116] == 0).all()

    # non progress move count
    state = from_fen("r3k2r/p7/8/8/8/8/8/R3K2R w KQkq - 0 1")
    state.save_svg("tests/assets/chess/observe_029.svg")
    state = step(state, jnp.int32(16))
    state.save_svg("tests/assets/chess/observe_030.svg")
    assert (state.observation[:, :, 118] == 0.01).all()
    state = step(state, jnp.int32(4104))
    state.save_svg("tests/assets/chess/observe_031.svg")
    assert (state.observation[:, :, 118] == 0.02).all()
    # capture move
    state = step(state, jnp.int32(4109))
    state.save_svg("tests/assets/chess/observe_032.svg")
    assert (state.observation[:, :, 118] == 0).all()
    # pawn move
    state = step(state, jnp.int32(90))
    state.save_svg("tests/assets/chess/observe_032.svg")
    assert (state.observation[:, :, 118] == 0).all()

    # repetition observation
    # normal
    state = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    state.save_svg("tests/assets/chess/observe_033.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_034.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_035.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_036.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_037.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_034.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_035.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(613))
    # "tests/assets/chess/observe_036.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 13] == 1.).all()  # rep

    # with castling rights
    state = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    # "tests/assets/chess/observe_033.svg"
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_034.svg"
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_035.svg"
    state = step(state, jnp.int32(613))
    # "tests/assets/chess/observe_036.svg"
    state = step(state, jnp.int32(613))
    # "tests/assets/chess/observe_037.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 0 + 13] == 0).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 4 + 13] == 0).all()
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_034.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 0 + 13] == 0).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 4 + 13] == 0).all()
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_035.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(613))
    # "tests/assets/chess/observe_036.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep

    # with en-passant
    state = from_fen("r3k2r/8/8/8/8/8/P7/R3K2R w - - 0 1")
    state.save_svg("tests/assets/chess/observe_038.svg")
    state = step(state, jnp.int32(90))
    state.save_svg("tests/assets/chess/observe_039.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_040.svg")
    state = step(state, jnp.int32(30))
    state.save_svg("tests/assets/chess/observe_041.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_042.svg")
    state = step(state, jnp.int32(613))
    state.save_svg("tests/assets/chess/observe_043.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 0 + 13] == 0).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 1).all()
    assert (state.observation[:, :, 14 * 4 + 13] == 0).all()
    state = step(state, jnp.int32(30))
    # "tests/assets/chess/observe_039.svg"
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep


def test_zobrist_hash():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    state = init(subkey)
    assert (state._x.hash_history[0] == INIT_ZOBRIST_HASH).all()
    assert (_zobrist_hash(state._x) == INIT_ZOBRIST_HASH).all()


def test_api():
    import pgx
    env = pgx.make("chess")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)