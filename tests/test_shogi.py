import jax.numpy as jnp

from pgx.shogi import *
from pgx.shogi import _step, _step_move, _step_drop, _flip, _apply_effects, _legal_moves


# check visualization results by image preview plugins
def visualize(state, fname="tests/assets/shogi/xxx.svg"):
    from pgx.visualizer import Visualizer
    v = Visualizer(color_mode="dark")
    v.save_svg(state, fname)


def xy2i(x, y):
    """
    >>> xy2i(2, 6)  # 26歩
    14
    """
    i = (x - 1) * 9 + (y - 1)
    return i


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_step_move():
    s = init()

    # 26歩
    piece, from_, to = PAWN, xy2i(2, 7), xy2i(2, 6)
    assert s.piece_board[from_] == PAWN
    assert s.piece_board[to] == EMPTY
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_001.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 76歩
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)
    assert s.piece_board[from_] == PAWN
    assert s.piece_board[to] == EMPTY
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_002.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 33角成
    piece, from_, to = BISHOP , xy2i(8, 8), xy2i(3, 3)
    assert s.piece_board[from_] == BISHOP
    assert s.piece_board[to] == OPP_PAWN
    assert s.hand[0, PAWN] == 0
    assert (s.hand[0, PAWN:] == 0).all()
    a = Action.make_move(piece=piece, from_=from_, to=to, is_promotion=True)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/step_move_003.svg")
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == HORSE
    assert s.hand[0, PAWN] == 1
    assert (s.hand[0, :PAWN] == 0).all()


def test_step_drop():
    s = init()
    s = s.replace(hand=s.hand.at[:, :].set(1))  # type: ignore
    # 52飛車打ち
    piece, to = ROOK, xy2i(5, 2)
    a = Action.make_drop(piece, to)
    assert s.piece_board[to] == EMPTY
    assert s.hand[0, ROOK] == 1
    s = _step_drop(s, a)
    visualize(s, "tests/assets/shogi/step_drop_001.svg")
    assert s.piece_board[to] == ROOK
    assert s.hand[0, ROOK] == 0


def test_flip():
    s = init()
     # 26歩
    piece, from_, to = PAWN, xy2i(2, 7), xy2i(2, 6)
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    s = s.replace(hand=s.hand.at[0, :].set(1))  # type: ignore
    visualize(s, "tests/assets/shogi/flip_001.svg")
    assert s.piece_board[xy2i(2, 6)] == PAWN
    assert s.piece_board[xy2i(8, 4)] == EMPTY
    assert (s.hand[0] == 1).all()
    assert (s.hand[1] == 0).all()
    s = _flip(s)
    visualize(s, "tests/assets/shogi/flip_002.svg")
    assert s.piece_board[xy2i(2, 6)] == EMPTY
    assert s.piece_board[xy2i(8, 4)] == OPP_PAWN
    assert (s.hand[0] == 0).all()
    assert (s.hand[1] == 1).all()
    s = _flip(s)
    visualize(s, "tests/assets/shogi/flip_003.svg")
    assert s.piece_board[xy2i(2, 6)] == PAWN
    assert s.piece_board[xy2i(8, 4)] == EMPTY
    assert (s.hand[0] == 1).all()
    assert (s.hand[1] == 0).all()


def test_legal_moves():
    # promotion
    s = init()
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)  # 77歩
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_001.svg")
    # 33角は成れるが、44では成れない
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert legal_moves[xy2i(8, 8), xy2i(3, 3)]  # 33角
    assert legal_moves[xy2i(8, 8), xy2i(4, 4)]  # 44角
    assert promotion[xy2i(8, 8), xy2i(3, 3)] == 1  # 33角
    assert promotion[xy2i(8, 8), xy2i(4, 4)] == 0  # 44角

    # 11への歩は成らないと行けない
    s = s.replace(hand=s.hand.at[0].set(1))  # type: ignore
    a = Action.make_drop(piece=PAWN, to=xy2i(1, 2))
    s = _step_drop(s, a)
    visualize(s, "tests/assets/shogi/legal_moves_002.svg")
    effects = _apply_effects(s)
    legal_moves, promotion = _legal_moves(s, effects)
    assert legal_moves[xy2i(1, 2), xy2i(1, 1)]  # 33角
    assert promotion[xy2i(1, 2), xy2i(1, 1)] == 2  # 33角
