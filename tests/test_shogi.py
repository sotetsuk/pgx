import jax.numpy as jnp

from pgx.shogi import *
from pgx.shogi import _step, _step_move, _step_drop


def xy2i(x, y, white=False):
    """
    >>> xy2i(2, 6)  # 26歩
    14
    """
    i = (x - 1) * 9 + (y - 1)
    if white:
        i = 80 - i
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
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 76歩
    piece, from_, to = PAWN, xy2i(7, 7), xy2i(7, 6)
    assert s.piece_board[from_] == PAWN
    assert s.piece_board[to] == EMPTY
    a = Action.make_move(piece=piece, from_=from_, to=to)  # type: ignore
    s = _step_move(s, a)
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == PAWN

    # 33角成
    piece, from_, to = BISHOP , xy2i(8, 8), xy2i(3, 3)
    assert s.piece_board[from_] == BISHOP
    assert s.piece_board[to] == OPP_PAWN
    a = Action.make_move(piece=piece, from_=from_, to=to, is_promotion=True)  # type: ignore
    s = _step_move(s, a)
    assert s.piece_board[from_] == EMPTY
    assert s.piece_board[to] == HORSE


def test_to_sfen():
    sfen = to_sfen(init())
    assert (sfen == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    pb = jnp.int8([15, -1, -1, 14, -1, 0, -1, -1, 1, -1, -1, -1, 1, 14, -1, -1, -1, 5, 8, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 6, 14, -1, 0, -1, 6, -1, -1, -1, -1, 14, -1, -1, 0, -1, 7, -1, -1, -1, -1, -1, -1, 2, 3, -1, -1, -1, 20, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, 16, 5, 0, -1, -1, -1, 8, -1, -1, -1, 1, -1, -1, -1])
    hand = jnp.int8([[1, 0, 0, 1, 0, 0, 0], [6, 0, 2, 2, 2, 0, 1]])
    s = State(turn=1, piece_board=pb, hand=hand)  # type: ignore
    sfen2 = to_sfen(s)
    assert (sfen2 == "6+P1l/+P8/2g2G3/4pp1Lp/1nk3Pp1/LR3P2P/1P1NP4/3S1G3/4K2RL w SP2bg2s2n6p 1")
