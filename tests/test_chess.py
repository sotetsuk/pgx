from pgx.chess import init, _move, ChessState, ChessAction, _piece_type, _board_status, _make_board, _pawn_moves, \
    _knight_moves
import numpy as np


def test_move():
    s = init()
    m1 = ChessAction(1, 1, 3)
    s1 = _move(s, m1)
    assert _piece_type(s1, 3) == 1
    m2 = ChessAction(1, 33, 35)
    m3 = ChessAction(2, 48, 42)
    m4 = ChessAction(3, 40, 19)
    m5 = ChessAction(6, 32, 48)
    s2 = _move(s, m2)
    assert _piece_type(s2, 35) == 1
    s3 = _move(s2, m3)
    assert _piece_type(s3, 42) == 2
    s4 = _move(s3, m4)
    assert _piece_type(s4, 19) == 3
    s5 = _move(s4, m5)
    assert _piece_type(s5, 48) == 6
    assert _piece_type(s5, 40) == 4
    assert _piece_type(s5, 56) == 0
    m6 = ChessAction(7, 30, 28)
    m7 = ChessAction(1, 35, 29)
    s6 = _move(s2, m6)
    assert _piece_type(s6, 28) == 7
    s7 = _move(s6, m7)
    assert _piece_type(s7, 29) == 1
    assert _piece_type(s7, 28) == 0
    sx1 = _move(s, m6)
    sx2 = _move(sx1, m2)
    m8 = ChessAction(7, 28, 34)
    s8 = _move(sx2, m8)
    assert _piece_type(s8, 34) == 7
    assert _piece_type(s8, 35) == 0


def test_pawn_move():
    b = np.zeros(64, dtype=np.int32)
    b[28] = 1
    b[20] = 7
    pm = _pawn_moves(b, 28, 20, 0)
    for i in range(64):
        if i == 29 or i == 21:
            assert pm[i] == 1
        else:
            assert pm[i] == 0
    b1 = np.zeros(64, dtype=np.int32)
    b1[9] = 1
    b1[2] = 7
    b1[18] = 7
    pm1 = _pawn_moves(b1, 9, -1, 0)
    for i in range(64):
        if i == 2 or i == 18 or i == 10 or i == 11:
            assert pm1[i] == 1
        else:
            assert pm1[i] == 0
    b2 = np.zeros(64, dtype=np.int32)
    b2[6] = 7
    b2[5] = 1
    pm2 = _pawn_moves(b2, 6, -1, 1)
    for i in range(64):
        assert pm2[i] == 0


def test_knight_move():
    b = np.zeros(64, dtype=np.int32)
    b[28] = 2
    km = _knight_moves(b, 28, 0)
    for i in range(64):
        if i == 22 or i == 38 or i == 45 or i == 43 or i == 34 or i == 18 or i == 11 or i == 13:
            assert km[i] == 1
        else:
            assert km[i] == 0
    b1 = np.zeros(64, dtype=np.int32)
    b1[9] = 2
    km1 = _knight_moves(b1, 9, 0)
    for i in range(64):
        if i == 3 or i == 19 or i == 26 or i == 24:
            assert km1[i] == 1
        else:
            assert km1[i] == 0
    b1[3] = 1
    b1[26] = 3
    b1[19] = 8
    b1[24] = 9
    km2 = _knight_moves(b1, 9, 0)
    for i in range(64):
        if i == 19 or i == 24:
            assert km2[i] == 1
        else:
            assert km2[i] == 0


if __name__ == '__main__':
    test_move()
    test_pawn_move()
    test_knight_move()
