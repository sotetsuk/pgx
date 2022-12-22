from pgx.chess import init, _move, ChessState, ChessAction, _piece_type, _board_status, _make_board, _pawn_moves, \
    _knight_moves, _bishop_moves, _rook_moves, _queen_moves, _king_moves, _legal_actions, _create_actions, step, \
    _is_mate, _is_check, _is_legal_action, int_to_action, _pin
import numpy as np
import time


def test_move():
    s = init()
    m1 = ChessAction(1, 1, 3)
    s1 = _move(s, m1, 0)
    assert _piece_type(s1, 3) == 1
    m2 = ChessAction(1, 33, 35)
    m3 = ChessAction(2, 48, 42)
    m4 = ChessAction(3, 40, 19)
    m5 = ChessAction(6, 32, 48)
    s2 = _move(s, m2, 0)
    assert _piece_type(s2, 35) == 1
    s3 = _move(s2, m3, 0)
    assert _piece_type(s3, 42) == 2
    s4 = _move(s3, m4, 0)
    assert _piece_type(s4, 19) == 3
    s5 = _move(s4, m5, 2)
    assert _piece_type(s5, 48) == 6
    assert _piece_type(s5, 40) == 4
    assert _piece_type(s5, 56) == 0
    m6 = ChessAction(7, 30, 28)
    m7 = ChessAction(1, 35, 29)
    s6 = _move(s2, m6, 0)
    assert _piece_type(s6, 28) == 7
    s6.en_passant = 28
    s7 = _move(s6, m7, 0)
    assert _piece_type(s7, 29) == 1
    assert _piece_type(s7, 28) == 0
    sx1 = _move(s, m6, 0)
    sx2 = _move(sx1, m2, 0)
    sx2.en_passant = 35
    m8 = ChessAction(7, 28, 34)
    s8 = _move(sx2, m8, 0)
    assert _piece_type(s8, 34) == 7
    assert _piece_type(s8, 35) == 0


def test_pawn_move():
    b = np.zeros(64, dtype=np.int32)
    b[28] = 1
    b[20] = 7
    pm = _pawn_moves(b, 28, 0, 0)
    for i in range(64):
        if i == 29:
            assert pm[i] == 1
        else:
            assert pm[i] == 0
    b1 = np.zeros(64, dtype=np.int32)
    b1[9] = 1
    b1[2] = 7
    b1[18] = 7
    pm1 = _pawn_moves(b1, 9, 0, 0)
    for i in range(64):
        if i == 2 or i == 18 or i == 10 or i == 11:
            assert pm1[i] == 1
        else:
            assert pm1[i] == 0
    b2 = np.zeros(64, dtype=np.int32)
    b2[6] = 7
    b2[5] = 1
    pm2 = _pawn_moves(b2, 6, 1, 0)
    for i in range(64):
        assert pm2[i] == 0


def test_knight_move():
    b = np.zeros(64, dtype=np.int32)
    b[28] = 2
    km = _knight_moves(b, 28, 0, 0)
    for i in range(64):
        if i == 22 or i == 38 or i == 45 or i == 43 or i == 34 or i == 18 or i == 11 or i == 13:
            assert km[i] == 1
        else:
            assert km[i] == 0
    b1 = np.zeros(64, dtype=np.int32)
    b1[9] = 2
    km1 = _knight_moves(b1, 9, 0, 0)
    for i in range(64):
        if i == 3 or i == 19 or i == 26 or i == 24:
            assert km1[i] == 1
        else:
            assert km1[i] == 0
    b1[3] = 1
    b1[26] = 3
    b1[19] = 8
    b1[24] = 9
    km2 = _knight_moves(b1, 9, 0, 0)
    for i in range(64):
        if i == 19 or i == 24:
            assert km2[i] == 1
        else:
            assert km2[i] == 0


def test_bishop_move():
    b = np.zeros(64, dtype=np.int32)
    b[32] = 3
    bm = _bishop_moves(b, 32, 0, 0)
    for i in range(64):
        if i == 41 or i == 50 or i == 59 or i == 25 or i == 18 or i == 11 or i == 4:
            assert bm[i] == 1
        else:
            assert bm[i] == 0
    b[50] = 1
    b[18] = 7
    bm1 = _bishop_moves(b, 32, 0, 0)
    for i in range(64):
        if i == 41 or i == 25 or i == 18:
            assert bm1[i] == 1
        else:
            assert bm1[i] == 0


def test_rook_move():
    b = np.zeros(64, dtype=np.int32)
    b[32] = 4
    rm = _rook_moves(b, 32, 0, 0)
    for i in range(64):
        if i == 32:
            assert rm[i] == 0
        elif i == 33 or i == 34 or i == 35 or i == 36 or i == 37 or i == 38 or i == 39 or i % 8 == 0:
            assert rm[i] == 1
        else:
            assert rm[i] == 0
    b[35] = 1
    b[16] = 8
    b[40] = 5
    rm1 = _rook_moves(b, 32, 0, 0)
    for i in range(64):
        if i == 33 or i == 34 or i == 24 or i == 16:
            assert rm1[i] == 1
        else:
            assert rm1[i] == 0


def test_king_move():
    b = np.zeros(64, dtype=np.int32)
    b[36] = 6
    b[37] = 1
    b[44] = 7
    b[43] = 1
    b[29] = 1
    km = _king_moves(b, 36, 0)
    for i in range(64):
        if i == 45 or i == 44 or i == 35 or i == 27 or i == 28:
            assert km[i] == 1
        else:
            assert km[i] == 0


def test_pin():
    b = np.zeros(64, dtype=np.int32)
    b[28] = 6
    b[29] = 1
    b[30] = 11
    b[37] = 1
    b[55] = 11
    b[44] = 1
    b[60] = 11
    b[42] = 2
    b[49] = 11
    b[27] = 7
    b[26] = 11
    b[19] = 1
    b[10] = 1
    b[1] = 11
    b[12] = 9
    b[18] = 1
    b[14] = 5
    b[21] = 1
    s = ChessState(board=_make_board(b))
    pin = _pin(s, 28)
    for i in range(64):
        if i == 29:
            assert pin[i] == 1
        elif i == 37:
            assert pin[i] == 3
        elif i == 44:
            assert pin[i] == 2
        elif i == 42:
            assert pin[i] == 4
        else:
            assert pin[i] == 0


def test_legal_action():
    b = np.zeros(64, dtype=np.int32)
    b[25] = 1
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 7] == 1
    assert actions[25 + 64 * 8] == 1
    b[25] = 2
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 56]
    assert actions[25 + 64 * 57]
    assert actions[25 + 64 * 62]
    assert actions[25 + 64 * 63]
    b[25] = 3
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 34]
    assert actions[25 + 64 * 35]
    assert actions[25 + 64 * 36]
    assert actions[25 + 64 * 37]
    assert actions[25 + 64 * 38]
    assert actions[25 + 64 * 46]
    assert actions[25 + 64 * 47]
    assert actions[25 + 64 * 48]
    assert actions[25 + 64 * 49]
    b[25] = 4
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 6]
    assert actions[25 + 64 * 7]
    assert actions[25 + 64 * 8]
    assert actions[25 + 64 * 9]
    assert actions[25 + 64 * 10]
    assert actions[25 + 64 * 11]
    assert actions[25 + 64 * 12]
    assert actions[25 + 64 * 18]
    assert actions[25 + 64 * 19]
    assert actions[25 + 64 * 20]
    assert actions[25 + 64 * 21]
    assert actions[25 + 64 * 22]
    assert actions[25 + 64 * 23]
    assert actions[25 + 64 * 24]
    b[25] = 5
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 34]
    assert actions[25 + 64 * 35]
    assert actions[25 + 64 * 36]
    assert actions[25 + 64 * 37]
    assert actions[25 + 64 * 38]
    assert actions[25 + 64 * 46]
    assert actions[25 + 64 * 47]
    assert actions[25 + 64 * 48]
    assert actions[25 + 64 * 49]
    assert actions[25 + 64 * 6]
    assert actions[25 + 64 * 7]
    assert actions[25 + 64 * 8]
    assert actions[25 + 64 * 9]
    assert actions[25 + 64 * 10]
    assert actions[25 + 64 * 11]
    assert actions[25 + 64 * 12]
    assert actions[25 + 64 * 18]
    assert actions[25 + 64 * 19]
    assert actions[25 + 64 * 20]
    assert actions[25 + 64 * 21]
    assert actions[25 + 64 * 22]
    assert actions[25 + 64 * 23]
    assert actions[25 + 64 * 24]
    b[25] = 6
    s = ChessState(board=_make_board(b))
    actions = _legal_actions(s)
    assert actions[25 + 64 * 6]
    assert actions[25 + 64 * 7]
    assert actions[25 + 64 * 20]
    assert actions[25 + 64 * 21]
    assert actions[25 + 64 * 34]
    assert actions[25 + 64 * 35]
    assert actions[25 + 64 * 48]
    assert actions[25 + 64 * 49]
    s = init()
    actions = _legal_actions(s)
    for i in range(4608):
        if i == 8 + 64 * 56 or i == 8 + 64 * 57 or i == 48 + 64 * 56 or i == 48 + 64 * 57:
            assert actions[i] == 1
        elif 64 * 7 <= i <= 64 * 9:
            if (i % 64) % 8 == 1:
                assert actions[i] == 1
            else:
                assert actions[i] == 0
        else:
            assert actions[i] == 0
    s = init()
    m1 = ChessAction(1, 33, 35)
    m2 = ChessAction(2, 48, 42)
    m3 = ChessAction(3, 40, 19)
    m4 = ChessAction(6, 32, 48)
    s2 = _move(s, m1, 0)
    s3 = _move(s2, m2, 0)
    assert _legal_actions(s3)[32 + 64 * 22] == 0
    s4 = _move(s3, m3, 0)
    la = _legal_actions(s4)
    for i in range(4608):
        if i == 1 + 64 * 7 or i == 9 + 64 * 7 or i == 17 + 64 * 7 or i == 25 + 64 * 7 or i == 35 + 64 * 7 or i == 49 + 64 * 7 or i == 57 + 64 * 7 or i == 32 + 64 * 7:
            assert la[i] == 1
        elif i == 1 + 64 * 8 or i == 9 + 64 * 8 or i == 25 + 64 * 8 or i == 49 + 64 * 8 or i == 57 + 64 * 8:
            assert la[i] == 1
        elif i == 56 + 64 * 20 or i == 56 + 64 * 19 or i == 32 + 64 * 21 or i == 32 + 64 * 22:
            assert la[i] == 1
        elif i == 19 + 64 * 34 or i == 19 + 64 * 35 or i == 19 + 64 * 36 or i == 19 + 64 * 37 or i == 19 + 64 * 47 or i == 19 + 64 * 48 or i == 19 + 64 * 49 or i == 19 + 64 * 50 or i == 19 + 64 * 51 or i == 24 + 64 * 35:
            assert la[i] == 1
        elif i == 8 + 64 * 56 or i == 8 + 64 * 57 or i == 42 + 64 * 56 or i == 42 + 64 * 57 or i == 42 + 64 * 58 or i == 42 + 64 * 60 or i == 42 + 64 * 63:
            assert la[i] == 1
        else:
            if la[i] == 1:
                print(i)
            assert la[i] == 0
    bs = _board_status(s4)
    bs[26] = 11
    s5 = ChessState(board=_make_board(bs))
    assert _legal_actions(s5)[32 + 64 * 22] == 0
    m1 = ChessAction(1, 33, 35)
    m2 = ChessAction(1, 35, 36)
    m3 = ChessAction(7, 30, 28)
    s2 = _move(s, m1, 0)
    s3 = _move(s2, m2, 0)
    s4 = _move(s3, m3, 0)
    s4.en_passant = 28
    assert _legal_actions(s4)[36 + 64 * 48] == 1
    m4 = ChessAction(7, 46, 44)
    s5 = _move(s4, m4, 0)
    s5.en_passant = 44
    assert _legal_actions(s5)[36 + 64 * 35] == 1


def test_is_mate():
    bs = np.zeros(64, dtype=np.int32)
    bs[11] = 3
    bs[6] = 4
    bs[39] = 9
    bs[40] = 6
    bs[47] = 12
    bs[55] = 10
    state = ChessState(turn=1, board=_make_board(bs))
    assert _is_mate(state, _legal_actions(state))


def test_step():
    s = init()
    m = [33 + 64 * 8, 38 + 64 * 5, 41 + 64 * 8, 36 + 64 * 49, 40 + 64 * 46, 31 + 64 * 52, 32 + 64 * 21, 14 + 64 * 5,
         19 + 64 * 48, 55 + 64 * 61, 48 + 64 * 56, 59 + 64 * 8, 25 + 64 * 7, 45 + 64 * 62, 42 + 64 * 63, 61 + 64 * 34,
         59 + 64 * 58, 22 + 64 * 6, 49 + 64 * 8, 60 + 64 * 58, 56 + 64 * 20, 21 + 64 * 34, 57 + 64 * 8, 52 + 64 * 7,
         59 + 64 * 7, 53 + 64 * 6, 24 + 64 * 36, 45 + 64 * 57, 16 + 64 * 37, 52 + 64 * 48, 8 + 64 * 57, 47 + 64 * 32,
         18 + 64 * 57, 45 + 64 * 31, 43 + 64 * 47, 20 + 64 * 35, 44 + 64 * 58, 39 + 64 * 21, 42 + 64 * 10]
    for move in m:
        s, r, t = step(s, move)
        if move == 42 + 64 * 10:
            assert t
            assert r == 1
        else:
            assert not t
            assert r == 0
    m2 = [33 + 64 * 8, 38 + 64 * 5, 48 + 64 * 56, 15 + 64 * 60, 40 + 64 * 46, 47 + 64 * 32, 9 + 64 * 8, 20 + 64 * 34,
          17 + 64 * 7, 11 + 64 * 48, 25 + 64 * 8, 36 + 64 * 34, 32 + 64 * 22, 27 + 64 * 6, 24 + 64 * 47, 31 + 64 * 50,
          35 + 64 * 7, 45 + 64 * 21, 40 + 64 * 20, 55 + 64 * 59, 16 + 64 * 47, 14 + 64 * 5, 10 + 64 * 8, 7 + 64 * 21,
          12 + 64 * 34, 4 + 64 * 35, 8 + 64 * 63, 23 + 64 * 34, 25 + 64 * 57, 53 + 64 * 34, 19 + 64 * 49, 44 + 64 * 22,
          35 + 64 * 57, 54 + 64 * 34, 36 + 64 * 35, 63 + 64 * 20, 64 * 23, 60 + 64 * 33, 32 + 64 * 12, 21 + 64 * 63,
          3 + 64 * 37, 39 + 64 * 34, 26 + 64 * 36, 30 + 64 * 35, 44 + 64 * 47, 39 + 64 * 21, 2 + 64 * 38]
    s = init()
    i = 0
    for move in m2:
        i += 1
        s, r, t = step(s, move)
        if move == 2 + 64 * 38:
            assert t
            assert r == 1
        else:
            assert not t
            assert r == 0
    m3 = [25 + 64 * 8, 54 + 64 * 5, 27 + 64 * 7, 15 + 64 * 60, 17 + 64 * 8, 21 + 64 * 60, 33 + 64 * 7, 6 + 64 * 5,
          34 + 64 * 48, 7 + 64 * 6, 40 + 64 * 47, 55 + 64 * 61, 48 + 64 * 56, 45 + 64 * 61, 26 + 64 * 35, 63 + 64 * 20,
          35 + 64 * 37, 4 + 64 * 6, 62 + 64 * 48, 6 + 64 * 7, 42 + 64 * 57, 7 + 64 * 6, 55 + 64 * 34]
    s = init()
    i = 0
    for move in m3:
        i += 1
        s, r, t = step(s, move)
        if i == 23:
            assert t
            assert r == 1
        else:
            assert not t
            assert r == 0
    # promotion Knight
    m4 = [49 + 64 * 8, 14 + 64 * 6, 51 + 64 * 7, 55 + 64 * 60, 52 + 64 * 7, 63 + 64 * 20, 53 + 64 * 35, 61 + 64 * 59,
          62 + 64 * 68, 44 + 64 * 61, 41 + 64 * 48, 15 + 64 * 61, 55 + 64 * 60, 46 + 64 * 5, 40 + 64 * 36, 7 + 64 * 21,
          58 + 64 * 47, 54 + 64 * 6, 44 + 64 * 35]
    s = init()
    i = 0
    for move in m4:
        i += 1
        s, r, t = step(s, move)
        if i == 19:
            assert t
            assert r == 1
        else:
            assert not t
            assert r == 0
    # stalemate
    m5 = [17 + 64 * 8, 62 + 64 * 5, 57 + 64 * 8, 6 + 64 * 5, 24 + 64 * 46, 7 + 64 * 5, 3 + 64 * 7, 5 + 64 * 27,
          4 + 64 * 36, 46 + 64 * 6, 22 + 64 * 21, 39 + 64 * 49, 30 + 64 * 19, 31 + 64 * 2, 14 + 64 * 7, 26 + 64 * 38,
          15 + 64 * 21, 46 + 64 * 49, 23 + 64 * 50]
    s = init()
    i = 0
    for move in m5:
        i += 1
        s, r, t = step(s, move)
        if i == 19:
            assert t
            assert r == 0
        else:
            assert not t
            assert r == 0
    m6 = [9 + 64 * 8, 62 + 64 * 5, 11 + 64 * 7, 6 + 64 * 5, 12 + 64 * 48, 60 + 64 * 6, 49 + 64 * 8, 59 + 64 * 34]
    s = init()
    for move in m6:
        s, r, t = step(s, move)
        assert not t
        assert r == 0
    bs = _board_status(s)
    assert bs[4] == 0
    assert bs[51] == 0


if __name__ == '__main__':
    test_move()
    test_pawn_move()
    test_knight_move()
    test_bishop_move()
    test_rook_move()
    test_king_move()
    test_pin()
    test_legal_action()
    test_is_mate()
    test_step()
