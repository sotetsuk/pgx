from pgx.chess import init, _move, ChessState, ChessAction, _piece_type, _board_status


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


if __name__ == '__main__':
    test_move()
