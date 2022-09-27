from pgx.shogi import init, _action_to_dlaction, _dlaction_to_action, ShogiAction, ShogiState, _move, _drop
import numpy as np


def make_test_board():
    board = np.zeros((29, 81), dtype=np.int32)
    board[0] = np.ones(81, dtype=np.int32)
    # 55に先手歩配置
    board[0][40] = 0
    board[1][40] = 1
    # 28に先手飛車配置
    board[0][16] = 0
    board[7][16] = 1
    # 19に先手香車配置
    board[0][8] = 0
    board[2][8] = 1
    # 36に後手桂馬配置
    board[0][23] = 0
    board[17][23] = 1
    # 59に後手馬配置
    board[0][44] = 0
    board[27][44] = 1
    return ShogiState(board=board)


def test_dlaction_to_action():
    s = make_test_board()
    # 54歩
    i = 39
    m = ShogiAction(False, 1, 39, 40, 0, False)
    assert m == _dlaction_to_action(i, s)
    # 68飛車
    i2 = 295
    m2 = ShogiAction(False, 7, 52, 16, 0, False)
    assert m2 == _dlaction_to_action(i2, s)
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert m3 == _dlaction_to_action(i3, s)
    s.turn = 1
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 7, True)
    assert m4 == _dlaction_to_action(i4, s)
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert m5 == _dlaction_to_action(i5, s)


def test_action_to_dlaction():
    i = 39
    m = ShogiAction(False, 1, 39, 40, 0, False)
    assert _action_to_dlaction(m, 0) == i
    # 68飛車
    i2 = 295
    m2 = ShogiAction(False, 7, 52, 16, 0, False)
    assert _action_to_dlaction(m2, 0) == i2
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert _action_to_dlaction(m3, 0) == i3
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 7, True)
    assert _action_to_dlaction(m4, 1) == i4
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert _action_to_dlaction(m5, 1) == i5


def test_move():
    i = init()
    #26歩
    action = 14
    b = _move(i, _dlaction_to_action(action, i))
    assert b.board[0][14] == 0
    assert b.board[1][15] == 0
    assert b.board[1][14] == 1
    assert b.board[0][15] == 1
    #76歩
    action = 59
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[0][59] == 0
    assert b.board[1][60] == 0
    assert b.board[1][59] == 1
    assert b.board[0][60] == 1
    # 33角成
    action = 992
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[15][20] == 0
    assert b.board[5][70] == 0
    assert b.board[13][20] == 1
    assert b.board[0][70] == 1
    assert b.hand[0] == 1
    b.turn = 1
    # 33桂馬（同桂）
    action = 749
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[13][20] == 0
    assert b.board[17][9] == 0
    assert b.board[17][20] == 1
    assert b.board[0][9] == 1
    assert b.hand[11] == 1


if __name__ == '__main__':
    test_dlaction_to_action()
    test_action_to_dlaction()
    test_move()
