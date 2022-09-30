from pgx.shogi import init, _action_to_dlaction, _dlaction_to_action, ShogiAction, ShogiState, _move, _drop, _piece_moves
import numpy as np


def make_test_board():
    board = np.zeros((29, 81), dtype=np.int32)
    board[0] = np.ones(81, dtype=np.int32)
    # 55に先手歩配置
    board[0][40] = 0
    board[1][40] = 1
    # 28に先手飛車配置
    board[0][16] = 0
    board[6][16] = 1
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
    m2 = ShogiAction(False, 6, 52, 16, 0, False)
    assert m2 == _dlaction_to_action(i2, s)
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert m3 == _dlaction_to_action(i3, s)
    # 43金（駒打ち）
    i4 = 81 * 26 + 29
    m4 = ShogiAction(True, 7, 29)
    assert m4 == _dlaction_to_action(i4, s)
    s.turn = 1
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 6, True)
    assert m4 == _dlaction_to_action(i4, s)
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert m5 == _dlaction_to_action(i5, s)
    # 98香車（駒打ち）
    i6 = 28 * 81 + 79
    m6 = ShogiAction(True, 16, 79)
    assert m6 == _dlaction_to_action(i6, s)


def test_action_to_dlaction():
    i = 39
    m = ShogiAction(False, 1, 39, 40, 0, False)
    assert _action_to_dlaction(m, 0) == i
    # 68飛車
    i2 = 295
    m2 = ShogiAction(False, 6, 52, 16, 0, False)
    assert _action_to_dlaction(m2, 0) == i2
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert _action_to_dlaction(m3, 0) == i3
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 6, True)
    assert _action_to_dlaction(m4, 1) == i4
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert _action_to_dlaction(m5, 1) == i5
    # 98香車（駒打ち）
    i6 = 28 * 81 + 79
    m6 = ShogiAction(True, 16, 79)
    assert _action_to_dlaction(m6, 1) == i6


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


def test_drop():
    i = init()
    i.hand = np.ones(14, dtype=np.int32)
    # 52飛車打ち
    action = 25 * 81 + 37
    b = _drop(i, _dlaction_to_action(action, i))
    assert b.board[0][37] == 0
    assert b.board[6][37] == 1
    assert b.hand[5] == 0
    b.turn = 1
    # 98香車打ち
    action = 28 * 81 + 79
    b = _drop(b, _dlaction_to_action(action, b))
    assert b.board[0][79] == 0
    assert b.board[16][79] == 1
    assert b.hand[8] == 0


def test_piece_moves():
    b1 = init()
    array1 = _piece_moves(b1, 6, 16)
    array2 = np.zeros(81, dtype=np.int32)
    for i in range(8):
        array2[9 * i + 7] = 1
    array2[15] = 1
    array2[16] = 0
    array2[17] = 1
    assert np.all(array1 == array2)
    array3 = _piece_moves(b1, 5, 70)
    array4 = np.zeros(81, dtype=np.int32)
    array4[60] = 1
    array4[62] = 1
    array4[78] = 1
    array4[80] = 1
    assert np.all(array3 == array4)
    # 76歩を指して角道を開けたときの挙動確認
    action = 59
    b1 = _move(b1, _dlaction_to_action(action, b1))
    new_array3 = _piece_moves(b1, 5, 70)
    for i in range(4):
        array4[20 + i * 10] = 1
    assert np.all(new_array3 == array4)
    b2 = make_test_board()
    array5 = _piece_moves(b2, 1, 40)
    array6 = np.zeros(81, dtype=np.int32)
    array6[39] = 1
    assert np.all(array5 == array6)
    array7 = _piece_moves(b2, 2, 8)
    array8 = np.zeros(81, dtype=np.int32)
    for i in range(8):
        array8[i] = 1
    assert np.all(array7 == array8)
    array9 = _piece_moves(b2, 27, 44)
    array10 = np.zeros(81, dtype=np.int32)
    for i in range(4):
        array10[34 - 10 * i] = 1
        array10[52 + 8 * i] = 1
    array10[43] = 1
    array10[35] = 1
    array10[53] = 1
    assert np.all(array9 == array10)


def test_init_legal_actions():
    s = init()
    array_b = np.zeros(2673, dtype=np.int32)
    array_w = np.zeros(2673, dtype=np.int32)
    # 歩のaction
    for i in range(9):
        array_b[5 + 9 * i] = 1
        array_w[3 + 9 * i] = 1
    # 香車のaction
    #for i in range(2):
    #    array_b[7 - i] = 1
    #    array_b[79 - i] = 1
    #    array_w[1 + i] = 1
    #    array_w[73 + i] = 1
    # 桂馬のaction
    for i in range(2):
        array_b[81 * 8 + 24 + 54 * i] = 1
        array_b[81 * 9 + 6 + 54 * i] = 1
        array_w[81 * 8 + 2 + 54 * i] = 1
        array_w[81 * 9 + 20 + 54 * i] = 1
    # 銀のaction
    for i in range(2):
        array_b[25 + 36 * i] = 1
        array_w[19 + 36 * i] = 1
        array_b[81 + 34 + 36 * i] = 1
        array_w[162 + 28 + 36 * i] = 1
        array_b[162 + 16 + 36 * i] = 1
        array_w[81 + 10 + 36 * i] = 1
    # 金のaction
    for i in range(2):
        array_b[34 + 18 * i] = 1
        array_w[28 + 18 * i] = 1
        array_b[81 + 43 + 18 * i] = 1
        array_w[162 + 37 + 18 * i] = 1
        array_b[162 + 25 + 18 * i] = 1
        array_w[81 + 19 + 18 * i] = 1
        array_b[243 + 44 + 18 * i] = 1
        array_w[243 + 18 + 18 * i] = 1
        array_b[324 + 26 + 18 * i] = 1
        array_w[324 + 36 + 18 * i] = 1
    # 玉のaction
    array_b[43] = 1
    array_w[37] = 1
    array_b[81 + 52] = 1
    array_w[81 + 28] = 1
    array_b[162 + 34] = 1
    array_w[162 + 46] = 1
    array_b[243 + 53] = 1
    array_w[243 + 27] = 1
    array_b[324 + 35] = 1
    array_w[324 + 45] = 1
    # 角のaction
    #array_b[81 + 78] = 1
    #array_b[162 + 60] = 1
    #array_b[81 * 6 + 80] = 1
    #array_b[81 * 7 + 62] = 1
    #array_w[81 + 2] = 1
    #array_w[162 + 20] = 1
    #array_w[81 * 6 + 0] = 1
    #array_w[81 * 7 + 18] = 1
    # 飛のaction
    #array_b[15] = 1
    #array_b[81 * 5 + 17] = 1
    #array_b[81 * 4 + 7] = 1
    #array_w[65] = 1
    #array_w[81 * 5 + 63] = 1
    #array_w[81 * 4 + 73] = 1
    #for i in range(6):
    #    array_b[81 * 3 + 25 + 9 * i] = 1
    #    array_w[81 * 3 + 55 - 9 * i] = 1
    assert np.all(array_b == s.legal_actions_black)
    assert np.all(array_w == s.legal_actions_white)


if __name__ == '__main__':
    test_dlaction_to_action()
    test_action_to_dlaction()
    test_move()
    test_drop()
    test_piece_moves()
    test_init_legal_actions()
