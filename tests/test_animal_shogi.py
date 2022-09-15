from pgx.animal_shogi import *
import numpy as np
import copy


INIT_BOARD = AnimalShogiState(
    turn=0,
    board=np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    hand=np.array([0, 0, 0, 0, 0, 0])
)
TEST_BOARD = AnimalShogiState(
    turn=0,
    board=np.array([
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    hand=np.array([1, 2, 1, 0, 0, 0])
)
TEST_BOARD2 = AnimalShogiState(
    turn=1,
    board=np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    hand=np.array([0, 0, 0, 1, 1, 1])
)


def test_another_color():
    b = copy.deepcopy(INIT_BOARD)
    assert another_color(b) == 1
    b2 = copy.deepcopy(TEST_BOARD2)
    assert another_color(b2) == 0


def test_move():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(False, 1, 5, 6, 6, 0)
    s = move(b, m)
    assert s.board[0][6] == 1
    assert s.board[1][6] == 0
    assert s.board[1][5] == 1
    assert s.board[6][5] == 0
    assert s.hand[0] == 1
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, 1)
    s2 = move(b2, m2)
    assert s2.board[0][1] == 1
    assert s2.board[1][1] == 0
    assert s2.board[5][0] == 1
    assert s2.board[8][0] == 0
    assert s2.hand[2] == 2
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(False, 6, 7, 6, 2, 1)
    s3 = move(b3, m3)
    assert s3.board[0][6] == 1
    assert s3.board[6][6] == 0
    assert s3.board[10][7] == 1
    assert s3.board[2][7] == 0
    assert s3.hand[4] == 2


def test_drop():
    b = copy.deepcopy(TEST_BOARD)
    d = AnimalShogiAction(True, 3, 2)
    s = drop(b, d)
    assert s.hand[2] == 0
    assert s.board[3][2] == 1
    assert s.board[0][2] == 0
    b2 = copy.deepcopy(TEST_BOARD)
    d2 = AnimalShogiAction(True, 1, 5)
    s2 = drop(b2, d2)
    assert s2.hand[0] == 0
    assert s2.board[1][5] == 1
    assert s2.board[0][5] == 0
    b3 = copy.deepcopy(TEST_BOARD2)
    d3 = AnimalShogiAction(True, 7, 2)
    s3 = drop(b3, d3)
    assert s3.hand[4] == 0
    assert s3.board[7][2] == 1
    assert s3.board[0][2] == 0


def test_piece_type():
    assert piece_type(INIT_BOARD, 3) == 2
    assert piece_type(INIT_BOARD, 5) == 6
    assert piece_type(INIT_BOARD, 9) == 0


def test_effected():
    assert np.all(effected(INIT_BOARD, 1) == np.array([1, 1, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0]))
    assert np.all(effected(TEST_BOARD, 0) == np.array([1, 0, 2, 0, 0, 1, 2, 3, 0, 0, 2, 0]))
    assert np.all(effected(TEST_BOARD2, 1) == np.array([3, 1, 2, 0, 0, 3, 1, 1, 2, 1, 1, 0]))


def test_is_check():
    assert not is_check(INIT_BOARD)
    assert is_check(TEST_BOARD)
    assert not is_check(TEST_BOARD2)


def test_legal_move():
    array1 = legal_moves(INIT_BOARD, np.zeros(180, dtype=np.int32))
    assert array1[2] == 1
    assert array1[5] == 1
    assert array1[26] == 1
    assert array1[22] == 1
    # 王手を受けている状態の挙動
    array2 = legal_moves(TEST_BOARD, np.zeros(180, dtype=np.int32))
    assert array2[43] == 1
    assert array2[67] == 1
    assert array2[55] == 1
    # 自殺手
    assert array2[10] == 0
    # この手は王手を放置する手なので指せない
    assert array2[96] == 0
    array3 = legal_moves(TEST_BOARD2, np.zeros(180, dtype=np.int32))
    assert array3[2] == 1
    assert array3[56] == 1
    assert array3[33] == 1
    assert array3[14] == 1
    assert array3[92] == 1
    assert array3[34] == 1
    assert array3[103] == 1


def test_legal_drop():
    array = legal_drop(TEST_BOARD2, np.zeros(180, dtype=np.int32))
    assert array[146] == 1
    assert array[152] == 1
    assert array[153] == 1
    assert array[154] == 1
    assert array[158] == 1
    assert array[164] == 1
    assert array[165] == 1
    assert array[166] == 1
    assert array[170] == 1
    assert array[176] == 1
    assert array[177] == 1
    assert array[178] == 1


def test_convert_action_to_int():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(0, 1, 5, 6, 6, 0)
    i = action_to_int(m, b.turn)
    # 6の位置のヒヨコを5に移動させる
    assert i == 5
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(0, 1, 0, 1, 8, 1)
    i2 = action_to_int(m2, b2.turn)
    # 1の位置のヒヨコを0に移動させる（成る）
    assert i2 == 96
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(0, 6, 7, 6, 2, 1)
    i3 = action_to_int(m3, b3.turn)
    # 6の位置のヒヨコを7に移動させる（成る）
    # 後手番なので反転してdirectionは0(成っているので8)
    assert i3 == 103
    d = AnimalShogiAction(1, 3, 2)
    i4 = action_to_int(d, b2.turn)
    # 先手のゾウを2の位置に打つ
    # 先手のゾウを打つdirectionは11
    assert i4 == 134
    d2 = AnimalShogiAction(1, 1, 5)
    i5 = action_to_int(d2, b2.turn)
    assert i5 == 113
    d3 = AnimalShogiAction(1, 7, 2)
    i6 = action_to_int(d3, b3.turn)
    # 後手のキリンを2の位置に打つ(後手キリンを打つdirectionは13)
    assert i6 == 158


def test_convert_int_to_action():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(0, 1, 5, 6, 6, 0)
    i = 5
    assert int_to_action(i, b) == m
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(0, 1, 0, 1, 8, 1)
    i2 = 96
    assert int_to_action(i2, b2) == m2
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(0, 6, 7, 6, 2, 1)
    i3 = 103
    assert int_to_action(i3, b3) == m3
    d = AnimalShogiAction(1, 3, 2)
    i4 = 134
    assert int_to_action(i4, b2) == d
    d2 = AnimalShogiAction(1, 1, 5)
    i5 = 113
    assert int_to_action(i5, b2) == d2
    d3 = AnimalShogiAction(1, 7, 2)
    i6 = 158
    assert int_to_action(i6, b3) == d3


if __name__ == '__main__':
    test_another_color()
    test_move()
    test_drop()
    test_piece_type()
    test_effected()
    test_is_check()
    test_legal_move()
    test_legal_drop()
    test_convert_action_to_int()
    test_convert_int_to_action()
