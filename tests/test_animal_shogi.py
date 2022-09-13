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
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    assert legal_moves(INIT_BOARD) == \
           [AnimalShogiAction(is_drop=0, piece=2, final=2, first=3, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=1, final=5, first=6, captured=6, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=4, final=2, first=7, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=4, final=10, first=7, captured=0, is_promote=0)]
    # 王手を受けている状態の挙動
    assert legal_moves(TEST_BOARD) == \
           [AnimalShogiAction(is_drop=0, piece=5, final=7, first=3, captured=7, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=2, final=7, first=6, captured=7, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=4, final=7, first=11, captured=7, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=4, final=10, first=11, captured=0, is_promote=0)]
    assert legal_moves(TEST_BOARD2) == \
           [AnimalShogiAction(is_drop=0, piece=10, final=2, first=1, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=9, final=8, first=4, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=9, final=9, first=4, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=8, final=2, first=5, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=8, final=8, first=5, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=8, final=10, first=5, captured=0, is_promote=0),
            AnimalShogiAction(is_drop=0, piece=6, final=7, first=6, captured=2, is_promote=1)]


def test_legal_drop():
    assert legal_drop(TEST_BOARD) == []
    assert legal_drop(TEST_BOARD2) == \
           [AnimalShogiAction(is_drop=1, piece=6, final=2), AnimalShogiAction(is_drop=True, piece=6, final=8),
            AnimalShogiAction(is_drop=1, piece=6, final=9), AnimalShogiAction(is_drop=True, piece=6, final=10),
            AnimalShogiAction(is_drop=1, piece=7, final=2), AnimalShogiAction(is_drop=True, piece=7, final=8),
            AnimalShogiAction(is_drop=1, piece=7, final=9), AnimalShogiAction(is_drop=True, piece=7, final=10),
            AnimalShogiAction(is_drop=1, piece=8, final=2), AnimalShogiAction(is_drop=True, piece=8, final=8),
            AnimalShogiAction(is_drop=1, piece=8, final=9), AnimalShogiAction(is_drop=True, piece=8, final=10)]


if __name__ == '__main__':
    test_another_color()
    test_move()
    test_drop()
    test_piece_type()
    test_effected()
    test_is_check()
    test_legal_move()
    test_legal_drop()
