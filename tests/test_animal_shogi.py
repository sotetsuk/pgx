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


def test_turn_change():
    b = copy.deepcopy(INIT_BOARD)
    s = turn_change(b)
    assert s.turn == 1


def test_move():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiMove(first=6, final=5, piece=1, captured=6, is_promote=0)
    s = move(b, m)
    assert s.board[0][6] == 1
    assert s.board[1][6] == 0
    assert s.board[1][5] == 1
    assert s.board[6][5] == 0
    assert s.hand[0] == 1
    assert s.turn == 1
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiMove(first=1, final=0, piece=1, captured=8, is_promote=1)
    s2 = move(b2, m2)
    assert s2.board[0][1] == 1
    assert s2.board[1][1] == 0
    assert s2.board[5][0] == 1
    assert s2.board[8][0] == 0
    assert s2.hand[2] == 2
    assert s2.turn == 1


def test_drop():
    b = copy.deepcopy(TEST_BOARD)
    d = AnimalShogiDrop(piece=3, point=2)
    s = drop(b, d)
    assert s.hand[2] == 0
    assert s.board[3][2] == 1
    b2 = copy.deepcopy(TEST_BOARD)
    d2 = AnimalShogiDrop(piece=1, point=5)
    s2 = drop(b2, d2)
    assert s2.hand[0] == 0
    assert s2.board[1][5] == 1


def test_piece_type():
    assert piece_type(INIT_BOARD, 3) == 2
    assert piece_type(INIT_BOARD, 5) == 6
    assert piece_type(INIT_BOARD, 9) == 0


def test_legal_move():
    assert legal_moves(INIT_BOARD) == [[3, 2, 2, 0, 0], [6, 5, 1, 6, 0], [7, 2, 4, 0, 0], [7, 10, 4, 0, 0]]
    assert legal_moves(TEST_BOARD) == [[1, 0, 1, 8, 1], [3, 2, 5, 0, 0], [3, 7, 5, 7, 0], [6, 2, 2, 0, 0], [6, 5, 2, 0, 0], [6, 7, 2, 7, 0], [6, 10, 2, 0, 0], [11, 7, 4, 7, 0], [11, 10, 4, 0, 0]]


def test_legal_drop():
    assert legal_drop(TEST_BOARD) == [[2, 1], [5, 1], [9, 1], [10, 1], [2, 2], [5, 2], [8, 2], [9, 2], [10, 2], [2, 3], [5, 3], [8, 3], [9, 3], [10, 3]]


if __name__ == '__main__':
    test_turn_change()
    test_move()
    test_drop()
    test_piece_type()
    test_legal_move()
    test_legal_drop()
