import numpy as np

from pgx.backgammon import _can_bear_off, _is_open, init


def make_test_boad():
    board = np.zeros(28, dtype=np.int8)
    # 白
    board[19] = -5
    board[20] = -5
    board[21] = -3
    board[26] = -2
    # 黒
    board[8] = 2
    board[10] = 5
    board[4] = 1
    board[11] = 3
    board[25] = 4
    return board


def test_init():
    state = init()
    assert state.turn[0] == 1 or state.turn[0] == 0


def test_is_open():
    board = make_test_boad()
    # 白
    turn = np.array([-1])
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 19)
    assert _is_open(board, turn, 19)
    assert _is_open(board, turn, 4)
    assert not _is_open(board, turn, 10)
    # 黒
    turn = np.array([1])
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 8)
    assert not _is_open(board, turn, 20)
    assert not _is_open(board, turn, 21)


def test_can_bear_off():
    board = make_test_boad()
    # 白
    turn = np.array([-1])
    assert _can_bear_off(board, turn)

    # 黒
    turn = np.array([1])
    assert not _can_bear_off(board, turn)
