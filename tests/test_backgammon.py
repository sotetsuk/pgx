import numpy as np

from pgx.backgammon import (
    _calc_src,
    _is_all_on_homeboad,
    _is_micro_action_legal,
    _is_open,
    _rear_distance,
    init,
)


def make_test_boad():
    board = np.zeros(28, dtype=np.int8)
    # 白
    board[19] = -5
    board[20] = -5
    board[21] = -3
    board[26] = -2
    # 黒
    board[3] = 2
    board[4] = 1
    board[10] = 5
    board[22] = 3
    board[25] = 4
    return board


def test_init():
    state = init()
    assert state.turn[0] == -1 or state.turn[0] == 1


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


def test_is_all_on_home_boad():
    board = make_test_boad()
    # 白
    turn = np.array([-1])
    print(_is_all_on_homeboad(board, turn))
    assert _is_all_on_homeboad(board, turn)
    # 黒
    turn = np.array([1])
    assert not _is_all_on_homeboad(board, turn)


def test_rear_distance():
    board = make_test_boad()
    turn = np.array([-1])
    # 白
    assert _rear_distance(board, turn) == 5
    # 黒
    turn = np.array([1])
    assert _rear_distance(board, turn) == 23


def test_calc_src():
    assert _calc_src(1, np.array([-1])) == 24
    assert _calc_src(1, np.array([1])) == 25
    assert _calc_src(2, np.array([1])) == 0


def test_is_micro_action_legal():
    board = make_test_boad()
    # 白
    turn = np.array([-1])
    assert _is_micro_action_legal(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_micro_action_legal(
        board, turn, (19 + 2) * 6 + 2
    )  # 19 -> 22
    assert not _is_micro_action_legal(
        board, turn, (19 + 2) * 6 + 2
    )  # 19 -> 22: 22に黒が複数ある.
    assert not _is_micro_action_legal(
        board, turn, (22 + 2) * 6 + 2
    )  # 22 -> 25: 22に白がない
    assert _is_micro_action_legal(board, turn, (19 + 2) * 6 + 6)  # bear off
    assert not _is_micro_action_legal(
        board, turn, (20 + 2) * 6 + 6
    )  # 後ろにまだ白があるためbear offできない.
    turn = np.array([1])
    # 黒
    assert not _is_micro_action_legal(
        board, turn, (3 + 2) * 6 + 0
    )  # 3->2: barにcheckerが残っているので動かせない.
    assert _is_micro_action_legal(board, turn, (1) * 6 + 0)  # bar -> 23
    assert not _is_micro_action_legal(board, turn, (1) * 6 + 2)  # bar -> 21
