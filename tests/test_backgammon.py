import jax
import jax.numpy as jnp
import numpy as np

from pgx.backgammon import (
    _calc_src,
    _calc_win_score,
    _is_action_legal,
    _is_all_on_homeboad,
    _is_open,
    _legal_action_mask,
    _move,
    _rear_distance,
    init,
    step,
)


def make_test_boad():
    board = jnp.zeros(28, dtype=np.int8)
    # 白
    board.at[19].set(-10)
    board.at[20].set(-1)
    board.at[21].set(-2)
    board.at[26].set(-2)
    # 黒
    board.at[3].set(2)
    board.at[4].set(1)
    board.at[10].set(5)
    board.at[22].set(3)
    board.at[25].set(4)
    return board


def make_test_state(
    board: np.ndarray,
    turn: np.int8,
    dice: np.ndarray,
    playable_dice: np.ndarray,
    played_dice_num: np.int8,
):
    state = init()
    state.board = board
    state.turn = turn
    state.dice = dice
    state.playable_dice = playable_dice
    state.played_dice_num = played_dice_num
    return state


def test_init():
    state = init()
    assert state.turn == -1 or state.turn == 1


def test_step():
    board = make_test_boad()
    state = make_test_state(
        board=board,
        turn=np.int8(1),
        dice=np.array([0, 1]),
        playable_dice=np.array([0, 1, -1, -1]),
        played_dice_num=np.int8(0),
    )
    # 黒がサイコロ2をplay
    state, _, _ = step(state=state, action=(22 + 2) * 6 + 1)
    assert (
        state.playable_dice - np.array([0, -1, -1, -1])
    ).sum() == 0  # playable diceが正しく更新されているか
    assert state.played_dice_num == 1  # played diceが増えているか.
    assert state.turn == 1  # turnが変わっていないか.
    assert (
        state.board[22] == 2 and state.board[20] == 1 and state.board[24] == -1
    )
    # 黒がサイコロ1をplay
    state, _, _ = step(state=state, action=(4 + 2) * 6 + 0)
    assert state.played_dice_num == 0
    assert state.turn == -1  # turnが変わっているか


def test_is_open():
    board = make_test_boad()
    # 白
    turn = np.int8(-1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 19)
    assert _is_open(board, turn, 4)
    assert not _is_open(board, turn, 10)
    # 黒
    turn = np.int8(1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 8)
    assert not _is_open(board, turn, 19)
    assert not _is_open(board, turn, 21)


def test_is_all_on_home_boad():
    board = make_test_boad()
    # 白
    turn = np.int8(-1)
    print(_is_all_on_homeboad(board, turn))
    assert _is_all_on_homeboad(board, turn)
    # 黒
    turn = np.int8(1)
    assert not _is_all_on_homeboad(board, turn)


def test_rear_distance():
    board = make_test_boad()
    turn = np.int8(-1)
    # 白
    assert _rear_distance(board, turn) == 5
    # 黒
    turn = np.int8(1)
    assert _rear_distance(board, turn) == 23


def test_calc_src():
    assert _calc_src(1, np.int8(-1)) == 24
    assert _calc_src(1, np.int8(1)) == 25
    assert _calc_src(2, np.int8(1)) == 0


def test_is_action_legal():
    board = make_test_boad()
    # 白
    turn = np.int8(-1)
    assert _is_action_legal(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_action_legal(board, turn, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _is_action_legal(
        board, turn, (19 + 2) * 6 + 2
    )  # 19 -> 22: 22に黒が複数ある.
    assert not _is_action_legal(
        board, turn, (22 + 2) * 6 + 2
    )  # 22 -> 25: 22に白がない
    assert _is_action_legal(board, turn, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, turn, (20 + 2) * 6 + 6
    )  # 後ろにまだ白があるためbear offできない.
    turn = np.int8(1)
    # 黒
    assert not _is_action_legal(
        board, turn, (3 + 2) * 6 + 0
    )  # 3->2: barにcheckerが残っているので動かせない.
    assert _is_action_legal(board, turn, (1) * 6 + 0)  # bar -> 23
    assert not _is_action_legal(board, turn, (1) * 6 + 2)  # bar -> 21


def test_move():
    # point to point
    board = make_test_boad()
    turn = np.int8(-1)
    board = _move(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert board[19] == -9 and board[21] == -3
    # point to off
    board = make_test_boad()
    turn = np.int8(-1)
    board = _move(board, turn, (19 + 2) * 6 + 5)  # 19->26
    assert board[19] == -9 and board[26] == -3
    # enter
    board = make_test_boad()
    turn = np.int8(1)
    board = _move(board, turn, (1) * 6 + 0)  # 25 -> 23
    assert board[25] == 3 and board[23] == 1
    # hit
    board = make_test_boad()
    turn = np.int8(1)
    board = _move(board, turn, (22 + 2) * 6 + 1)  # 22 -> 20
    assert board[22] == 2 and board[20] == 1 and board[24] == -1


def test_legal_action():
    board = make_test_boad()
    # 白
    turn = np.int8(-1)
    playable_dice = np.array([3, 2, -1, -1])
    expected_legal_action_mask = np.zeros(6 * 26 + 6)
    _ = [6 * 21 + 2, 6 * 22 + 2]
    expected_legal_action_mask[_] = 1
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    playable_dice = np.array([6, 6, 6, 6])
    expected_legal_action_mask = np.zeros(6 * 26 + 6)
    _ = [6 * 21 + 5]
    expected_legal_action_mask[_] = 1
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)

    # 黒
    turn = np.int8(1)
    playable_dice = np.array([4, 1, -1, -1])
    expected_legal_action_mask = np.zeros(6 * 26 + 6)
    _ = [6 * 1 + 1]
    expected_legal_action_mask[_] = 1
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    turn = np.int8(1)
    playable_dice = np.array([4, 4, 4, 4])
    expected_legal_action_mask = np.zeros(6 * 26 + 6)  # dance
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0


def test_calc_win_score():
    turn: np.int8 = np.int8(-1)
    # 白のバックギャモン勝ち
    back_gammon_board = np.zeros(28, dtype=np.int8)
    back_gammon_board[26] = -15
    back_gammon_board[1] = 15
    assert _calc_win_score(back_gammon_board, turn) == 3

    # 白のギャモン勝ち
    gammon_board = np.zeros(28, dtype=np.int8)
    gammon_board[26] = -15
    gammon_board[7] = 15
    assert _calc_win_score(gammon_board, turn) == 2

    # 白のシングル勝ち
    single_board = np.zeros(28, dtype=np.int8)
    single_board[26] = -15
    single_board[27] = 3
    single_board[3] = 12
    assert _calc_win_score(single_board, turn) == 1
