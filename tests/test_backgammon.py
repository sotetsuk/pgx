import sys

import jax
import jax.numpy as jnp

sys.path.append("../")

from pgx.backgammon import (
    BackgammonState,
    _calc_src,
    _calc_tgt,
    _calc_win_score,
    _exists,
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
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    # 白
    board = board.at[19].set(-10)
    board = board.at[20].set(-1)
    board = board.at[21].set(-2)
    board = board.at[26].set(-2)
    # 黒
    board = board.at[3].set(2)
    board = board.at[4].set(1)
    board = board.at[10].set(5)
    board = board.at[22].set(3)
    board = board.at[25].set(4)
    return board


def make_test_state(
    board: jnp.ndarray,
    turn: jnp.int8,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.int8,
):
    state = init()
    state.board = board
    state.turn = turn
    state.dice = dice
    state.playable_dice = playable_dice
    state.played_dice_num = played_dice_num
    return state


def test_init():
    state: BackgammonState = init()
    assert state.turn == -1 or state.turn == 1


def test_step():
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([0, 1]),
        playable_dice=jnp.array([0, 1, -1, -1]),
        played_dice_num=jnp.int8(0),
    )
    # 黒がサイコロ2をplay
    state, _, _ = step(state=state, action=(22 + 2) * 6 + 1)
    assert (
        state.playable_dice - jnp.array([0, -1, -1, -1])
    ).sum() == 0  # playable diceが正しく更新されているか
    assert state.played_dice_num == 1  # played diceが増えているか.
    assert state.turn == 1  # turnが変わっていないか.
    assert (
        state.board.at[22].get() == 2
        and state.board.at[20].get() == 1
        and state.board.at[24].get() == -1
    )
    # 黒がサイコロ1をplay
    state, _, _ = step(state=state, action=(4 + 2) * 6 + 0)
    assert state.played_dice_num == 0
    assert state.turn == -1  # turnが変わっているか


def test_is_open():
    board = make_test_boad()
    # 白
    turn = jnp.int8(-1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 19)
    assert _is_open(board, turn, 4)
    assert not _is_open(board, turn, 10)
    # 黒
    turn = jnp.int8(1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 8)
    assert not _is_open(board, turn, 19)
    assert not _is_open(board, turn, 21)


def test_is_all_on_home_boad():
    board: jnp.ndarray = make_test_boad()
    # 白
    turn: jnp.int8 = jnp.int8(-1)
    assert _is_all_on_homeboad(board, turn)
    # 黒
    turn = jnp.int8(1)
    assert not _is_all_on_homeboad(board, turn)


def test_rear_distance():
    board = make_test_boad()
    turn = jnp.int8(-1)
    # 白
    assert _rear_distance(board, turn) == 5
    # 黒
    turn = jnp.int8(1)
    assert _rear_distance(board, turn) == 23


def test_calc_src():
    assert _calc_src(1, jnp.int8(-1)) == 24
    assert _calc_src(1, jnp.int8(1)) == 25
    assert _calc_src(2, jnp.int8(1)) == 0


def test_calc_tgt():
    assert _calc_tgt(24, jnp.int8(-1), 1) == 0
    assert _calc_tgt(6, jnp.int8(1), 2) == 4
    assert _calc_tgt(2, jnp.int8(1), 6) == 27


def test_is_action_legal():
    board: jnp.ndarray = make_test_boad()
    # 白
    turn = jnp.int8(-1)
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
    turn = jnp.int8(1)
    # 黒
    assert not _is_action_legal(
        board, turn, (3 + 2) * 6 + 0
    )  # 3->2: barにcheckerが残っているので動かせない.
    assert _is_action_legal(board, turn, (1) * 6 + 0)  # bar -> 23
    assert not _is_action_legal(board, turn, (1) * 6 + 2)  # bar -> 21


def test_move():
    # point to point
    board = make_test_boad()
    turn = jnp.int8(-1)
    board = _move(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert board.at[19].get() == -9 and board.at[21].get() == -3
    # point to off
    board = make_test_boad()
    turn = jnp.int8(-1)
    board = _move(board, turn, (19 + 2) * 6 + 5)  # 19->26
    assert board.at[19].get() == -9 and board.at[26].get() == -3
    # enter
    board = make_test_boad()
    turn = jnp.int8(1)
    board = _move(board, turn, (1) * 6 + 0)  # 25 -> 23
    assert board.at[25].get() == 3 and board.at[23].get() == 1
    # hit
    board = make_test_boad()
    turn = jnp.int8(1)
    board = _move(board, turn, (22 + 2) * 6 + 1)  # 22 -> 20
    assert (
        board.at[22].get() == 2
        and board.at[20].get() == 1
        and board.at[24].get() == -1
    )


def test_legal_action():
    board = make_test_boad()
    # 白
    turn = jnp.int8(-1)
    playable_dice = jnp.array([3, 2, -1, -1])
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int8
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 21 + 2].set(
        1
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 22 + 2].set(
        1
    )
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    print(_is_open(board, turn, 2), _is_open(board, turn, 3))
    print(_exists(board, turn, 0))
    print(legal_action_mask)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    playable_dice = jnp.array([6, 6, 6, 6])
    expected_legal_action_mask = jnp.zeros(6 * 26 + 6)
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 21 + 5].set(
        1
    )
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)

    # 黒
    turn = jnp.int8(1)
    playable_dice = jnp.array([4, 1, -1, -1])
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int8
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        1
    )
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    turn = jnp.int8(1)
    playable_dice = jnp.array([4, 4, 4, 4])
    expected_legal_action_mask = jnp.zeros(6 * 26 + 6)  # dance
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0


def test_calc_win_score():
    turn: jnp.int8 = jnp.int8(-1)
    # 白のバックギャモン勝ち
    back_gammon_board = jnp.zeros(28, dtype=jnp.int8)
    back_gammon_board = back_gammon_board.at[26].set(-15)
    back_gammon_board = back_gammon_board.at[1].set(15)
    assert _calc_win_score(back_gammon_board, turn) == 3

    # 白のギャモン勝ち
    gammon_board = jnp.zeros(28, dtype=jnp.int8)
    gammon_board = gammon_board.at[26].set(-15)
    gammon_board = gammon_board.at[7].set(15)
    assert _calc_win_score(gammon_board, turn) == 2

    # 白のシングル勝ち
    single_board = jnp.zeros(28, dtype=jnp.int8)
    single_board = single_board.at[26].set(-15)
    single_board = single_board.at[27].set(3)
    single_board = single_board.at[3].set(12)
    assert _calc_win_score(single_board, turn) == 1


if __name__ == "__main__":
    # test_init()
    test_is_open()
    test_calc_src()
    test_calc_win_score()
    test_is_all_on_home_boad()
    test_rear_distance()
    test_calc_tgt()
    test_move()
    test_is_action_legal()
    test_legal_action()
