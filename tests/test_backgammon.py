import jax
import jax.numpy as jnp
from pgx.visualizer import VisualizerConfig, Visualizer
from pgx.backgammon import (
    BackgammonState,
    _calc_src,
    _calc_tgt,
    _calc_win_score,
    _change_turn,
    _is_action_legal,
    _is_all_on_home_board,
    _is_open,
    _legal_action_mask,
    _move,
    _rear_distance,
    _roll_init_dice,
    init,
    observe,
    step,
    _distance_to_goal,
    _is_turn_end,
    _no_winning_step,
)

seed = 1701
rng = jax.random.PRNGKey(seed)
init = jax.jit(init)
step = jax.jit(step)
observe = jax.jit(observe)
_no_winning_step = jax.jit(_no_winning_step)
_calc_src = jax.jit(_calc_src)
_calc_tgt = jax.jit(_calc_tgt)
_calc_win_score = jax.jit(_calc_win_score)
_change_turn = jax.jit(_change_turn)
_is_action_legal = jax.jit(_is_action_legal)
_is_all_on_home_board = jax.jit(_is_all_on_home_board)
_is_open = jax.jit(_is_open)
_legal_action_mask = jax.jit(_legal_action_mask)
_move = jax.jit(_move)
_rear_distance = jax.jit(_rear_distance)



def make_test_boad():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int16)
    # 黒
    board = board.at[19].set(-5)
    board = board.at[20].set(-1)
    board = board.at[21].set(-2)
    board = board.at[26].set(-7)
    # 白
    board = board.at[3].set(2)
    board = board.at[4].set(1)
    board = board.at[10].set(5)
    board = board.at[22].set(3)
    board = board.at[25].set(4)
    return board

"""
黒: - 白: +
12 13 14 15 16 17  18 19 20 21 22 23
                       -  -  -  +
                       -     -  +
                       -        +
                       -
                       -
 
    +
    +
    +
    +                     +
    +                  +  +
11 10  9  8  7  6   5  4  3  2  1  0
Bar ++++
Off -------
"""

def make_test_state(
    curr_player: jnp.ndarray,
    rng: jax.random.KeyArray,
    board: jnp.ndarray,
    turn: jnp.ndarray,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    legal_action_mask = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
):
    return BackgammonState(
        curr_player=curr_player,
        rng=rng,
        board=board,
        turn=turn,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        legal_action_mask = legal_action_mask
    )


def test_init():
    _, state = init(rng)
    assert state.turn == -1 or state.turn == 1


def test_init_roll():
    a = _roll_init_dice(rng)
    assert len(a) == 2
    assert a[0] != a[1]


def test_is_turn_end():
    _, state = init(rng)
    assert not _is_turn_end(state)

    # 白のdance
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([2, 2, 2, 2], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    assert _is_turn_end(state)

    # playable diceがない場合
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(2),
    )
    assert _is_turn_end(state)


def test_change_turn():
    _, state = init(rng)
    _turn = state.turn
    state = _change_turn(state)
    assert state.turn == -1 * _turn

    # 白のdance
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([2, 2, 2, 2], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    state = _change_turn(state)
    assert state.turn == jnp.int16(-1)

    # playable diceがない場合
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(2),
    )
    state = _change_turn(state)
    assert state.turn == jnp.int16(-1)


def test_continual_pass():
    # 連続パスが可能かテスト
    # 白のdance
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(board, jnp.int16(1), jnp.array([0, 1, -1, -1], dtype=jnp.int16))
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([2, 2, 2, 2], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask = legal_action_mask,
    )
    _, state, _ = step(state, 6 * (1) + 0)  # actionによらずターンが変わる.
    assert state.turn == jnp.int16(-1)  # ターンが変わっていることを確認


def test_step():
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(board, jnp.int16(1), jnp.array([0, 1, -1, -1], dtype=jnp.int16))
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask = legal_action_mask
    )
    # legal_actionが正しいかtest
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (1) + 0].set(
        1
    ) # 25(off)->23
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (1) + 1].set(
        1
    ) # 25(off)->22
    assert (expected_legal_action_mask - state.legal_action_mask).sum() == 0 


    # 白がサイコロ2をplay 25(off)->22
    _, state, _ = step(state=state, action=(1) * 6 + 1)
    assert (
        state.playable_dice - jnp.array([0, -1, -1, -1], dtype=jnp.int16)
    ).sum() == 0  # playable diceが正しく更新されているか
    assert state.played_dice_num == 1  # played diceが増えているか.
    assert state.turn == 1  # turnが変わっていないか.
    assert (
        state.board.at[22].get() == 4
        and state.board.at[25].get() == 3
    )
    # legal_actionが正しく更新されているかテスト
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (1) + 0].set(
        1
    ) # 25(off)->23
    assert (expected_legal_action_mask - state.legal_action_mask).sum() == 0
    # 白がサイコロ1をplay 25(off)->23
    _, state, _ = step(state=state, action=(1) * 6 + 0)
    assert state.played_dice_num == 0
    assert (
        state.board.at[23].get() == 1
        and state.board.at[25].get() == 2
    )
    # legal_actionが正しいかtest
    legal_action_mask = _legal_action_mask(board, jnp.int16(-1), jnp.array([4, 5, -1, -1], dtype=jnp.int16))
    state = make_test_state(
        curr_player=jnp.int16(-1),
        rng=rng,
        board=board,
        turn=jnp.int16(-1),
        dice=jnp.array([4, 5], dtype=jnp.int16),
        playable_dice=jnp.array([4, 5, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask = legal_action_mask
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (19 + 2) + 5].set(
        1
    ) # 19 -> off
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (19 + 2) + 4].set(
        1
    ) # 19 -> off
    assert (expected_legal_action_mask - state.legal_action_mask).sum() == 0 



def test_observe():
    board: jnp.ndarray = make_test_boad()

    # curr_playerが白で, playできるdiceが{1, 2}の場合
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([1, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int16(1)) - expected_obs).sum() == 0

    # curr_playerが黒で, playできるdiceが(2)のみの場合
    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(-1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (-1 * board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int16(1)) - expected_obs).sum() == 0

    state = make_test_state(
        curr_player=jnp.int16(1),
        rng=rng,
        board=board,
        turn=jnp.int16(-1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (1 * board, jnp.array([0, 0, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int16(-1)) - expected_obs).sum() == 0


def test_is_open():
    board = make_test_boad()
    # 黒
    turn = jnp.int16(-1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 19)
    assert _is_open(board, turn, 4)
    assert not _is_open(board, turn, 10)
    # 白
    turn = jnp.int16(1)
    assert _is_open(board, turn, 9)
    assert _is_open(board, turn, 8)
    assert not _is_open(board, turn, 19)
    assert not _is_open(board, turn, 21)


def test_is_all_on_home_boad():
    board: jnp.ndarray = make_test_boad()
    # 黒
    turn: jnp.int16 = jnp.int16(-1)
    assert _is_all_on_home_board(board, turn)
    # 白
    turn = jnp.int16(1)
    assert not _is_all_on_home_board(board, turn)


def test_rear_distance():
    board = make_test_boad()
    turn = jnp.int16(-1)
    # 黒
    assert _rear_distance(board, turn) == 5
    # 白
    turn = jnp.int16(1)
    assert _rear_distance(board, turn) == 23


def test_distance_to_goal():
    board = make_test_boad()
    # 黒
    turn = jnp.int16(-1)
    src = 23
    assert _distance_to_goal(src, turn) == 1
    src = 10
    assert _distance_to_goal(src, turn) == 14
    # rear_istanceと同じはずのsrcでテスト
    assert _rear_distance(board, turn) == _distance_to_goal(19, turn)
    # 白
    turn = jnp.int16(1)
    src = 23
    assert _distance_to_goal(src, turn) == 24
    src = 2
    assert _distance_to_goal(src, turn) == 3
    # rear_istanceと同じはずのsrcでテスト
    assert _rear_distance(board, turn) == _distance_to_goal(22, turn)


def test_calc_src():
    assert _calc_src(1, jnp.int16(-1)) == 24
    assert _calc_src(1, jnp.int16(1)) == 25
    assert _calc_src(2, jnp.int16(1)) == 0


def test_calc_tgt():
    assert _calc_tgt(24, jnp.int16(-1), 1) == 0
    assert _calc_tgt(6, jnp.int16(1), 2) == 4
    assert _calc_tgt(2, jnp.int16(1), 6) == 27


def test_is_action_legal():
    board: jnp.ndarray = make_test_boad()
    # 黒
    turn = jnp.int16(-1)
    assert _is_action_legal(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_action_legal(board, turn, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _is_action_legal(
        board, turn, (19 + 2) * 6 + 2
    )  # 19 -> 22: 22に白が複数ある.
    assert not _is_action_legal(
        board, turn, (22 + 2) * 6 + 2
    )  # 22 -> 25: 22に黒がない
    assert _is_action_legal(board, turn, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, turn, (20 + 2) * 6 + 5
    )  # 後ろにまだ黒があるためbear offできない.
    turn = jnp.int16(1)
    # 白
    assert not _is_action_legal(
        board, turn, (3 + 2) * 6 + 0
    )  # 3->2: barにcheckerが残っているので動かせない.
    assert _is_action_legal(board, turn, (1) * 6 + 0)  # bar -> 23
    assert not _is_action_legal(board, turn, (1) * 6 + 2)  # bar -> 21


def test_move():
    # point to point
    board = make_test_boad()
    turn = jnp.int16(-1)
    board = _move(board, turn, (19 + 2) * 6 + 1)  # 19->21
    assert board.at[19].get() == -4 and board.at[21].get() == -3 and board.at[25].get() == 4
    # point to off
    board = make_test_boad()
    turn = jnp.int16(-1)
    board = _move(board, turn, (19 + 2) * 6 + 5)  # 19->26
    assert board.at[19].get() == -4 and board.at[26].get() == -8 and board.at[25].get() == 4
    # enter
    board = make_test_boad()
    turn = jnp.int16(1)
    board = _move(board, turn, (1) * 6 + 0)  # 25 -> 23
    assert board.at[25].get() == 3 and board.at[23].get() == 1 and board.at[24].get() == 0
    # hit
    board = make_test_boad()
    turn = jnp.int16(1)
    board = _move(board, turn, (22 + 2) * 6 + 1)  # 22 -> 20
    assert (
        board.at[22].get() == 2
        and board.at[20].get() == 1
        and board.at[24].get() == -1
    )


def test_legal_action():
    board = make_test_boad()
    # 黒
    turn = jnp.int16(-1)
    playable_dice = jnp.array([3, 2, -1, -1])
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (19 + 2) + 2].set(
        1
    ) # 19->21
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (20 + 2) + 2].set(
        1
    ) # 20->22
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    playable_dice = jnp.array([5, 5, 5, 5])
    expected_legal_action_mask = jnp.zeros(6 * 26 + 6)
    expected_legal_action_mask = expected_legal_action_mask.at[6 * (19 + 2) + 5].set(
        1
    )
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    print(jnp.where(legal_action_mask!=0)[0])
    print(jnp.where(expected_legal_action_mask!=0)[0])
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    # 白
    turn = jnp.int16(1)
    playable_dice = jnp.array([4, 1, -1, -1], dtype=jnp.int16)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.int16
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        1
    )
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0

    turn = jnp.int16(1)
    playable_dice = jnp.array([4, 4, 4, 4])
    expected_legal_action_mask = jnp.zeros(6 * 26 + 6)  # dance
    legal_action_mask = _legal_action_mask(board, turn, playable_dice)
    assert (expected_legal_action_mask - legal_action_mask).sum() == 0


def test_calc_win_score():
    turn: jnp.int16 = jnp.int16(-1)
    # 黒のバックギャモン勝ち
    back_gammon_board = jnp.zeros(28, dtype=jnp.int16)
    back_gammon_board = back_gammon_board.at[26].set(-15)
    back_gammon_board = back_gammon_board.at[1].set(15)
    assert _calc_win_score(back_gammon_board, turn) == 3

    # 黒のギャモン勝ち
    gammon_board = jnp.zeros(28, dtype=jnp.int16)
    gammon_board = gammon_board.at[26].set(-15)
    gammon_board = gammon_board.at[7].set(15)
    assert _calc_win_score(gammon_board, turn) == 2

    # 黒のシングル勝ち
    single_board = jnp.zeros(28, dtype=jnp.int16)
    single_board = single_board.at[26].set(-15)
    single_board = single_board.at[27].set(3)
    single_board = single_board.at[3].set(12)
    assert _calc_win_score(single_board, turn) == 1


def test_black_off():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int16)
    board = board.at[0].set(15)
    playable_dice = jnp.array([3, 2, -1, -1])
    legal_action_mask = _legal_action_mask(board, jnp.int16(1), playable_dice)
    print("3, 2", jnp.where(legal_action_mask!=0)[0])
    playable_dice = jnp.array([1, 1, -1, -1])
    legal_action_mask = _legal_action_mask(board, jnp.int16(1), playable_dice)
    print("1, 1", jnp.where(legal_action_mask!=0)[0])
