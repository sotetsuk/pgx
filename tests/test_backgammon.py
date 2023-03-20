import jax
import jax.numpy as jnp
from pgx.backgammon import (
    State,
    _flip_board,
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
    _distance_to_goal,
    _is_turn_end,
    _no_winning_step,
    _exists,
    Backgammon
)

seed = 1701
rng = jax.random.PRNGKey(seed)
env = Backgammon()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
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
_exists = jax.jit(_exists)


def make_test_boad():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    # 黒
    board = board.at[19].set(5)
    board = board.at[20].set(1)
    board = board.at[21].set(2)
    board = board.at[26].set(7)
    # 白
    board = board.at[3].set(-2)
    board = board.at[4].set(-1)
    board = board.at[10].set(-5)
    board = board.at[22].set(-3)
    board = board.at[25].set(-4)
    return board


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                       +  +  +  -
                       +     +  -
                       +        -
                       +
                       +
 
    -
    -
    -
    -                     -
    -                  -  -
11 10  9  8  7  6   5  4  3  2  1  0
Bar ----
Off +++++++
"""


def make_test_state(
    current_player: jnp.ndarray,
    rng: jax.random.KeyArray,
    board: jnp.ndarray,
    turn: jnp.ndarray,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    legal_action_mask=jnp.zeros(6 * 26 + 6, dtype=jnp.bool_),
):
    return State(
        current_player=current_player,
        rng=rng,
        board=board,
        turn=turn,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def test_flip_board():
    test_board = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    flipped_board = _flip_board(test_board)
    print(_flip_board, test_board)
    assert  (flipped_board == board).all()



def test_init():
    state = init(rng)
    assert state.turn == 0 or state.turn == 1


def test_init_roll():
    a = _roll_init_dice(rng)
    assert len(a) == 2
    assert a[0] != a[1]


def test_is_turn_end():
    state = init(rng)
    assert not _is_turn_end(state)

    # 白のdance
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    assert _is_turn_end(state)

    # playable diceがない場合
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(2),
    )
    assert _is_turn_end(state)


def test_change_turn():
    state = init(rng)
    _turn = state.turn
    state = _change_turn(state)
    assert state.turn == (_turn + 1) % 2

    test_board: jnp.ndarray = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    state = make_test_state(
        current_player=jnp.int8(0),
        rng=rng,
        board=test_board,
        turn=jnp.int8(0),
        dice=jnp.array([2, 2], dtype=jnp.int16),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(2),
    )
    state = _change_turn(state)
    print(state.board, board)
    assert state.turn == jnp.int8(1)  # ターンが変わっている.
    assert (state.board == board).all()  # 反転している.

def test_no_op():
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int16)
    )
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask=legal_action_mask,
    )
    state = step(state, 0)  # execute no-op action
    assert state.turn == jnp.int8(0)  # no-opの後はturnが変わっていることを確認.


def test_step():
    # 白
    board: jnp.ndarray = make_test_boad()
    board = _flip_board(board)  # 反転
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int16)
    )
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 1
    ].set(
        True
    )  # 24(bar)->1
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # legal_actionが正しいかtest

    # 白がサイコロ2をplay 24(bar)->1
    state = step(state=state, action=(1) * 6 + 1)
    assert (
        state.playable_dice == jnp.array([0, -1, -1, -1], dtype=jnp.int16)
    ).all()  # playable diceが正しく更新されているか
    assert state.played_dice_num == 1  # played diceが増えているか.
    assert state.turn == 1  # turnが変わっていないか.
    assert state.board.at[1].get() == 4 and state.board.at[24].get() == 3
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # legal_actionが正しく更新されているか
    # 白がサイコロ1をplay 24(off)->0
    state = step(state=state, action=(1) * 6 + 0)
    assert state.played_dice_num == 0
    assert state.turn == 0  # turn が黒に変わっているか.
    assert state.board.at[23].get() == -1 and state.board.at[25].get() == -2
    
    # 黒
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([4, 5, -1, -1], dtype=jnp.int16)
    )
    state = make_test_state(
        current_player=jnp.int8(0),
        rng=rng,
        board=board,
        turn=jnp.int8(0),
        dice=jnp.array([4, 5], dtype=jnp.int16),
        playable_dice=jnp.array([4, 5, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(
        True
    )  # 19 -> off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 4
    ].set(
        True
    )  # 19 -> off
    print(jnp.where(state.legal_action_mask==1)[0], jnp.where(expected_legal_action_mask==1)[0])
    assert (expected_legal_action_mask == state.legal_action_mask).all()


def test_observe():
    board: jnp.ndarray = make_test_boad()

    # current_playerが白で, playできるdiceが{1, 2}の場合
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([1, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int8(1)) == expected_obs).all()

    # current_playerが黒で, playできるdiceが(2)のみの場合
    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(-1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int8(1)) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int8(1),
        rng=rng,
        board=board,
        turn=jnp.int8(-1),
        dice=jnp.array([0, 1], dtype=jnp.int16),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int16),
        played_dice_num=jnp.int16(0),
    )
    expected_obs = jnp.concatenate(
        (1 * board, jnp.array([0, 0, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int8(0)) == expected_obs).all()


def test_is_open():
    board = make_test_boad()
    # 黒
    assert _is_open(board, 9)
    assert _is_open(board, 19)
    assert _is_open(board, 4)
    assert not _is_open(board, 10)
    # 白
    board = _flip_board(board)
    assert _is_open(board, 9)
    assert _is_open(board, 8)
    assert not _is_open(board, 2)
    assert not _is_open(board, 4)


def test_exists():
    board = make_test_boad()
    # 黒
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 4)
    # 白
    board = _flip_board(board)
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 2)


def test_is_all_on_home_boad():
    board: jnp.ndarray = make_test_boad()
    # 黒
    assert _is_all_on_home_board(board)
    # 白
    board = _flip_board(board)
    assert not _is_all_on_home_board(board)


def test_rear_distance():
    board = make_test_boad()
    turn = jnp.int8(-1)
    # 黒
    assert _rear_distance(board) == 5
    # 白
    board = _flip_board(board)
    assert _rear_distance(board) == 23


def test_distance_to_goal():
    board = make_test_boad()
    # 黒
    turn = jnp.int8(-1)
    src = 23
    assert _distance_to_goal(src) == 1
    src = 10
    assert _distance_to_goal(src) == 14
    # rear_istanceと同じはずのsrcでテスト
    assert _rear_distance(board) == _distance_to_goal(19)
    # 白もロジックは同様


def test_calc_src():
    assert _calc_src(1) == 24
    assert _calc_src(2) == 0


def test_calc_tgt():
    assert _calc_tgt(24, 1) == 0  # bar to board (die is transformed from 0~5 -> 1~ 6)
    assert _calc_tgt(6, 2) == 8  # board to board
    assert _calc_tgt(23, 6) == 26  # to off


def test_is_action_legal():
    board: jnp.ndarray = make_test_boad()
    # 黒
    assert _is_action_legal(board, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_action_legal(board, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _is_action_legal(
        board, (19 + 2) * 6 + 2
    )  # 19 -> 22: 22に白が複数ある.
    assert not _is_action_legal(
        board, (22 + 2) * 6 + 2
    )  # 22 -> 25: 22に黒がない
    assert _is_action_legal(board, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 5
    )  # 後ろにまだ黒があるためbear offできない.
    # 白
    board = _flip_board(board)
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 0
    )  # 20->21(反転後): barにcheckerが残っているので動かせない.
    assert _is_action_legal(board, (1) * 6 + 0)  # bar -> 1(反転後)
    assert not _is_action_legal(board, (1) * 6 + 2)  # bar -> 2(反転後)


def test_move():
    # point to point 黒
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 1)  # 19->21
    assert (
        board.at[19].get() == 4
        and board.at[21].get() == 3
        and board.at[25].get() == -4
    )
    # point to off 黒
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 5)  # 19->26
    assert (
        board.at[19].get() == 4
        and board.at[26].get() == 8
        and board.at[25].get() == -4
    )
    # enter 白
    board = make_test_boad()
    board = _flip_board(board)
    board = _move(board, (1) * 6 + 0)  # 25 -> 0
    assert (
        board.at[24].get() == 3
        and board.at[0].get() == 1
    )
    # hit 白
    board = make_test_boad()
    board = _flip_board(board)
    board = _move(board, (1 + 2) * 6 + 1)  # 1 -> 3
    print(board)
    assert (
        board.at[1].get() == 2
        and board.at[3].get() == 1
        and board.at[25].get() == -1
    )


def test_legal_action():
    board = make_test_boad()
    # 黒
    playable_dice = jnp.array([3, 2, -1, -1], dtype=jnp.int16)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 3
    ].set(
        True
    )  # 19->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 2
    ].set(
        True
    )  # 20->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 3
    ].set(
        True
    )  # 20->off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (21 + 2) + 2
    ].set(
        True
    )  # 21->off
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print(jnp.where(legal_action_mask != 0)[0])
    print(jnp.where(expected_legal_action_mask != 0)[0])
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([5, 5, 5, 5], dtype=jnp.int16)
    expected_legal_action_mask = jnp.zeros(6 * 26 + 6, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(True)
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    # 白
    board = _flip_board(board)
    playable_dice = jnp.array([4, 1, -1, -1], dtype=jnp.int16)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        True
    )
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([4, 4, 4, 4], dtype=jnp.int16)
    expected_legal_action_mask = jnp.zeros(
        6 * 26 + 6, dtype=jnp.bool_
    )  # dance
    expected_legal_action_mask = expected_legal_action_mask.at[0:6].set(
        True
    )  # no-opのみ
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()


def test_calc_win_score():
    # 黒のバックギャモン勝ち
    back_gammon_board = jnp.zeros(28, dtype=jnp.int16)
    back_gammon_board = back_gammon_board.at[26].set(15)
    back_gammon_board = back_gammon_board.at[23].set(-15)  # 黒のhome boardに残っている.
    print(_calc_win_score(back_gammon_board))
    assert _calc_win_score(back_gammon_board) == 3

    # 黒のギャモン勝ち
    gammon_board = jnp.zeros(28, dtype=jnp.int16)
    gammon_board = gammon_board.at[26].set(15)
    gammon_board = gammon_board.at[7].set(-15)
    assert _calc_win_score(gammon_board) == 2

    # 黒のシングル勝ち
    single_board = jnp.zeros(28, dtype=jnp.int16)
    single_board = single_board.at[26].set(15)
    single_board = single_board.at[27].set(-3)
    single_board = single_board.at[3].set(-12)
    assert _calc_win_score(single_board) == 1


def test_black_off():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int16)
    board = board.at[0].set(15)
    playable_dice = jnp.array([3, 2, -1, -1])
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print("3, 2", jnp.where(legal_action_mask != 0)[0])
    playable_dice = jnp.array([1, 1, -1, -1])
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print("1, 1", jnp.where(legal_action_mask != 0)[0])

def test_api():
    import pgx
    env = pgx.make("backgammon")
    pgx.api_test(env, 10)
