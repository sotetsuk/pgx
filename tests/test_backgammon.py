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
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
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
    board: jnp.ndarray,
    turn: jnp.ndarray,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    legal_action_mask=jnp.zeros(6 * 26, dtype=jnp.bool_),
):
    return State(
        current_player=current_player,
        _board=board,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def test_flip_board():
    test_board = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
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
    assert state._turn == 0 or state._turn == 1


def test_init_roll():
    a = _roll_init_dice(rng)
    assert len(a) == 2
    assert a[0] != a[1]


def test_is_turn_end():
    state = init(rng)
    assert not _is_turn_end(state)

    # white dance
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    assert _is_turn_end(state)

    # No playable dice
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    assert _is_turn_end(state)


def test_change_turn():
    state = init(rng)
    _turn = state._turn
    state = _change_turn(state, jax.random.PRNGKey(0))
    assert state._turn == (_turn + 1) % 2

    test_board: jnp.ndarray = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
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
        current_player=jnp.int32(0),
        board=test_board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    state = _change_turn(state, jax.random.PRNGKey(0))
    print(state._board, board)
    assert state._turn == jnp.int32(1)  # Turn changed
    assert (state._board == board).all()  # Flipped.


def test_no_op():
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    state = step(state, 0, jax.random.PRNGKey(0))  # execute no-op action
    assert state._turn == jnp.int32(0)  # Turn changes after no-op.


def test_step():
    # 白
    board: jnp.ndarray = make_test_boad()
    board = _flip_board(board)  # Flipped
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
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
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # Test legal action

    # White plays die=2 24(bar)->1
    state = step(state=state, action=(1) * 6 + 1, key=jax.random.PRNGKey(0))
    assert (
            state._playable_dice == jnp.array([0, -1, -1, -1], dtype=jnp.int32)
    ).all()  # Is playable dice updated correctly?
    assert state._played_dice_num == 1  # played dice increased?
    assert state._turn == 1  # turn is not changed?
    assert state._board.at[1].get() == 4 and state._board.at[24].get() == 3
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # test legal action
    # White plays die=1 24(off)->0
    state = step(state=state, action=(1) * 6 + 0, key=jax.random.PRNGKey(0))
    assert state._played_dice_num == 0
    assert state._turn == 0  # turn changed to black?
    assert state._board.at[23].get() == -1 and state._board.at[25].get() == -2
    
    # black
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([4, 5, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([4, 5], dtype=jnp.int32),
        playable_dice=jnp.array([4, 5, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
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

    # current_player = white, playable_dice = (1, 2)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([1, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 4, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

    # current_player = black, playabl_dice = (2)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (1 * board, jnp.array([0, 0, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int32(0)) == expected_obs).all()


def test_is_open():
    board = make_test_boad()
    # Black
    assert _is_open(board, 9)
    assert _is_open(board, 19)
    assert _is_open(board, 4)
    assert not _is_open(board, 10)
    # White
    board = _flip_board(board)
    assert _is_open(board, 9)
    assert _is_open(board, 8)
    assert not _is_open(board, 2)
    assert not _is_open(board, 4)


def test_exists():
    board = make_test_boad()
    # Black
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 4)
    # White
    board = _flip_board(board)
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 2)


def test_is_all_on_home_boad():
    board: jnp.ndarray = make_test_boad()
    # Black
    assert _is_all_on_home_board(board)
    # White
    board = _flip_board(board)
    assert not _is_all_on_home_board(board)


def test_rear_distance():
    board = make_test_boad()
    turn = jnp.int32(-1)
    # Black
    assert _rear_distance(board) == 5
    # White
    board = _flip_board(board)
    assert _rear_distance(board) == 23


def test_distance_to_goal():
    board = make_test_boad()
    # Black
    turn = jnp.int32(-1)
    src = 23
    assert _distance_to_goal(src) == 1
    src = 10
    assert _distance_to_goal(src) == 14
    # Teat at the src where rear_distance is same
    assert _rear_distance(board) == _distance_to_goal(19)


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
    )  # 19 -> 22: Some whites on 22
    assert not _is_action_legal(
        board, (22 + 2) * 6 + 2
    )  # 22 -> 25: No black on 22 
    assert _is_action_legal(board, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 5
    )  # cannot bear off as some blacks behind
    # white
    board = _flip_board(board)
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 0
    )  # 20->21(after flipped): cannot move checkers as some left on bar
    assert _is_action_legal(board, (1) * 6 + 0)  # bar -> 1(after flipped)
    assert not _is_action_legal(board, (1) * 6 + 2)  # bar -> 2(after flipped)


def test_move():
    # point to point black
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 1)  # 19->21
    assert (
        board.at[19].get() == 4
        and board.at[21].get() == 3
        and board.at[25].get() == -4
    )
    # point to off black
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 5)  # 19->26
    assert (
        board.at[19].get() == 4
        and board.at[26].get() == 8
        and board.at[25].get() == -4
    )
    # enter white
    board = make_test_boad()
    board = _flip_board(board)
    board = _move(board, (1) * 6 + 0)  # 25 -> 0
    assert (
        board.at[24].get() == 3
        and board.at[0].get() == 1
    )
    # hit white
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
    # black
    playable_dice = jnp.array([3, 2, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
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

    playable_dice = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(6 * 26, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(True)
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    # white
    board = _flip_board(board)
    playable_dice = jnp.array([4, 1, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        True
    )
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([4, 4, 4, 4], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )  # dance
    expected_legal_action_mask = expected_legal_action_mask.at[0:6].set(
        True
    )  # only no-op
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()


def test_calc_win_score():
    # backgammon win by black
    back_gammon_board = jnp.zeros(28, dtype=jnp.int32)
    back_gammon_board = back_gammon_board.at[26].set(15)
    back_gammon_board = back_gammon_board.at[23].set(-15)  # black on home board
    print(_calc_win_score(back_gammon_board))
    assert _calc_win_score(back_gammon_board) == 3

    # gammon win by black
    gammon_board = jnp.zeros(28, dtype=jnp.int32)
    gammon_board = gammon_board.at[26].set(15)
    gammon_board = gammon_board.at[7].set(-15)
    assert _calc_win_score(gammon_board) == 2

    # single win by black
    single_board = jnp.zeros(28, dtype=jnp.int32)
    single_board = single_board.at[26].set(15)
    single_board = single_board.at[27].set(-3)
    single_board = single_board.at[3].set(-12)
    assert _calc_win_score(single_board) == 1


def test_black_off():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
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
    pgx.api_test(env, 3, use_key=True)
