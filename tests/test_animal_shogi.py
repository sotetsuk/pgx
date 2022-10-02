from pgx._animal_shogi import (
    AnimalShogiState,
    AnimalShogiAction,
    INIT_BOARD,
    init,
    step,
    _another_color,
    _move,
    _drop,
    _piece_type,
    _effected_positions,
    _is_check,
    _create_piece_actions,
    _add_move_actions,
    _init_legal_actions,
    _legal_actions,
    _action_to_dlaction,
    _dlaction_to_action,
    _update_legal_move_actions,
    _update_legal_drop_actions
)
from pgx.animal_shogi import JaxAnimalShogiState
from pgx.animal_shogi import init as jax_init
from pgx.animal_shogi import step as jax_step
import numpy as np
import jax.numpy as jnp
import copy


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
    hand=np.array([1, 2, 1, 0, 0, 0]),
    is_check=True,
    checking_piece=np.array([
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    ])
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
    assert _another_color(b) == 1
    b2 = copy.deepcopy(TEST_BOARD2)
    assert _another_color(b2) == 0


def test_move():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(False, 1, 5, 6, 6, 0)
    s = _move(b, m)
    assert s.board[0][6] == 1
    assert s.board[1][6] == 0
    assert s.board[1][5] == 1
    assert s.board[6][5] == 0
    assert s.hand[0] == 1
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, 1)
    s2 = _move(b2, m2)
    assert s2.board[0][1] == 1
    assert s2.board[1][1] == 0
    assert s2.board[5][0] == 1
    assert s2.board[8][0] == 0
    assert s2.hand[2] == 2
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(False, 6, 7, 6, 2, 1)
    s3 = _move(b3, m3)
    assert s3.board[0][6] == 1
    assert s3.board[6][6] == 0
    assert s3.board[10][7] == 1
    assert s3.board[2][7] == 0
    assert s3.hand[4] == 2


def test_drop():
    b = copy.deepcopy(TEST_BOARD)
    d = AnimalShogiAction(True, 3, 2)
    s = _drop(b, d)
    assert s.hand[2] == 0
    assert s.board[3][2] == 1
    assert s.board[0][2] == 0
    b2 = copy.deepcopy(TEST_BOARD)
    d2 = AnimalShogiAction(True, 1, 5)
    s2 = _drop(b2, d2)
    assert s2.hand[0] == 0
    assert s2.board[1][5] == 1
    assert s2.board[0][5] == 0
    b3 = copy.deepcopy(TEST_BOARD2)
    d3 = AnimalShogiAction(True, 7, 2)
    s3 = _drop(b3, d3)
    assert s3.hand[4] == 0
    assert s3.board[7][2] == 1
    assert s3.board[0][2] == 0


def test_piece_type():
    assert _piece_type(INIT_BOARD, 3) == 2
    assert _piece_type(INIT_BOARD, 5) == 6
    assert _piece_type(INIT_BOARD, 9) == 0


def test_effected():
    assert np.all(_effected_positions(INIT_BOARD, 1) == np.array([1, 1, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0]))
    assert np.all(_effected_positions(TEST_BOARD, 0) == np.array([1, 0, 2, 0, 0, 1, 2, 3, 0, 0, 2, 0]))
    assert np.all(_effected_positions(TEST_BOARD2, 1) == np.array([3, 1, 2, 0, 0, 3, 1, 1, 2, 1, 1, 0]))


def test_is_check():
    assert not _is_check(INIT_BOARD)
    assert _is_check(TEST_BOARD)
    assert not _is_check(TEST_BOARD2)


def test_create_actions():
    array1 = _create_piece_actions(5, 4)
    array2 = np.zeros(180, dtype=np.int32)
    array2[4] = 1
    array2[20] = 1
    array2[24] = 1
    array2[45] = 1
    array2[49] = 1
    array2[66] = 1
    array2[82] = 1
    array2[86] = 1
    for i in range(180):
        assert array1[i] == array2[i]


def test_add_actions():
    array1 = np.zeros(180, dtype=np.int32)
    array2 = np.zeros(180, dtype=np.int32)
    array1 = _add_move_actions(5, 4, array1)
    array1 = _add_move_actions(6, 5, array1)
    array2[4] = 1
    array2[20] = 1
    array2[24] = 1
    array2[45] = 1
    array2[49] = 1
    array2[66] = 1
    array2[82] = 1
    array2[86] = 1
    array2[5] = 1
    array2[21] = 1
    array2[25] = 1
    array2[46] = 1
    array2[50] = 1
    array2[67] = 1
    for i in range(180):
        assert array1[i] == array2[i]


def test_create_legal_actions():
    c_board = _init_legal_actions(copy.deepcopy(INIT_BOARD))
    array1 = np.zeros(180, dtype=np.int32)
    array2 = np.zeros(180, dtype=np.int32)
    array1[2] = 1
    array1[5] = 1
    array1[6] = 1
    array1[22] = 1
    array1[26] = 1
    array1[30] = 1
    array1[43] = 1
    array1[47] = 1
    array1[51] = 1
    array2[9] = 1
    array2[5] = 1
    array2[6] = 1
    array2[13] = 1
    array2[29] = 1
    array2[33] = 1
    array2[36] = 1
    array2[40] = 1
    array2[56] = 1
    for i in range(180):
        assert array1[i] == c_board.legal_actions_black[i]
        assert array2[i] == c_board.legal_actions_white[i]


def test_legal_actions():
    b1 = _init_legal_actions(copy.deepcopy(INIT_BOARD))
    b2 = _init_legal_actions(copy.deepcopy(TEST_BOARD))
    b3 = _init_legal_actions(copy.deepcopy(TEST_BOARD2))
    n1 = _legal_actions(b1)
    n2 = _legal_actions(b2)
    n3 = _legal_actions(b3)
    array1 = np.zeros(180, dtype=np.int32)
    array2 = np.zeros(180, dtype=np.int32)
    array3 = np.zeros(180, dtype=np.int32)
    array1[2] = 1
    array1[5] = 1
    array1[26] = 1
    array1[22] = 1
    # 王手を受けている状態の挙動
    array2[43] = 1
    array2[67] = 1
    array2[55] = 1
    array3[2] = 1
    array3[7] = 1
    array3[56] = 1
    array3[33] = 1
    array3[14] = 1
    array3[92] = 1
    array3[34] = 1
    array3[103] = 1
    array3[146] = 1
    array3[152] = 1
    array3[153] = 1
    array3[154] = 1
    array3[158] = 1
    array3[164] = 1
    array3[165] = 1
    array3[166] = 1
    array3[170] = 1
    array3[176] = 1
    array3[177] = 1
    array3[178] = 1
    for i in range(180):
        assert n1[i] == array1[i]
        assert n2[i] == array2[i]
        assert n3[i] == array3[i]


def test_convert_action_to_int():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(False, 1, 5, 6, 6, False)
    i = _action_to_dlaction(m, b.turn)
    # 6の位置のヒヨコを5に移動させる
    assert i == 5
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, True)
    i2 = _action_to_dlaction(m2, b2.turn)
    # 1の位置のヒヨコを0に移動させる（成る）
    assert i2 == 96
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(False, 6, 7, 6, 2, True)
    i3 = _action_to_dlaction(m3, b3.turn)
    # 6の位置のヒヨコを7に移動させる（成る）
    # 後手番なので反転してdirectionは0(成っているので8)
    assert i3 == 103
    d = AnimalShogiAction(True, 3, 2)
    i4 = _action_to_dlaction(d, b2.turn)
    # 先手のゾウを2の位置に打つ
    # 先手のゾウを打つdirectionは11
    assert i4 == 134
    d2 = AnimalShogiAction(True, 1, 5)
    i5 = _action_to_dlaction(d2, b2.turn)
    assert i5 == 113
    d3 = AnimalShogiAction(True, 7, 2)
    i6 = _action_to_dlaction(d3, b3.turn)
    # 後手のキリンを2の位置に打つ(後手キリンを打つdirectionは13)
    assert i6 == 158


def test_convert_int_to_action():
    b = copy.deepcopy(INIT_BOARD)
    m = AnimalShogiAction(False, 1, 5, 6, 6, False)
    i = 5
    assert _dlaction_to_action(i, b) == m
    b2 = copy.deepcopy(TEST_BOARD)
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, True)
    i2 = 96
    assert _dlaction_to_action(i2, b2) == m2
    b3 = copy.deepcopy(TEST_BOARD2)
    m3 = AnimalShogiAction(False, 6, 7, 6, 2, True)
    i3 = 103
    assert _dlaction_to_action(i3, b3) == m3
    d = AnimalShogiAction(True, 3, 2)
    i4 = 134
    assert _dlaction_to_action(i4, b2) == d
    d2 = AnimalShogiAction(True, 1, 5)
    i5 = 113
    assert _dlaction_to_action(i5, b2) == d2
    d3 = AnimalShogiAction(True, 7, 2)
    i6 = 158
    assert _dlaction_to_action(i6, b3) == d3


def test_update_legal_actions_move():
    m = AnimalShogiAction(False, 1, 5, 6, 6, False)
    updated1 = _init_legal_actions(copy.deepcopy(INIT_BOARD))
    updated1 = _update_legal_move_actions(updated1, m)
    black1 = updated1.legal_actions_black
    white1 = updated1.legal_actions_white
    b1 = np.zeros(180, dtype=np.int32)
    w1 = np.zeros(180, dtype=np.int32)
    b1[2] = 1
    b1[4] = 1
    b1[6] = 1
    b1[22] = 1
    b1[26] = 1
    b1[30] = 1
    b1[43] = 1
    b1[47] = 1
    b1[51] = 1
    b1[100] = 1
    for i in range(12):
        b1[108 + i] = 1
    w1[9] = 1
    w1[5] = 1
    w1[13] = 1
    w1[29] = 1
    w1[33] = 1
    w1[36] = 1
    w1[40] = 1
    w1[56] = 1
    for i in range(180):
        assert black1[i] == b1[i]
        assert white1[i] == w1[i]


def test_update_legal_actions_drop():
    d = AnimalShogiAction(True, 7, 2)
    updated1 = _init_legal_actions(copy.deepcopy(TEST_BOARD2))
    updated1 = _update_legal_drop_actions(updated1, d)
    black1 = updated1.legal_actions_black
    white1 = updated1.legal_actions_white
    b1 = np.zeros(180, dtype=np.int32)
    w1 = np.zeros(180, dtype=np.int32)
    b1[2] = 1
    b1[6] = 1
    b1[10] = 1
    b1[18] = 1
    b1[30] = 1
    b1[43] = 1
    b1[47] = 1
    b1[51] = 1
    b1[55] = 1
    w1[2] = 1
    w1[3] = 1
    w1[5] = 1
    w1[7] = 1
    w1[13] = 1
    w1[14] = 1
    w1[29] = 1
    w1[30] = 1
    w1[33] = 1
    w1[34] = 1
    w1[36] = 1
    w1[53] = 1
    w1[54] = 1
    w1[56] = 1
    w1[60] = 1
    w1[61] = 1
    w1[72] = 1
    w1[92] = 1
    w1[103] = 1
    for i in range(12):
        w1[144 + i] = 1
        w1[168 + i] = 1
    for i in range(180):
        assert black1[i] == b1[i]
        assert white1[i] == w1[i]


def convert_jax_state(state: AnimalShogiState) -> JaxAnimalShogiState:
    turn = jnp.array([state.turn])
    board = jnp.zeros((11, 12), dtype=jnp.int32)
    for i in range(11):
        for j in range(12):
            if state.board[i][j] == 1:
                board = board.at[i, j].set(1)
    hand = jnp.zeros(6, dtype=jnp.int32)
    for i in range(6):
        hand = hand.at[i].set(state.hand[i])
    legal_actions_black = jnp.zeros(180, dtype=jnp.int32)
    legal_actions_white = jnp.zeros(180, dtype=jnp.int32)
    for i in range(180):
        if state.legal_actions_black[i] == 1:
            legal_actions_black = legal_actions_black.at[i].set(1)
        if state.legal_actions_white[i] == 1:
            legal_actions_white = legal_actions_white.at[i].set(1)
    is_check = jnp.zeros(1, dtype=jnp.int32)
    if state.is_check:
        is_check = is_check.at[0].set(1)
    checking_piece = jnp.zeros(12, dtype=jnp.int32)
    for i in range(12):
        if state.checking_piece[i] == 1:
            checking_piece = checking_piece.at[i].set(1)
    return JaxAnimalShogiState(
        turn=turn,
        board=board,
        hand=hand,
        legal_actions_black=legal_actions_black,
        legal_actions_white=legal_actions_white,
        is_check=is_check,
        checking_piece=checking_piece
    )  # type: ignore


#def test_jax_init():
#    np_init = init()
#    j_init = convert_jax_state(np_init)
#    j_init2 = jax_init()
#    assert (j_init.board == j_init2.board).all()
#    assert (j_init.legal_actions_black == j_init2.legal_actions_black).all()
#    assert (j_init.legal_actions_white == j_init2.legal_actions_white).all()


#def test_jax_step():
#    np_init = init()
#    j_init = convert_jax_state(np_init)
#    np_test = _init_legal_actions(copy.deepcopy(TEST_BOARD))
#    j_test = convert_jax_state(np_test)
#    np_test2 = _init_legal_actions(copy.deepcopy(TEST_BOARD2))
#    j_test2 = convert_jax_state(np_test2)
#    for i in range(180):
#        np_stepped = step(np_init, i)
#        jax_stepped = jax_step(j_init, i)
#        assert (jax_stepped[0].board == convert_jax_state(np_stepped[0]).board).all()
#        assert (jax_stepped[0].legal_actions_black == convert_jax_state(np_stepped[0]).legal_actions_black).all()
#        assert (jax_stepped[0].legal_actions_white == convert_jax_state(np_stepped[0]).legal_actions_white).all()
#        assert jax_stepped[1] == np_stepped[1]
#        assert jax_stepped[2] == np_stepped[2]
#        np_stepped_test = step(np_test, i)
#        jax_stepped_test = jax_step(j_test, i)
#        assert (jax_stepped_test[0].board == convert_jax_state(np_stepped_test[0]).board).all()
#        assert (jax_stepped_test[0].legal_actions_black == convert_jax_state(np_stepped_test[0]).legal_actions_black).all()
#        assert (jax_stepped_test[0].legal_actions_white == convert_jax_state(np_stepped_test[0]).legal_actions_white).all()
#        assert jax_stepped_test[1] == np_stepped_test[1]
#        assert jax_stepped_test[2] == np_stepped_test[2]
#        np_stepped_test2 = step(np_test2, i)
#        jax_stepped_test2 = jax_step(j_test2, i)
#        assert (jax_stepped_test2[0].board == convert_jax_state(np_stepped_test2[0]).board).all()
#        assert (jax_stepped_test2[0].legal_actions_black == convert_jax_state(np_stepped_test2[0]).legal_actions_black).all()
#        assert (jax_stepped_test2[0].legal_actions_white == convert_jax_state(np_stepped_test2[0]).legal_actions_white).all()
#        assert jax_stepped_test2[1] == np_stepped_test2[1]
#        assert jax_stepped_test2[2] == np_stepped_test2[2]


if __name__ == '__main__':
    x = 1
    #test_another_color()
    #test_move()
    #test_drop()
    #test_piece_type()
    #test_effected()
    #test_is_check()
    #test_convert_action_to_int()
    #test_convert_int_to_action()
    #test_create_actions()
    #test_add_actions()
    #test_create_legal_actions()
    #test_legal_actions()
    #test_update_legal_actions_move()
    #test_update_legal_actions_drop()
    #test_jax_init()
    #test_jax_step()
