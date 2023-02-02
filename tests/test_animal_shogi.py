from pgx.animal_shogi import (
    JaxAnimalShogiState as AnimalShogiState,
    JaxAnimalShogiAction as AnimalShogiAction,
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
import jax.numpy as np
import jax


init = jax.jit(init)
step = jax.jit(step)
_another_color = jax.jit(_another_color)
_move = jax.jit(_move)
_drop = jax.jit(_drop)
_piece_type = jax.jit(_piece_type)
_effected_positions= jax.jit(_effected_positions)
_is_check = jax.jit(_is_check)
_create_piece_actions = jax.jit(_create_piece_actions)
_add_move_actions = jax.jit(_add_move_actions)
_init_legal_actions = jax.jit(_init_legal_actions)
_legal_actions = jax.jit(_legal_actions)
_action_to_dlaction = jax.jit(_action_to_dlaction)
_dlaction_to_action = jax.jit(_dlaction_to_action)
_update_legal_move_actions = jax.jit(_update_legal_move_actions)
_update_legal_drop_actions = jax.jit(_update_legal_drop_actions)


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
    b = AnimalShogiState()
    assert _another_color(b) == 1
    b2 = TEST_BOARD2
    assert _another_color(b2) == 0


def test_move():
    b = AnimalShogiState()
    m = AnimalShogiAction(False, 1, 5, 6, 6, 0)
    s = _move(b, m)
    assert s.board[0][6] == 1
    assert s.board[1][6] == 0
    assert s.board[1][5] == 1
    assert s.board[6][5] == 0
    assert s.hand[0] == 1
    b2 = TEST_BOARD
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, 1)
    s2 = _move(b2, m2)
    assert s2.board[0][1] == 1
    assert s2.board[1][1] == 0
    assert s2.board[5][0] == 1
    assert s2.board[8][0] == 0
    assert s2.hand[2] == 2
    b3 = TEST_BOARD2
    m3 = AnimalShogiAction(False, 6, 7, 6, 2, 1)
    s3 = _move(b3, m3)
    assert s3.board[0][6] == 1
    assert s3.board[6][6] == 0
    assert s3.board[10][7] == 1
    assert s3.board[2][7] == 0
    assert s3.hand[4] == 2


def test_drop():
    b = TEST_BOARD
    d = AnimalShogiAction(True, 3, 2)
    s = _drop(b, d)
    assert s.hand[2] == 0
    assert s.board[3][2] == 1
    assert s.board[0][2] == 0
    b2 = TEST_BOARD
    d2 = AnimalShogiAction(True, 1, 5)
    s2 = _drop(b2, d2)
    assert s2.hand[0] == 0
    assert s2.board[1][5] == 1
    assert s2.board[0][5] == 0
    b3 = TEST_BOARD2
    d3 = AnimalShogiAction(True, 7, 2)
    s3 = _drop(b3, d3)
    assert s3.hand[4] == 0
    assert s3.board[7][2] == 1
    assert s3.board[0][2] == 0


def test_piece_type():
    assert _piece_type(AnimalShogiState(), 3) == 2
    assert _piece_type(AnimalShogiState(), 5) == 6
    assert _piece_type(AnimalShogiState(), 9) == 0


def test_effected():
    assert np.all(_effected_positions(AnimalShogiState(), 1) == np.array([1, 1, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0]))
    assert np.all(_effected_positions(TEST_BOARD, 0) == np.array([1, 0, 2, 0, 0, 1, 2, 3, 0, 0, 2, 0]))
    assert np.all(_effected_positions(TEST_BOARD2, 1) == np.array([3, 1, 2, 0, 0, 3, 1, 1, 2, 1, 1, 0]))


def test_is_check():
    assert not _is_check(AnimalShogiState())
    assert _is_check(TEST_BOARD)
    assert not _is_check(TEST_BOARD2)


def test_create_actions():
    array1 = _create_piece_actions(5, 4)
    array2 = np.zeros(180, dtype=np.bool_)
    array2 = array2.at[4].set(True)
    array2 = array2.at[20].set(True)
    array2 = array2.at[24].set(True)
    array2 = array2.at[45].set(True)
    array2 = array2.at[49].set(True)
    array2 = array2.at[66].set(True)
    array2 = array2.at[82].set(True)
    array2 = array2.at[86].set(True)
    assert (array1 == array2).all()


def test_add_actions():
    array1 = np.zeros(180, dtype=np.bool_)
    array2 = np.zeros(180, dtype=np.bool_)
    array1 = _add_move_actions(5, 4, array1)
    array1 = _add_move_actions(6, 5, array1)
    array2 = array2.at[4].set(True)
    array2 = array2.at[20].set(True)
    array2 = array2.at[24].set(True)
    array2 = array2.at[45].set(True)
    array2 = array2.at[49].set(True)
    array2 = array2.at[66].set(True)
    array2 = array2.at[82].set(True)
    array2 = array2.at[86].set(True)
    array2 = array2.at[5].set(True)
    array2 = array2.at[21].set(True)
    array2 = array2.at[25].set(True)
    array2 = array2.at[46].set(True)
    array2 = array2.at[50].set(True)
    array2 = array2.at[67].set(True)
    assert (array1 == array2).all()


def test_create_legal_actions():
    c_board = _init_legal_actions()
    array1 = np.zeros(180, dtype=np.int32)
    array2 = np.zeros(180, dtype=np.int32)
    array1 = array1.at[2].set(True)
    array1 = array1.at[5].set(True)
    array1 = array1.at[6].set(True)
    array1 = array1.at[22].set(True)
    array1 = array1.at[26].set(True)
    array1 = array1.at[30].set(True)
    array1 = array1.at[43].set(True)
    array1 = array1.at[47].set(True)
    array1 = array1.at[51].set(True)
    array2 = array2.at[9].set(True)
    array2 = array2.at[5].set(True)
    array2 = array2.at[6].set(True)
    array2 = array2.at[13].set(True)
    array2 = array2.at[29].set(True)
    array2 = array2.at[33].set(True)
    array2 = array2.at[36].set(True)
    array2 = array2.at[40].set(True)
    array2 = array2.at[56].set(True)
    assert (array1 == c_board.legal_actions_black).all()
    assert (array2 == c_board.legal_actions_white).all()


def test_legal_actions():
    b1 = AnimalShogiState()
    b2 = _init_legal_actions(TEST_BOARD)
    b3 = _init_legal_actions(TEST_BOARD2)
    n1 = _legal_actions(b1)
    n2 = _legal_actions(b2)
    n3 = _legal_actions(b3)
    array1 = np.zeros(180, dtype=np.int32)
    array2 = np.zeros(180, dtype=np.int32)
    array3 = np.zeros(180, dtype=np.int32)
    array1 = array1.at[2].set(True)
    array1 = array1.at[5].set(True)
    array1 = array1.at[26].set(True)
    array1 = array1.at[22].set(True)
    # 王手を受けている状態の挙動
    array2 = array2.at[43].set(True)
    array2 = array2.at[67].set(True)
    array2 = array2.at[55].set(True)
    array3 = array3.at[2].set(True)
    array3 = array3.at[7].set(True)
    array3 = array3.at[56].set(True)
    array3 = array3.at[33].set(True)
    array3 = array3.at[14].set(True)
    array3 = array3.at[92].set(True)
    array3 = array3.at[34].set(True)
    array3 = array3.at[103].set(True)
    array3 = array3.at[146].set(True)
    array3 = array3.at[152].set(True)
    array3 = array3.at[153].set(True)
    array3 = array3.at[154].set(True)
    array3 = array3.at[158].set(True)
    array3 = array3.at[164].set(True)
    array3 = array3.at[165].set(True)
    array3 = array3.at[166].set(True)
    array3 = array3.at[170].set(True)
    array3 = array3.at[176].set(True)
    array3 = array3.at[177].set(True)
    array3 = array3.at[178].set(True)
    assert (n1 == array1).all()
    assert (n2 == array2).all()
    assert (n3 == array3).all()


def test_convert_action_to_int():
    b = AnimalShogiState()
    m = AnimalShogiAction(False, 1, 5, 6, 6, False)
    i = _action_to_dlaction(m, b.turn)
    # 6の位置のヒヨコを5に移動させる
    assert i == 5
    b2 = TEST_BOARD
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, True)
    i2 = _action_to_dlaction(m2, b2.turn)
    # 1の位置のヒヨコを0に移動させる（成る）
    assert i2 == 96
    b3 = TEST_BOARD2
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
    b = AnimalShogiState()
    m = AnimalShogiAction(False, 1, 5, 6, 6, False)
    i = 5
    assert _dlaction_to_action(i, b) == m
    b2 = TEST_BOARD
    m2 = AnimalShogiAction(False, 1, 0, 1, 8, True)
    i2 = 96
    assert _dlaction_to_action(i2, b2) == m2
    b3 = TEST_BOARD2
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
    updated1 = _init_legal_actions(AnimalShogiState())
    updated1 = _update_legal_move_actions(updated1, m)
    black1 = updated1.legal_actions_black
    white1 = updated1.legal_actions_white
    b1 = np.zeros(180, dtype=np.bool_)
    w1 = np.zeros(180, dtype=np.bool_)
    b1 = b1.at[2].set(True)
    b1 = b1.at[4].set(True)
    b1 = b1.at[6].set(True)
    b1 = b1.at[22].set(True)
    b1 = b1.at[26].set(True)
    b1 = b1.at[30].set(True)
    b1 = b1.at[43].set(True)
    b1 = b1.at[47].set(True)
    b1 = b1.at[51].set(True)
    b1 = b1.at[100].set(True)
    for i in range(12):
        b1 = b1.at[108 + i].set(True)
    w1 = w1.at[9].set(True)
    w1 = w1.at[5].set(True)
    w1 = w1.at[13].set(True)
    w1 = w1.at[29].set(True)
    w1 = w1.at[33].set(True)
    w1 = w1.at[36].set(True)
    w1 = w1.at[40].set(True)
    w1 = w1.at[56].set(True)
    assert (black1 == b1).all()
    assert (white1 == w1).all()


def test_update_legal_actions_drop():
    d = AnimalShogiAction(True, 7, 2)
    updated1 = _init_legal_actions(TEST_BOARD2)
    updated1 = _update_legal_drop_actions(updated1, d)
    black1 = updated1.legal_actions_black
    white1 = updated1.legal_actions_white
    b1 = np.zeros(180, dtype=np.bool_)
    w1 = np.zeros(180, dtype=np.bool_)
    b1 = b1.at[2].set(True)
    b1 = b1.at[6].set(True)
    b1 = b1.at[10].set(True)
    b1 = b1.at[18].set(True)
    b1 = b1.at[30].set(True)
    b1 = b1.at[43].set(True)
    b1 = b1.at[47].set(True)
    b1 = b1.at[51].set(True)
    b1 = b1.at[55].set(True)
    w1 = w1.at[2].set(True)
    w1 = w1.at[3].set(True)
    w1 = w1.at[5].set(True)
    w1 = w1.at[7].set(True)
    w1 = w1.at[13].set(True)
    w1 = w1.at[14].set(True)
    w1 = w1.at[29].set(True)
    w1 = w1.at[30].set(True)
    w1 = w1.at[33].set(True)
    w1 = w1.at[34].set(True)
    w1 = w1.at[36].set(True)
    w1 = w1.at[53].set(True)
    w1 = w1.at[54].set(True)
    w1 = w1.at[56].set(True)
    w1 = w1.at[60].set(True)
    w1 = w1.at[61].set(True)
    w1 = w1.at[72].set(True)
    w1 = w1.at[92].set(True)
    w1 = w1.at[103].set(True)
    for i in range(12):
        w1 = w1.at[144 + i].set(True)
        w1 = w1.at[168 + i].set(True)
    assert (black1 == b1).all()
    assert (white1 == w1).all()


def test_step():
    s = init(jax.random.PRNGKey(0))
    # 詰みによる勝ち判定
    moves = [
        22, 13, 91, 40, 2, 100000
    ]
    for i in range(6):
        s, r, t = step(s, moves[i])
        if i == 5:
            assert r == 1
            assert t
        else:
            assert not t
    s = init(jax.random.PRNGKey(0))
    # トライルールによる勝ち判定(先手)
    moves = [
        26, 33, 1, 10, 0
    ]
    for i in range(5):
        s, r, t = step(s, moves[i])
        if i == 4:
            assert r == 1
            assert t
        else:
            assert not t
    s = init(jax.random.PRNGKey(0))
    # トライルールによる勝ち判定(後手)
    moves = [
        26, 33, 1, 10, 5, 11
    ]
    for i in range(6):
        s, r, t = step(s, moves[i])
        if i == 5:
            assert r == -1
            assert t
        else:
            assert not t
