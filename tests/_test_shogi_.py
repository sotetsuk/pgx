from pgx._shogi import init, _action_to_dlaction, _dlaction_to_action, ShogiAction, ShogiState, _move, _drop, \
    _piece_moves, _is_check, _legal_actions, _add_drop_actions, _init_legal_actions, _update_legal_move_actions, \
    _update_legal_drop_actions, _is_double_pawn, _is_stuck, _board_status, step, _between, _pin, \
    _is_mate


import jax
import jax.numpy as jnp

init = jax.jit(init)
step = jax.jit(step)
_action_to_dlaction = jax.jit(_action_to_dlaction)
_dlaction_to_action = jax.jit(_dlaction_to_action)
_move = jax.jit(_move)
_drop = jax.jit(_drop)
_piece_moves = jax.jit(_piece_moves)
_is_check = jax.jit(_is_check)
_legal_actions = jax.jit(_legal_actions)
_add_drop_actions = jax.jit(_add_drop_actions)
_init_legal_actions = jax.jit(_init_legal_actions)
_update_legal_move_actions = jax.jit(_update_legal_move_actions)
_update_legal_drop_actions = jax.jit(_update_legal_drop_actions)
_is_double_pawn = jax.jit(_is_double_pawn)
_is_stuck = jax.jit(_is_stuck)
_board_status = jax.jit(_board_status)
_between = jax.jit(_between)
_pin = jax.jit(_pin)
_is_mate = jax.jit(_is_mate)


# 盤面の情報をStateに変換
def _make_board(bs: jnp.ndarray) -> ShogiState:
    board = jnp.zeros((29, 81), dtype=jnp.int32)
    for i in range(81):
        board = board.at[0, i].set(0)
        board = board.at[bs[i], i].set(1)
    return ShogiState(board=board)  # type: ignore


def make_test_board():
    board = jnp.zeros((29, 81), dtype=jnp.int32)
    board = board.at[0].set(jnp.ones(81, dtype=jnp.int32))
    # 55に先手歩配置
    board = board.at[0, 40].set(0)
    board = board.at[1, 40].set(1)
    # 28に先手飛車配置
    board = board.at[0, 16].set(0)
    board = board.at[6, 16].set(1)
    # 19に先手香車配置
    board = board.at[0, 8].set(0)
    board = board.at[2, 8].set(1)
    # 36に後手桂馬配置
    board = board.at[0, 23].set(0)
    board = board.at[17, 23].set(1)
    # 59に後手馬配置
    board = board.at[0, 44].set(0)
    board = board.at[27, 44].set(1)
    return ShogiState(board=board)


def test_dlaction_to_action():
    s = make_test_board()
    # 54歩
    i = 39
    m = ShogiAction(False, 1, 39, 40, 0, False)
    assert m == _dlaction_to_action(i, s)
    # 68飛車
    i2 = 295
    m2 = ShogiAction(False, 6, 52, 16, 0, False)
    assert m2 == _dlaction_to_action(i2, s)
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert m3 == _dlaction_to_action(i3, s)
    # 43金（駒打ち）
    i4 = 81 * 26 + 29
    m4 = ShogiAction(True, 7, 29)
    assert m4 == _dlaction_to_action(i4, s)
    s = s.replace(turn=1)  # type: ignore
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 6, True)
    assert m4 == _dlaction_to_action(i4, s)
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert m5 == _dlaction_to_action(i5, s)
    # 98香車（駒打ち）
    i6 = 28 * 81 + 79
    m6 = ShogiAction(True, 16, 79)
    assert m6 == _dlaction_to_action(i6, s)


def test_action_to_dlaction():
    i = 39
    m = ShogiAction(False, 1, 39, 40, 0, False)
    assert _action_to_dlaction(m, 0) == i
    # 68飛車
    i2 = 295
    m2 = ShogiAction(False, 6, 52, 16, 0, False)
    assert _action_to_dlaction(m2, 0) == i2
    # 12香車不成
    i3 = 1
    m3 = ShogiAction(False, 2, 1, 8, 0, False)
    assert _action_to_dlaction(m3, 0) == i3
    # 28桂馬成
    i4 = 1474
    m4 = ShogiAction(False, 17, 16, 23, 6, True)
    assert _action_to_dlaction(m4, 1) == i4
    # 95馬
    i5 = 643
    m5 = ShogiAction(False, 27, 76, 44, 0, False)
    assert _action_to_dlaction(m5, 1) == i5
    # 98香車（駒打ち）
    i6 = 28 * 81 + 79
    m6 = ShogiAction(True, 16, 79)
    assert _action_to_dlaction(m6, 1) == i6


def test_move():
    i = init()
    #26歩
    action = 14
    b = _move(i, _dlaction_to_action(action, i))
    assert b.board[0,14] == 0
    assert b.board[1,15] == 0
    assert b.board[1,14] == 1
    assert b.board[0,15] == 1
    #76歩
    action = 59
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[0,59] == 0
    assert b.board[1,60] == 0
    assert b.board[1,59] == 1
    assert b.board[0,60] == 1
    # 33角成
    action = 992
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[15,20] == 0
    assert b.board[5,70] == 0
    assert b.board[13,20] == 1
    assert b.board[0,70] == 1
    assert b.hand[0] == 1
    b = b.replace(turn=1)  # type: ignore
    # 33桂馬（同桂）
    action = 749
    b = _move(b, _dlaction_to_action(action, b))
    assert b.board[13,20] == 0
    assert b.board[17,9] == 0
    assert b.board[17,20] == 1
    assert b.board[0,9] == 1
    assert b.hand[11] == 1


def test_drop():
    i = init()
    i = i.replace(hand=jnp.ones(14, dtype=jnp.int32))  # type: ignore
    # 52飛車打ち
    action = 25 * 81 + 37
    b = _drop(i, _dlaction_to_action(action, i))
    assert b.board[0,37] == 0
    assert b.board[6,37] == 1
    assert b.hand[5] == 0
    b = b.replace(turn=1)  # type: ignore
    # 98香車打ち
    action = 28 * 81 + 79
    b = _drop(b, _dlaction_to_action(action, b))
    assert b.board[0,79] == 0
    assert b.board[16,79] == 1
    assert b.hand[8] == 0


def test_piece_moves():
    b1 = init()
    array1 = _piece_moves(_board_status(b1), 6, 16)
    array2 = jnp.zeros(81, dtype=jnp.int32)
    for i in range(8):
        array2 = array2.at[9 * i + 7].set(1)
    array2 = array2.at[15].set(1)
    array2 = array2.at[16].set(0)
    array2 = array2.at[17].set(1)
    assert jnp.all(array1 == array2)
    array3 = _piece_moves(_board_status(b1), 5, 70)
    array4 = jnp.zeros(81, dtype=jnp.int32)
    array4 = array4.at[60].set(1)
    array4 = array4.at[62].set(1)
    array4 = array4.at[78].set(1)
    array4 = array4.at[80].set(1)
    assert jnp.all(array3 == array4)
    # 76歩を指して角道を開けたときの挙動確認
    action = 59
    b1 = _move(b1, _dlaction_to_action(action, b1))
    new_array3 = _piece_moves(_board_status(b1), 5, 70)
    for i in range(4):
        array4 = array4.at[20 + i * 10].set(1)
    assert jnp.all(new_array3 == array4)
    b2 = make_test_board()
    array5 = _piece_moves(_board_status(b2), 1, 40)
    array6 = jnp.zeros(81, dtype=jnp.int32)
    array6 = array6.at[39].set(1)
    assert jnp.all(array5 == array6)
    array7 = _piece_moves(_board_status(b2), 2, 8)
    array8 = jnp.zeros(81, dtype=jnp.int32)
    for i in range(8):
        array8 = array8.at[i].set(1)
    assert jnp.all(array7 == array8)
    array9 = _piece_moves(_board_status(b2), 27, 44)
    array10 = jnp.zeros(81, dtype=jnp.int32)
    for i in range(4):
        array10 = array10.at[34 - 10 * i].set(1)
        array10 = array10.at[52 + 8 * i].set(1)
    array10 = array10.at[43].set(1)
    array10 = array10.at[35].set(1)
    array10 = array10.at[53].set(1)
    assert jnp.all(array9 == array10)


def test_init_legal_actions():
    s = init()
    array_b = jnp.zeros(2754, dtype=jnp.int32)
    array_w = jnp.zeros(2754, dtype=jnp.int32)
    # 歩のaction
    for i in range(9):
        array_b = array_b.at[5 + 9 * i].set(1)
        array_w = array_w.at[3 + 9 * i].set(1)
    # 香車のaction
    #for i in range(2):
    #    array_b[7 - i] = 1
    #    array_b[79 - i] = 1
    #    array_w[1 + i] = 1
    #    array_w[73 + i] = 1
    # 桂馬のaction
    for i in range(2):
        array_b = array_b.at[81 * 8 + 24 + 54 * i].set(1)
        array_b = array_b.at[81 * 9 + 6 + 54 * i].set(1)
        array_w = array_w.at[81 * 8 + 2 + 54 * i].set(1)
        array_w = array_w.at[81 * 9 + 20 + 54 * i].set(1)
    # 銀のaction
    for i in range(2):
        array_b = array_b.at[25 + 36 * i].set(1)
        array_w = array_w.at[19 + 36 * i].set(1)
        array_b = array_b.at[81 + 34 + 36 * i].set(1)
        array_w = array_w.at[162 + 28 + 36 * i].set(1)
        array_b = array_b.at[162 + 16 + 36 * i].set(1)
        array_w = array_w.at[81 + 10 + 36 * i].set(1)
    # 金のaction
    for i in range(2):
        array_b = array_b.at[34 + 18 * i].set(1)
        array_w = array_w.at[28 + 18 * i].set(1)
        array_b = array_b.at[81 + 43 + 18 * i].set(1)
        array_w = array_w.at[162 + 37 + 18 * i].set(1)
        array_b = array_b.at[162 + 25 + 18 * i].set(1)
        array_w = array_w.at[81 + 19 + 18 * i].set(1)
        array_b = array_b.at[243 + 44 + 18 * i].set(1)
        array_w = array_w.at[243 + 18 + 18 * i].set(1)
        array_b = array_b.at[324 + 26 + 18 * i].set(1)
        array_w = array_w.at[324 + 36 + 18 * i].set(1)
    # 玉のaction
    array_b = array_b.at[43].set(1)
    array_w = array_w.at[37].set(1)
    array_b = array_b.at[81 + 52].set(1)
    array_w = array_w.at[81 + 28].set(1)
    array_b = array_b.at[162 + 34].set(1)
    array_w = array_w.at[162 + 46].set(1)
    array_b = array_b.at[243 + 53].set(1)
    array_w = array_w.at[243 + 27].set(1)
    array_b = array_b.at[324 + 35].set(1)
    array_w = array_w.at[324 + 45].set(1)
    # 角のaction
    #array_b[81 + 78] = 1
    #array_b[162 + 60] = 1
    #array_b[81 * 6 + 80] = 1
    #array_b[81 * 7 + 62] = 1
    #array_w[81 + 2] = 1
    #array_w[162 + 20] = 1
    #array_w[81 * 6 + 0] = 1
    #array_w[81 * 7 + 18] = 1
    # 飛のaction
    #array_b[15] = 1
    #array_b[81 * 5 + 17] = 1
    #array_b[81 * 4 + 7] = 1
    #array_w[65] = 1
    #array_w[81 * 5 + 63] = 1
    #array_w[81 * 4 + 73] = 1
    #for i in range(6):
    #    array_b[81 * 3 + 25 + 9 * i] = 1
    #    array_w[81 * 3 + 55 - 9 * i] = 1
    assert jnp.all(array_b == s.legal_actions_black)
    assert jnp.all(array_w == s.legal_actions_white)


def test_is_check():
    board = jnp.zeros((29, 81), dtype=jnp.int32)
    board = board.at[0].set(jnp.ones(81, dtype=jnp.int32))
    board = board.at[8, 44].set(1)
    board = board.at[0, 44].set(0)
    board = board.at[16, 36].set(1)
    board = board.at[0, 36].set(0)
    board = board.at[17, 33].set(1)
    board = board.at[0, 33].set(0)
    board = board.at[18, 52].set(1)
    board = board.at[0, 52].set(0)
    board = board.at[21, 53].set(1)
    board = board.at[0, 53].set(0)
    board = board.at[27, 4].set(1)
    board = board.at[0, 4].set(0)
    board = board.at[28, 8].set(1)
    board = board.at[0, 8].set(0)
    s = ShogiState(board=board)
    assert _is_check(s)


def test_legal_actions():
    state = init()
    actions1 = _legal_actions(state)
    actions2 = jnp.zeros(2754, dtype=jnp.int32)
    # 歩のaction
    for i in range(9):
        actions2 = actions2.at[5 + 9 * i].set(1)
    # 香車のaction
    actions2 = actions2.at[7].set(1)
    actions2 = actions2.at[79].set(1)
    # 桂馬のaction
    # 銀のaction
    actions2 = actions2.at[25].set(1)
    actions2 = actions2.at[61].set(1)
    actions2 = actions2.at[81 + 34].set(1)
    actions2 = actions2.at[162 + 52].set(1)
    # 金のaction
    for i in range(2):
        actions2 = actions2.at[34 + 18 * i].set(1)
        actions2 = actions2.at[81 + 43 + 18 * i].set(1)
        actions2 = actions2.at[162 + 25 + 18 * i].set(1)
    # 玉のaction
    actions2 = actions2.at[43].set(1)
    actions2 = actions2.at[81 + 52].set(1)
    actions2 = actions2.at[162 + 34].set(1)
    # 角のaction
    # 飛のaction
    actions2 = actions2.at[81 * 4 + 7].set(1)
    for i in range(5):
        actions2 = actions2.at[81 * 3 + 25 + 9 * i].set(1)
    a3 = actions2 - actions1
    print(jnp.where(a3 == -1))
    assert jnp.all(actions2 == actions1)
    state = state.replace(turn=1)  # type: ignore
    actions1 = _legal_actions(state)
    actions2 = jnp.zeros(2754, dtype=jnp.int32)
    # 歩のaction
    for i in range(9):
        actions2 = actions2.at[3 + 9 * i].set(1)
    # 香車のaction
    actions2 = actions2.at[1].set(1)
    actions2 = actions2.at[73].set(1)
    # 桂馬のaction
    # 銀のaction
    actions2 = actions2.at[19].set(1)
    actions2 = actions2.at[55].set(1)
    actions2 = actions2.at[81 + 46].set(1)
    actions2 = actions2.at[162 + 28].set(1)
    # 金のaction
    for i in range(2):
        actions2 = actions2.at[28 + 18 * i].set(1)
        actions2 = actions2.at[81 + 37 - 18 * i].set(1)
        actions2 = actions2.at[162 + 55 - 18 * i].set(1)
    # 玉のaction
    actions2 = actions2.at[37].set(1)
    actions2 = actions2.at[81 + 28].set(1)
    actions2 = actions2.at[162 + 46].set(1)
    # 角のaction
    # 飛のaction
    actions2 = actions2.at[81 * 4 + 73].set(1)
    for i in range(5):
        actions2 = actions2.at[81 * 3 + 55 - 9 * i].set(1)
    assert jnp.all(actions1 == actions2)
    board = state.board
    board = board.at[16,39].set(1)
    board = board.at[0,39].set(0)
    board = board.at[19,43].set(1)
    board = board.at[0,43].set(0)
    state = state.replace(board=board)  # type: ignore
    actions1 = _legal_actions(state)
    actions2 = actions2.at[40].set(1)
    actions2 = actions2.at[41].set(1)
    actions2 = actions2.at[42].set(1)
    actions2 = actions2.at[42 + 810].set(1)
    actions2 = actions2.at[81 + 35].set(1)
    actions2 = actions2.at[162 + 53].set(1)
    actions2 = actions2.at[81 * 6 + 33].set(1)
    actions2 = actions2.at[81 * 7 + 51].set(1)
    actions2 = actions2.at[81 + 35 + 810].set(1)
    actions2 = actions2.at[162 + 53 + 810].set(1)
    actions2 = actions2.at[81 * 6 + 33 + 810].set(1)
    actions2 = actions2.at[81 * 7 + 51 + 810].set(1)
    actions2 = actions2.at[39].set(0)
    assert jnp.all(actions1 == actions2)
    # 後手の持ち駒に金と桂馬を追加
    legal_actions_white = state.legal_actions_white
    legal_actions_white = _add_drop_actions(17, legal_actions_white)
    legal_actions_white = _add_drop_actions(21, legal_actions_white)
    state = state.replace(legal_actions_white=legal_actions_white)  # type: ignore
    actions1 = _legal_actions(state)
    for i in range(9):
        if i == 0 or i == 2 or i == 6 or i == 8:
            continue
        for j in range(9):
            if i == 1 and j == 1:
                continue
            if i == 1 and j == 7:
                continue
            if i == 7 and j == 1:
                continue
            if i == 7 and j == 7:
                continue
            if i == 3 and j == 4:
                continue
            if i == 7 and j == 4:
                continue
            actions2 = actions2.at[81 * 29 + 9 * j + i].set(1)
            actions2 = actions2.at[81 * 33 + 9 * j + i].set(1)
    assert jnp.all(actions1 == actions2)


def test_update_legal_actions():
    s = init()
    # 58玉
    action = ShogiAction(False, 8, 43, 44, 0, False)
    s1 = _update_legal_move_actions(s, action)
    assert s1.legal_actions_black[43] == 0
    assert s1.legal_actions_black[81 + 52] == 0
    assert s1.legal_actions_black[162 + 34] == 0
    assert s1.legal_actions_black[243 + 53] == 0
    assert s1.legal_actions_black[324 + 35] == 0
    assert s1.legal_actions_black[42] == 1
    assert s1.legal_actions_black[81 + 51] == 1
    assert s1.legal_actions_black[162 + 33] == 1
    assert s1.legal_actions_black[243 + 52] == 1
    assert s1.legal_actions_black[324 + 34] == 1
    assert s1.legal_actions_black[405 + 44] == 1
    assert s1.legal_actions_black[486 + 53] == 1
    assert s1.legal_actions_black[567 + 35] == 1
    # 飛車の動きはもともと入っていないので更新しないでよい
    # 58飛車
    action2 = ShogiAction(False, 6, 43, 16, 0, False)
    s2 = _update_legal_move_actions(s, action2)
    # legal_actionsは更新しない
    assert jnp.all(s.legal_actions_black == s2.legal_actions_black)
    s3 = init()
    # 17の歩を消す
    board = s3.board
    board = board.at[1,6].set(0)
    board = board.at[0,6].set(1)
    s3.replace(board=board)  # type: ignore
    s3 = _init_legal_actions(s3)
    # 13香車成
    action3 = ShogiAction(False, 2, 2, 8, 15, True)
    s4 = _update_legal_move_actions(s3, action3)
    # 成香によるactionを追加する
    assert s4.legal_actions_black[1] == 1
    assert s4.legal_actions_black[81 + 10] == 1
    assert s4.legal_actions_black[243 + 11] == 1
    assert s4.legal_actions_black[405 + 3] == 1
    assert s4.legal_actions_white[3] == 0
    # 持ち駒に歩が増えたので歩を打つ手が追加される
    for i in range(81):
        assert s4.legal_actions_black[81 * 20 + i] == 1
    s4 = _move(s4, action3)
    s4 = s4.replace(turn=1)  # type: ignore
    # 13桂馬
    action4 = ShogiAction(False, 17, 2, 9, 10, False)
    s4 = _update_legal_move_actions(s4, action4)
    assert s4.legal_actions_black[1] == 0
    assert s4.legal_actions_black[81 + 10] == 0
    assert s4.legal_actions_black[243 + 11] == 0
    assert s4.legal_actions_black[405 + 3] == 0
    assert s4.legal_actions_white[81 * 8 + 2] == 0
    assert s4.legal_actions_white[81 * 9 + 20] == 0
    assert s4.legal_actions_white[81 * 9 + 13] == 1
    for i in range(81):
        assert s4.legal_actions_white[81 * 28 + i] == 1
    s4 = _move(s4, action4)
    s4 = s4.replace(turn=0)  # type: ignore
    # 12歩（駒打ち）
    action5 = ShogiAction(True, 1, 1)
    s5 = _update_legal_drop_actions(s4, action5)
    assert s5.legal_actions_black[0] == 1
    # 持ち駒の歩がなくなったので歩を打つactionを折る
    for i in range(81):
        assert s5.legal_actions_black[81 * 20 + i] == 0
    s5 = _drop(s5, action5)
    s5 = s5.replace(turn=1)  # type: ignore
    # 54香車（駒打ち）
    action6 = ShogiAction(True, 16, 39)
    s6 = _update_legal_drop_actions(s5, action6)
    for i in range(81):
        assert s6.legal_actions_white[81 * 28 + i] == 0


def test_is_double_pawn():
    s = init()
    board = s.board
    board = board.at[0,5].set(0)
    board = board.at[1,5].set(1)
    s = s.replace(board=board)  # type: ignore
    assert _is_double_pawn(s)
    s = s.replace(turn=1)
    board = s.board
    board = board.at[0,43].set(0)
    board = board.at[23,43].set(1)
    s = s.replace(board=board)  # type: ignore
    assert not _is_double_pawn(s)
    board = s.board
    board = board.at[0,41].set(0)
    board = board.at[15,41].set(1)
    s = s.replace(board=board)  # type: ignore
    assert _is_double_pawn(s)


def test_is_stuck():
    s = init()
    board = s.board
    board = board.at[0,1].set(0)
    board = board.at[3,1].set(1)
    s = s.replace(board=board)  # type: ignore
    assert _is_stuck(s)
    board = s.board
    board = board.at[2,8].set(0)
    board = board.at[16,8].set(1)
    s = s.replace(board=board)  # type: ignore
    assert _is_stuck(s)


# 駒種ごとに実行できるstepの確認
def test_step_piece():
    board = jnp.zeros((29, 81), dtype=jnp.int32)
    board = board.at[0].set(jnp.ones(81, dtype=jnp.int32))
    # 先手歩
    board = board.at[0,40].set(0)
    board = board.at[1,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手香車
    board = board.at[1,40].set(0)
    board = board.at[2,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if 37 <= i <= 39:
            s_, r, t = step(s, i)
            assert not t
        elif 846 <= i <= 848:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手桂馬
    board = board.at[2,40].set(0)
    board = board.at[3,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 695 or i == 1505 or i == 758 or i == 1568:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手銀
    board = board.at[3,40].set(0)
    board = board.at[4,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39 or i == 129 or i == 192 or i == 536 or i == 599:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手角
    board = board.at[4,40].set(0)
    board = board.at[5,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 129 or i == 137 or i == 145 or i == 153 or i == 947 or i == 955 or i == 963:
            s_, r, t = step(s, i)
            assert not t
        elif i == 192 or i == 182 or i == 172 or i == 162 or i == 992 or i == 982 or i == 972:
            s_, r, t = step(s, i)
            assert not t
        elif i == 536 or i == 546 or i == 556 or i == 566:
            s_, r, t = step(s, i)
            assert not t
        elif i == 599 or i == 591 or i == 583 or i == 575:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手飛車
    board = board.at[5,40].set(0)
    board = board.at[6,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39 or i == 38 or i == 37 or i == 36 or i == 848 or i == 847 or i == 846:
            s_, r, t = step(s, i)
            assert not t
        elif i == 446 or i == 447 or i == 448 or i == 449:
            s_, r, t = step(s, i)
            assert not t
        elif i == 292 or i == 301 or i == 310 or i == 319:
            s_, r, t = step(s, i)
            assert not t
        elif i == 355 or i == 346 or i == 337 or i == 328:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手金
    board = board.at[6,40].set(0)
    board = board.at[7,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39 or i == 129 or i == 192 or i == 292 or i == 355 or i == 446:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手玉
    board = board.at[7,40].set(0)
    board = board.at[8,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39 or i == 129 or i == 192 or i == 292 or i == 355 or i == 446 or i == 536 or i == 599:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手成金
    for j in range(4):
        board = board.at[8 + j,40].set(0)
        board = board.at[9 + j,40].set(1)
        s = _init_legal_actions(ShogiState(board=board))
        for i in range(1620):
            # s_, r, t = step(s, i)
            if i == 39 or i == 129 or i == 192 or i == 292 or i == 355 or i == 446:
                s_, r, t = step(s, i)
                assert not t
            else:
                continue
                assert t
    # 先手馬
    board = board.at[12,40].set(0)
    board = board.at[13,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 129 or i == 137 or i == 145 or i == 153:
            s_, r, t = step(s, i)
            assert not t
        elif i == 192 or i == 182 or i == 172 or i == 162:
            s_, r, t = step(s, i)
            assert not t
        elif i == 536 or i == 546 or i == 556 or i == 566:
            s_, r, t = step(s, i)
            assert not t
        elif i == 599 or i == 591 or i == 583 or i == 575:
            s_, r, t = step(s, i)
            assert not t
        elif i == 39 or i == 292 or i == 355 or i == 446:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 先手龍
    board = board.at[13,40].set(0)
    board = board.at[14,40].set(1)
    s = _init_legal_actions(ShogiState(board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 39 or i == 38 or i == 37 or i == 36:
            s_, r, t = step(s, i)
            assert not t
        elif i == 446 or i == 447 or i == 448 or i == 449:
            s_, r, t = step(s, i)
            assert not t
        elif i == 292 or i == 301 or i == 310 or i == 319:
            s_, r, t = step(s, i)
            assert not t
        elif i == 355 or i == 346 or i == 337 or i == 328:
            s_, r, t = step(s, i)
            assert not t
        elif i == 129 or i == 192 or i == 536 or i == 599:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手歩
    board = board.at[14,40].set(0)
    board = board.at[15,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手香車
    board = board.at[15,40].set(0)
    board = board.at[16,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if 41 <= i <= 43:
            s_, r, t = step(s, i)
            assert not t
        elif 852 <= i <= 854:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手桂馬
    board = board.at[16,40].set(0)
    board = board.at[17,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 681 or i == 1491 or i == 780 or i == 1590:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手銀
    board = board.at[17,40].set(0)
    board = board.at[18,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41 or i == 113 or i == 212 or i == 516 or i == 615:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手角
    board = board.at[18,40].set(0)
    board = board.at[19,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 113 or i == 105 or i == 97 or i == 89 or i == 915 or i == 907 or i == 899:
            s_, r, t = step(s, i)
            assert not t
        elif i == 212 or i == 222 or i == 232 or i == 242 or i == 1032 or i == 1042 or i == 1052:
            s_, r, t = step(s, i)
            assert not t
        elif i == 516 or i == 506 or i == 496 or i == 486:
            s_, r, t = step(s, i)
            assert not t
        elif i == 615 or i == 623 or i == 631 or i == 639:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手飛車
    board = board.at[19,40].set(0)
    board = board.at[20,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41 or i == 42 or i == 43 or i == 44 or i == 852 or i == 853 or i == 854:
            s_, r, t = step(s, i)
            assert not t
        elif i == 444 or i == 443 or i == 442 or i == 441:
            s_, r, t = step(s, i)
            assert not t
        elif i == 373 or i == 382 or i == 391 or i == 400:
            s_, r, t = step(s, i)
            assert not t
        elif i == 274 or i == 265 or i == 256 or i == 247:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手金
    board = board.at[20,40].set(0)
    board = board.at[21,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41 or i == 113 or i == 212 or i == 274 or i == 373 or i == 444:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手玉
    board = board.at[21,40].set(0)
    board = board.at[22,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41 or i == 113 or i == 212 or i == 274 or i == 373 or i == 444 or i == 516 or i == 615:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手成金
    for j in range(4):
        board = board.at[22 + j,40].set(0)
        board = board.at[23 + j,40].set(1)
        s = _init_legal_actions(ShogiState(turn=1, board=board))
        for i in range(1620):
            # s_, r, t = step(s, i)
            if i == 41 or i == 113 or i == 212 or i == 274 or i == 373 or i == 444:
                s_, r, t = step(s, i)
                assert not t
            else:
                continue
                assert t
    # 後手馬
    board = board.at[26,40].set(0)
    board = board.at[27,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 113 or i == 105 or i == 97 or i == 89:
            s_, r, t = step(s, i)
            assert not t
        elif i == 212 or i == 222 or i == 232 or i == 242:
            s_, r, t = step(s, i)
            assert not t
        elif i == 516 or i == 506 or i == 496 or i == 486:
            s_, r, t = step(s, i)
            assert not t
        elif i == 615 or i == 623 or i == 631 or i == 639:
            s_, r, t = step(s, i)
            assert not t
        elif i == 41 or i == 274 or i == 373 or i == 444:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t
    # 後手龍
    board = board.at[27,40].set(0)
    board = board.at[28,40].set(1)
    s = _init_legal_actions(ShogiState(turn=1, board=board))
    for i in range(1620):
        # s_, r, t = step(s, i)
        if i == 41 or i == 42 or i == 43 or i == 44:
            s_, r, t = step(s, i)
            assert not t
        elif i == 444 or i == 443 or i == 442 or i == 441:
            s_, r, t = step(s, i)
            assert not t
        elif i == 373 or i == 382 or i == 391 or i == 400:
            s_, r, t = step(s, i)
            assert not t
        elif i == 274 or i == 265 or i == 256 or i == 247:
            s_, r, t = step(s, i)
            assert not t
        elif i == 113 or i == 212 or i == 516 or i == 615:
            s_, r, t = step(s, i)
            assert not t
        else:
            continue
            assert t


def test_step():
    board = jnp.zeros((29, 81), dtype=jnp.int32)
    board = board.at[0].set(jnp.ones(81, dtype=jnp.int32))
    board = board.at[0,11].set(0)
    board = board.at[7,11].set(1)
    board = board.at[0,10].set(0)
    board = board.at[1,10].set(1)
    board = board.at[0,0].set(0)
    board = board.at[22,0].set(1)
    hand = jnp.zeros(14, dtype=jnp.int32)
    hand = hand.at[0].set(1)
    hand = hand.at[1].set(1)
    s = _init_legal_actions(ShogiState(board=board, hand=hand))
    # 打ち不詰め
    action1 = _action_to_dlaction(ShogiAction(True, 1, 1), 0)
    s1, r1, t = step(s, action1)
    assert r1 == -1
    assert t
    # スティルメイト
    print(s.hand)
    action1 = _action_to_dlaction(ShogiAction(True, 1, 2), 0)
    s1, r1, t = step(s, action1)
    assert r1 == 0
    assert not t
    # 詰み（合い駒なし）
    action2 = _action_to_dlaction(ShogiAction(True, 2, 8), 0)
    s2, r2, t = step(s, action2)
    assert r2 == 1
    assert t
    # 行き所のない駒
    action3 = _action_to_dlaction(ShogiAction(True, 2, 9), 0)
    s3, r3, t = step(s, action3)
    assert r3 == -1
    assert t
    # 二歩
    action4 = _action_to_dlaction(ShogiAction(True, 1, 17), 0)
    s4, r4, t = step(s, action4)
    assert r4 == -1
    assert t
    # 行き所のない駒
    action5 = _action_to_dlaction(ShogiAction(False, 1, 9, 10, 0, False), 0)
    s5, r5, t = step(s, action5)
    assert r5 == -1
    assert t
    # 合法手
    action6 = _action_to_dlaction(ShogiAction(False, 1, 9, 10, 0, True), 0)
    s6, r6, t = step(s, action6)
    assert r6 == 0
    assert not t
    s = init()
    # 相手は指せるが自分は指せない手
    action7 = _action_to_dlaction(ShogiAction(False, 1, 2, 3, 0, False), 0)
    s7, r7, t = step(s, action7)
    assert r7 == -1
    assert t
    # 自分の駒を跳び越す手
    action8 = _action_to_dlaction(ShogiAction(False, 5, 10, 70, 15, True), 0)
    s8, r8, t = step(s, action8)
    assert r8 == -1
    assert t
    moves = [
        59, 66, 162 + 52, 21, 81 + 60, 57, 14, 55, 13, 81 + 19, 81 + 61, 48, 81 + 34, 648 + 56, 567 + 62, 162 + 28,
        23, 81 + 47, 162 + 24, 39, 243 + 53, 67, 162 + 52, 81 + 20, 41, 567 + 18, 81 + 32, 49, 22, 22, 162 + 22,
        162 + 48, 162 + 32, 81 + 32, 486 + 32, 46, 243 + 62, 81 + 28, 77, 405 + 63, 1944 + 7, 243 + 36, 648 + 24, 30,
        81 + 70, 3, 243 + 43, 162 + 29, 40, 40, 40, 2187 + 39, 405 + 43, 2187 + 21, 81 + 15, 243 + 19, 5, 2511 + 48, 34,
        12, 4, 4, 1620 + 2, 13, 1620 + 40, 81 + 12, 39, 162 + 39, 1620 + 22, 22, 81 + 39, 81 + 39, 2106 + 21, 2187 + 42,
        42, 2187 + 41, 41, 2187 + 40, 405 + 44, 68, 68, 23, 324 + 12, 810 + 24, 324 + 26, 2187 + 25, 324 + 17, 50, 50,
        162 + 34, 243 + 21, 2511 + 41, 243 + 30, 2349 + 75, 1620 + 20, 729 + 20, 1620 + 21, 324 + 63, 810 + 20,
        567 + 27, 1620 + 38, 972 + 61, 324 + 61, 648 + 68, 81 + 69, 1539 + 79, 1620 + 68, 2673 + 61, 1863 + 28
             ]
    for i in range(109):
        action = moves[i]
        s, r, t = step(s, action)
        if i == 108:
            assert r == 1
            assert t
        else:
            assert r == 0
            assert not t
    assert jnp.all(_board_status(s) == jnp.array([16, 0, 1, 0, 15, 0, 0, 0, 2, 0, 0, 0, 0, 15, 0, 0, 0, 6, 0, 0, 9, 0, 0, 0, 0, 15, 0, 22, 4, 0, 7, 0, 4, 1, 23, 0, 0, 0, 1, 18, 15, 0, 0, 0, 0, 0, 21, 0, 19, 0, 1, 0, 0, 0, 0, 0, 17, 15, 0, 1, 4, 21, 0, 20, 0, 0, 0, 0, 1, 8, 0, 3, 16, 0, 15, 0, 0, 1, 0, 25, 2]))
    s = init()
    moves = [
        14, 66, 13, 67, 59, 81 + 19, 162 + 60, 21, 162 + 52, 972 + 60, 81 + 60, 81 + 10, 81 + 34, 162 + 20, 23, 81 + 46,
        162 + 24, 48, 81 + 32, 47, 81 + 52, 81 + 39, 243 + 61, 162 + 55, 52, 30, 22, 486 + 29, 81 + 43, 57, 50, 22,
        162 + 22, 2187 + 21, 486 + 32, 81 + 28, 77, 75, 81 + 51, 81 + 47, 243 + 70, 3, 5, 648 + 56, 243 + 61, 486 + 18,
        1620 + 22, 81 + 10, 41, 567 + 18, 21, 81 + 21, 81 + 40, 2187 + 23, 567 + 32, 243 + 46, 243 + 25, 81 + 10, 23,
        2511 + 16, 162 + 22, 81 + 13, 1620 + 21, 567 + 28, 243 + 32, 1377 + 32, 486 + 32, 49, 49, 648 + 49, 567 + 52,
        68, 68, 2187 + 69, 81 + 69, 2187 + 50, 50, 2592 + 16, 405 + 51, 810 + 17, 1620 + 50, 58, 58, 2349 + 65,
        1944 + 66, 324 + 55, 1944 + 7, 486 + 7, 7, 2511 + 53, 49, 648 + 58, 567 + 58, 58, 1620 + 59, 2511 + 61,
        1782 + 62, 1377 + 69, 648 + 69, 2673 + 61, 243 + 79, 243 + 49, 1620 + 50, 2187 + 70, 49, 810 + 71, 1944 + 50,
        76, 2025 + 63, 324 + 80, 405 + 80, 2268 + 78, 1782 + 79, 1296 + 43, 162 + 30, 2187 + 20, 243 + 60, 324 + 52,
        405 + 61, 324 + 61, 2106 + 70, 2349 + 60, 486 + 60, 405 + 60, 162 + 60, 810 + 79, 79, 2430 + 78, 78, 2511 + 62,
        1782 + 70, 77, 405 + 79, 810 + 78, 405 + 80, 81 + 70
    ]
    for i in range(136):
        action = moves[i]
        s, r, t = step(s, action)
        if i == 135:
            assert r == -1
            assert t
        else:
            assert r == 0
            assert not t
    assert jnp.all(_board_status(s) == jnp.array([16, 0, 0, 15, 0, 1, 0, 2, 0, 17, 22, 15, 0, 18, 0, 0, 0, 0, 0, 21, 15, 1, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 4, 1, 0, 0, 0, 0, 15, 0, 0, 1, 0, 0, 0, 0, 0, 21, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 19, 6, 0, 0, 0, 0, 1, 3, 23, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8]))
    s = init()
    s = s.replace(turn=1)
    moves = [
        21, 41, 66, 40, 67, 43, 68, 42, 810 + 69, 41, 70, 243 + 70, 2187 + 69, 324 + 16, 810 + 70, 81 + 70, 810 + 69,
        567 + 62, 162 + 79, 14, 81 + 71, 162 + 52, 324 + 80, 39, 405 + 79, 13, 2349 + 22, 12, 729 + 33, 810 + 11,
        1458 + 26, 81 + 19, 972 + 60, 243 + 28, 162 + 28, 810 + 38, 162 + 38, 810 + 10, 2187 + 39, 1620 + 37, 81 + 37,
        162, 486, 1701 + 40, 162 + 10, 39, 39, 1620 + 38, 38, 1620 + 40, 162 + 40, 567 + 33, 2592 + 41, 34, 324 + 50,
        162 + 42, 81 + 32, 162 + 32, 243 + 32, 32, 2511 + 22, 324 + 23, 2430 + 32, 1944 + 33, 972 + 42, 324 + 25,
        729 + 20, 324 + 16, 2268 + 10, 15, 3, 14, 4, 567 + 15, 486 + 69, 162 + 23, 486 + 59, 162 + 13, 567 + 67,
        81 + 21, 81 + 31, 567 + 13, 243 + 40, 567 + 5, 486 + 21, 162 + 43, 486 + 11, 162 + 33, 486 + 1, 162 + 23,
        405 + 25, 50, 162 + 50, 77, 162 + 60, 76, 162 + 70, 75, 243 + 16
    ]
    for i in range(99):
        action = moves[i]
        s, r, t = step(s, action)
        if i == 98:
            assert r == -1
            assert t
        else:
            assert r == 0
            assert not t
    assert jnp.all(_board_status(s) == jnp.array(
        [0, 18, 0, 0, 15, 5, 1, 0, 2, 0, 16, 0, 0, 0, 7, 8, 25, 3, 0, 0, 17, 0, 19, 7, 1, 0, 0, 21, 0, 15, 0, 0, 0, 0,
         0, 0, 22, 0, 21, 0, 0, 0, 26, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 18, 0, 15, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0,
         0, 28, 0, 16, 0, 15, 1, 0, 0, 0, 0, 0]))


def test_between():
    b1 = _between(36, 44)
    for i in range(81):
        if 37 <= i <= 43:
            assert b1[i] == 1
        else:
            assert b1[i] == 0
    b2 = _between(44, 36)
    for i in range(81):
        if 37 <= i <= 43:
            assert b2[i] == 1
        else:
            assert b2[i] == 0
    b3 = _between(40, 0)
    for i in range(81):
        if i == 10 or i == 20 or i == 30:
            assert b3[i] == 1
        else:
            assert b3[i] == 0
    b4 = _between(40, 1)
    for i in range(81):
        assert b4[i] == 0


def test_pin():
    s = ShogiState()
    board = s.board
    board = board.at[8,40].set(1)
    board = board.at[0,40].set(0)
    board = board.at[16,38].set(1)
    board = board.at[0,38].set(0)
    board = board.at[3,39].set(1)
    board = board.at[0,39].set(0)
    board = board.at[19,64].set(1)
    board = board.at[0,64].set(0)
    board = board.at[15,56].set(1)
    board = board.at[0,56].set(0)
    board = board.at[28,76].set(1)
    board = board.at[0,76].set(0)
    board = board.at[1,67].set(1)
    board = board.at[0,67].set(0)
    board = board.at[27,80].set(1)
    board = board.at[0,80].set(0)
    board = board.at[2,50].set(1)
    board = board.at[0,50].set(0)
    s = s.replace(board=board)  # type: ignore
    pins = _pin(s)
    assert pins[39] == 1
    assert pins[56] == 0
    assert pins[67] == 4
    assert pins[50] == 3
    s2 = ShogiState(board=jnp.zeros((29, 81), dtype=jnp.int32))
    board = s2.board
    board = board.at[22,40].set(1)
    board = board.at[0,40].set(0)
    board = board.at[14,38].set(1)
    board = board.at[0,38].set(0)
    board = board.at[23,39].set(1)
    board = board.at[0,39].set(0)
    board = board.at[5,64].set(1)
    board = board.at[0,64].set(0)
    board = board.at[15,56].set(1)
    board = board.at[0,56].set(0)
    board = board.at[6,76].set(1)
    board = board.at[0,76].set(0)
    board = board.at[20,67].set(1)
    board = board.at[0,67].set(0)
    board = board.at[26,58].set(1)
    board = board.at[0,58].set(0)
    board = board.at[13,80].set(1)
    board = board.at[0,80].set(0)
    board = board.at[17,50].set(1)
    board = board.at[0,50].set(0)
    board = board.at[2,44].set(1)
    board = board.at[0,44].set(0)
    board = board.at[17,42].set(1)
    board = board.at[0,42].set(0)
    s2 = s2.replace(board=board, turn=1)  # type: ignore
    pins2 = _pin(s2)
    assert pins2[39] == 1
    assert pins2[56] == 2
    assert pins2[67] == 0
    assert pins2[58] == 0
    assert pins2[50] == 3
    assert pins2[42] == 1


def test_is_mate():
    board = jnp.zeros(81, dtype=jnp.int32)
    board = board.at[30].set(1)
    board = board.at[31].set(1)
    board = board.at[32].set(1)
    board = board.at[48].set(1)
    board = board.at[49].set(1)
    board = board.at[50].set(1)
    board = board.at[40].set(8)
    board = board.at[38].set(16)
    s = _make_board(board)
    s = _init_legal_actions(s)
    assert _is_mate(s)
    s = s.replace(hand=s.hand.at[0].set(1))  # type: ignore
    s = _init_legal_actions(s)
    assert not _is_mate(s)
    board1 = jnp.zeros(81, dtype=jnp.int32)
    board1 = board1.at[8].set(8)
    board1 = board1.at[7].set(1)
    board1 = board1.at[16].set(1)
    board1 = board1.at[26].set(25)
    board1 = board1.at[15].set(17)
    board1 = board1.at[64].set(27)
    s1 = _make_board(board1)
    s1 = _init_legal_actions(s1)
    assert _is_mate(s1)
    board1 = board1.at[56].set(28)
    s1 = _make_board(board1)
    s1 = _init_legal_actions(s1)
    assert not _is_mate(s1)
    board2 = jnp.zeros(81, dtype=jnp.int32)
    board2 = board2.at[8].set(8)
    board2 = board2.at[7].set(1)
    board2 = board2.at[64].set(27)
    board2 = board2.at[56].set(5)
    board2 = board2.at[48].set(19)
    s2 = _make_board(board2)
    s2 = _init_legal_actions(s2)
    assert not _is_mate(s2)
    board3 = jnp.zeros(81, dtype=jnp.int32)
    board3 = board3.at[8].set(8)
    board3 = board3.at[7].set(15)
    board3 = board3.at[14].set(17)
    board3 = board3.at[16].set(7)
    board3 = board3.at[17].set(4)
    board3 = board3.at[24].set(19)
    board3 = board3.at[35].set(28)
    s3 = _make_board(board3)
    s3 = _init_legal_actions(s3)
    assert _is_mate(s3)
