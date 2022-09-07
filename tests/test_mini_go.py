import numpy as np

from pgx.mini_go import (
    BLACK,
    WHITE,
    MiniGoState,
    _is_surrounded,
    _is_surrounded_v2,
    init,
    legal_actions,
    step,
    to_init_board,
)


def test_is_surrounded():
    init_board = to_init_board("+@+++@O@++@O@+++@+++@++++")
    state = MiniGoState(board=init_board)
    """
      [ 0 1 2 3 4 ]
    [0] + @ + + +
    [1] @ O @ + +
    [2] @ O @ + +
    [3] + @ + + +
    [4] @ + + + +
    """

    b = _is_surrounded(
        state.board,
        1,
        1,
        WHITE,
    )
    assert b

    b = _is_surrounded(
        state.board,
        4,
        0,
        BLACK,
    )
    assert not b

    b = _is_surrounded_v2(
        state.board,
        1,
        1,
        WHITE,
    )
    assert b

    b = _is_surrounded_v2(
        state.board,
        4,
        0,
        BLACK,
    )
    assert not b

    init_board = to_init_board("++@OO@@@O@@OOOO@O@OO@OOOO")
    state = MiniGoState(board=init_board)
    """
      [ 0 1 2 3 4 ]
    [0] + + @ O O
    [1] @ @ @ O @
    [2] @ O O O O
    [3] @ O @ O O
    [4] @ O O O O
    """

    b = _is_surrounded(
        state.board,
        0,
        4,
        WHITE,
    )
    assert b

    b = _is_surrounded(
        state.board,
        4,
        1,
        WHITE,
    )
    assert b

    b = _is_surrounded_v2(
        state.board,
        0,
        4,
        WHITE,
    )
    assert b

    b = _is_surrounded_v2(
        state.board,
        4,
        1,
        WHITE,
    )
    assert b


def test_end_by_pass():
    state = init()

    state, _, done = step(state=state, action=None)
    assert state.passed[0]
    assert not done
    state, _, done = step(state=state, action=np.array([0, 1]))
    assert not state.passed[0]
    assert not done
    state, _, done = step(state=state, action=None)
    assert state.passed[0]
    assert not done
    state, _, done = step(state=state, action=None)
    assert state.passed[0]
    assert done


def test_remove():
    init_board = to_init_board("++@OO@@@OO@OOOO@O+OO@OOOO")
    state = MiniGoState(board=init_board)
    """
      [ 0 1 2 3 4 ]
    [0] + + @ O O
    [1] @ @ @ O O
    [2] @ O O O O
    [3] @ O + O O
    [4] @ O O O O
    """

    state, _, _ = step(state=state, action=np.array([3, 2]))
    expected_board = np.array(
        [
            [2, 2, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [0, 2, 2, 2, 2],
            [0, 2, 0, 2, 2],
            [0, 2, 2, 2, 2],
        ]
    )
    """
      [ 0 1 2 3 4 ]
    [0] + + @ + +
    [1] @ @ @ + +
    [2] @ + + + +
    [3] @ + @ + +
    [4] @ + + + +
    """

    assert (state.board == expected_board).all()


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    state = init()
    state, _, _ = step(state=state, action=np.array([2, 2]))  # BLACK
    state, _, _ = step(state=state, action=np.array([2, 1]))  # WHITE
    state, _, _ = step(state=state, action=np.array([3, 2]))
    state, _, _ = step(state=state, action=np.array([1, 2]))
    state, _, _ = step(state=state, action=np.array([1, 3]))
    state, _, _ = step(state=state, action=np.array([0, 1]))
    state, _, _ = step(state=state, action=np.array([0, 3]))
    state, _, _ = step(state=state, action=np.array([3, 1]))
    state, _, _ = step(state=state, action=np.array([4, 1]))
    state, _, _ = step(state=state, action=np.array([0, 2]))
    state, _, _ = step(state=state, action=np.array([2, 0]))
    state, _, _ = step(state=state, action=np.array([1, 0]))
    state, _, _ = step(state=state, action=np.array([2, 4]))
    state, _, _ = step(state=state, action=np.array([3, 0]))
    state, _, _ = step(state=state, action=np.array([4, 3]))
    state, _, _ = step(state=state, action=np.array([4, 0]))
    state, _, _ = step(state=state, action=None)
    state, _, _ = step(state=state, action=np.array([4, 2]))
    state, _, _ = step(state=state, action=np.array([3, 4]))
    state, _, _ = step(state=state, action=np.array([4, 1]))
    state, _, _ = step(state=state, action=None)
    state, _, done = step(state=state, action=None)

    expected_board = np.array(
        [
            [2, 1, 1, 0, 2],
            [1, 2, 1, 0, 2],
            [2, 1, 0, 2, 0],
            [1, 1, 0, 2, 0],
            [1, 1, 1, 0, 2],
        ]
    )
    """
      [ 0 1 2 3 4 ]
    [0] + O O @ +
    [1] O + O @ +
    [2] + O @ + @
    [3] O O @ + @
    [4] O O O @ +
    """
    assert (state.board == expected_board).all()
    assert done


def test_legal_actions():
    init_board = to_init_board("++@OO@@@O+@OOOO@O@++@++++")
    state = MiniGoState(board=init_board)
    """
    turn=0なので黒番

        [ 0 1 2 3 4 ]
    [0] + + @ O O
    [1] @ @ @ O +
    [2] @ O O O O
    [3] @ O @ + +
    [4] @ + + + +
    """

    expected = np.array(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, True, True, True, True],
        ],
        dtype=bool,
    )
    assert (legal_actions(state) == expected).all()

    init_board = to_init_board("OOOOOOOOOOOOOOOOOOOOOOOOO")
    state = MiniGoState(board=init_board)

    assert True not in legal_actions(state)


def test_kou():
    init_board = to_init_board("++O+++O+O++@O@+++@+++++++")
    state = MiniGoState(board=init_board)
    """
      [ 0 1 2 3 4 ]
    [0] + + O + +
    [1] + O + O +
    [2] + @ O @ +
    [3] + + @ + +
    [4] + + + + +
    """

    state, _, _ = step(state=state, action=np.array([1, 2]))

    """
      [ 0 1 2 3 4 ]
    [0] + + O + +
    [1] + O @ O +
    [2] + @ + @ +
    [3] + + @ + +
    [4] + + + + +

    """
    expected = np.array(
        [
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ]
    )
    assert (state.kou == np.array([2, 2])).all()
    assert (expected == legal_actions(state)).all()

    _, _, done = step(state=state, action=np.array([2, 2]))
    assert done  # ルール違反により終局


def test_random_play():
    state = init()
    done = False
    while not done:
        actions = np.where(legal_actions(state))
        if actions[0].size == 0:
            a = None
        else:
            i = np.random.randint(0, actions[0].size)
            a = np.array([actions[0][i], actions[1][i]])
        state, _, done = step(state=state, action=a)

        if state.turn[0] > 1000:
            break
