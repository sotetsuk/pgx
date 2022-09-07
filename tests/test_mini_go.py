import numpy as np

from pgx.mini_go import (
    BLACK,
    WHITE,
    MiniGoState,
    _is_surrounded,
    _is_surrounded_v2,
    _to_init_board,
    init,
    step,
)


def test_is_surrounded():
    init_board = _to_init_board("+@+++@O@++@O@+++@+++@++++")
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

    init_board = _to_init_board("++@OO@@@O@@OOOO@O@OO@OOOO")
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
    state, _, done = step(state=state, action=np.array([0, 1, BLACK]))
    assert not state.passed[0]
    assert not done
    state, _, done = step(state=state, action=None)
    assert state.passed[0]
    assert not done
    state, _, done = step(state=state, action=None)
    assert state.passed[0]
    assert done


def test_remove():
    init_board = _to_init_board("++@OO@@@OO@OOOO@O+OO@OOOO")
    state = MiniGoState(board=init_board)
    """
      [ 0 1 2 3 4 ]
    [0] + + @ O O
    [1] @ @ @ O O
    [2] @ O O O O
    [3] @ O + O O
    [4] @ O O O O
    """

    state, _, _ = step(state=state, action=np.array([3, 2, BLACK]))
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
    state, _, _ = step(state=state, action=np.array([2, 2, BLACK]))
    state, _, _ = step(state=state, action=np.array([2, 1, WHITE]))
    state, _, _ = step(state=state, action=np.array([3, 2, BLACK]))
    state, _, _ = step(state=state, action=np.array([1, 2, WHITE]))
    state, _, _ = step(state=state, action=np.array([1, 3, BLACK]))
    state, _, _ = step(state=state, action=np.array([0, 1, WHITE]))
    state, _, _ = step(state=state, action=np.array([0, 3, BLACK]))
    state, _, _ = step(state=state, action=np.array([3, 1, WHITE]))
    state, _, _ = step(state=state, action=np.array([4, 1, BLACK]))
    state, _, _ = step(state=state, action=np.array([0, 2, WHITE]))
    state, _, _ = step(state=state, action=np.array([2, 0, BLACK]))
    state, _, _ = step(state=state, action=np.array([1, 0, WHITE]))
    state, _, _ = step(state=state, action=np.array([2, 4, BLACK]))
    state, _, _ = step(state=state, action=np.array([3, 0, WHITE]))
    state, _, _ = step(state=state, action=np.array([4, 3, BLACK]))
    state, _, _ = step(state=state, action=np.array([4, 0, WHITE]))
    state, _, _ = step(state=state, action=None)
    state, _, _ = step(state=state, action=np.array([4, 2, WHITE]))
    state, _, _ = step(state=state, action=np.array([3, 4, BLACK]))
    state, _, _ = step(state=state, action=np.array([4, 1, WHITE]))
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
