import numpy as np

from pgx.mini_go import (
    BLACK,
    WHITE,
    _is_surrounded,
    _is_surrounded_v2,
    init,
    step,
    to_init_board,
)


def test_is_surrounded():
    init_board = to_init_board("+@+++@O@++@O@+++@+++@++++")
    state = init(init_board)
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
    state = init(init_board)
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
    state = init(None)

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
