import numpy as np

from pgx.mini_go import BLACK, WHITE, _is_surrounded, init, step, to_init_board


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

    b, _ = _is_surrounded(
        state.board, 1, 1, WHITE, np.zeros_like(state.board, dtype=bool)
    )
    assert b

    b, _ = _is_surrounded(
        state.board, 4, 0, BLACK, np.zeros_like(state.board, dtype=bool)
    )
    assert not b


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
