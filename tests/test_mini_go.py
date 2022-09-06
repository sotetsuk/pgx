import numpy as np

from pgx.mini_go import BLACK, WHITE, _is_surrounded, init, step


def test_is_surrounded():
    state = init()
    state, _, _ = step(state=state, action=np.array([0, 1, BLACK]))
    state, _, _ = step(state=state, action=np.array([1, 0, BLACK]))
    state, _, _ = step(state=state, action=np.array([2, 0, BLACK]))
    state, _, _ = step(state=state, action=np.array([1, 2, BLACK]))
    state, _, _ = step(state=state, action=np.array([2, 2, BLACK]))
    state, _, _ = step(state=state, action=np.array([3, 1, BLACK]))
    state, _, _ = step(state=state, action=np.array([4, 0, BLACK]))
    state, _, _ = step(state=state, action=np.array([1, 1, WHITE]))
    state, _, _ = step(state=state, action=np.array([2, 1, WHITE]))
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
    state = init()

    state, _, done = step(state=state, action=None)
    assert not done
    state, _, done = step(state=state, action=np.array([0, 1, BLACK]))
    assert not done
    state, _, done = step(state=state, action=None)
    assert not done
    _, _, done = step(state=state, action=None)
    assert done
