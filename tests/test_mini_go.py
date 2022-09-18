import numpy as np

from pgx.mini_go import get_board, init, legal_actions, step


def test_end_by_pass():
    state = init()

    state, _, done = step(_state=state, action=None)
    assert state.passed[0]
    assert not done
    state, _, done = step(_state=state, action=0)
    assert not state.passed[0]
    assert not done
    state, _, done = step(_state=state, action=None)
    assert state.passed[0]
    assert not done
    state, _, done = step(_state=state, action=None)
    assert state.passed[0]
    assert done


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    state = init()
    state, _, _ = step(_state=state, action=12)  # BLACK
    state, _, _ = step(_state=state, action=11)  # WHITE
    state, _, _ = step(_state=state, action=17)
    state, _, _ = step(_state=state, action=7)
    state, _, _ = step(_state=state, action=8)
    state, _, _ = step(_state=state, action=1)
    state, _, _ = step(_state=state, action=3)
    state, _, _ = step(_state=state, action=16)
    state, _, _ = step(_state=state, action=21)
    state, _, _ = step(_state=state, action=2)
    state, _, _ = step(_state=state, action=10)
    state, _, _ = step(_state=state, action=5)
    state, _, _ = step(_state=state, action=14)
    state, _, _ = step(_state=state, action=15)
    state, _, _ = step(_state=state, action=23)
    state, _, _ = step(_state=state, action=20)
    state, _, _ = step(_state=state, action=None)
    state, _, _ = step(_state=state, action=22)
    state, _, _ = step(_state=state, action=19)
    state, _, _ = step(_state=state, action=21)
    state, _, _ = step(_state=state, action=None)
    state, r, done = step(_state=state, action=None)

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
    assert (get_board(state) == expected_board.ravel()).all()
    assert done
    assert (r == np.array([0, 0])).all()


def test_kou():
    state = init()
    state, _, _ = step(_state=state, action=2)  # BLACK
    state, _, _ = step(_state=state, action=17)  # WHITE
    state, _, _ = step(_state=state, action=6)  # BLACK
    state, _, _ = step(_state=state, action=13)  # WHITE
    state, _, _ = step(_state=state, action=8)  # BLACK
    state, _, _ = step(_state=state, action=11)  # WHITE
    state, _, _ = step(_state=state, action=12)  # BLACK
    state, _, _ = step(_state=state, action=7)  # WHITE

    assert (state.kou == np.array([2, 2])).all()

    _, r, done = step(_state=state, action=12)  # BLACK
    # ルール違反により黒の負け
    assert done
    assert (r == np.array([-1, 1])).all()

    state, _, done = step(_state=state, action=0)  # BLACK
    # 回避した場合
    assert not done
    assert (state.kou == np.array([-1, -1])).all()


def test_random_play():
    state = init()
    done = False
    while not done:
        actions = np.where(legal_actions(state))
        if len(actions[0]) == 0:
            a = None
        else:
            a = np.random.choice(actions[0], 1)[0]
        state, _, done = step(_state=state, action=a)

        if state.turn[0] > 1000:
            break
