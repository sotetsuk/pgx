import jax
import jax.numpy as jnp

from pgx.mini_go import get_board, init, legal_actions, step


def test_end_by_pass():
    state = init()

    state, _, done = step(state=state, action=-1)
    assert state.passed[0]
    assert not done
    state, _, done = step(state=state, action=0)
    assert not state.passed[0]
    assert not done
    state, _, done = step(state=state, action=-1)
    assert state.passed[0]
    assert not done
    state, _, done = step(state=state, action=-1)
    assert state.passed[0]
    assert done


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    state = init()
    state, _, _ = step(state=state, action=12)  # BLACK
    state, _, _ = step(state=state, action=11)  # WHITE
    state, _, _ = step(state=state, action=17)
    state, _, _ = step(state=state, action=7)
    state, _, _ = step(state=state, action=8)
    state, _, _ = step(state=state, action=1)
    state, _, _ = step(state=state, action=3)
    state, _, _ = step(state=state, action=16)
    state, _, _ = step(state=state, action=21)
    state, _, _ = step(state=state, action=2)
    state, _, _ = step(state=state, action=10)
    state, _, _ = step(state=state, action=5)
    state, _, _ = step(state=state, action=14)
    state, _, _ = step(state=state, action=15)
    state, _, _ = step(state=state, action=23)
    state, _, _ = step(state=state, action=20)
    state, _, _ = step(state=state, action=-1)  # pass
    state, _, _ = step(state=state, action=22)
    state, _, _ = step(state=state, action=19)
    state, _, _ = step(state=state, action=21)
    state, _, _ = step(state=state, action=-1)  # pass
    state, r, done = step(state=state, action=-1)  # pass

    expected_board: jnp.ndarray = jnp.array(
        [
            [2, 1, 1, 0, 2],
            [1, 2, 1, 0, 2],
            [2, 1, 0, 2, 0],
            [1, 1, 0, 2, 0],
            [1, 1, 1, 0, 2],
        ]
    )  # type:ignore
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
    assert (r == jnp.array([0, 0])).all()


def test_kou():
    state = init()
    state, _, _ = step(state=state, action=2)  # BLACK
    state, _, _ = step(state=state, action=17)  # WHITE
    state, _, _ = step(state=state, action=6)  # BLACK
    state, _, _ = step(state=state, action=13)  # WHITE
    state, _, _ = step(state=state, action=8)  # BLACK
    state, _, _ = step(state=state, action=11)  # WHITE
    state, _, _ = step(state=state, action=12)  # BLACK
    state, _, _ = step(state=state, action=7)  # WHITE

    assert (state.kou == jnp.array([2, 2])).all()

    _, r, done = step(state=state, action=12)  # BLACK
    # ルール違反により黒の負け
    assert done
    assert (r == jnp.array([-1, 1])).all()

    state, _, done = step(state=state, action=0)  # BLACK
    # 回避した場合
    assert not done
    assert (state.kou == jnp.array([-1, -1])).all()


def test_numpy_and_jax():
    import numpy as np

    from pgx._mini_go import get_board as _get_board
    from pgx._mini_go import init as _init
    from pgx._mini_go import legal_actions as _legal_actions
    from pgx._mini_go import step as _step

    kifu = []
    state_boards = []
    state = _init()
    done = False

    # numpy
    while not done:
        actions = np.where(_legal_actions(state))
        if len(actions[0]) == 0:
            a = -1
        else:
            a = np.random.choice(actions[0], 1)[0]
        state, _, done = _step(state=state, action=a)
        kifu.append(a)
        state_boards.append(_get_board(state))
        if state.turn[0] > 50:
            break

    # jax
    state = init()
    for i, a in enumerate(kifu):
        state, _, done = step(state=state, action=a)
        expected_board: jnp.ndarray = jnp.array(state_boards[i])  # type:ignore
        assert (expected_board == get_board(state)).all()

        if done:
            assert i == len(state_boards) - 1
            break


# ものすごくコンパイルに時間がかかる
def _test_random_play():
    state = init()
    done = False
    while not done:
        actions = jnp.where(legal_actions(state))
        if len(actions[0]) == 0:
            a = -1
        else:
            a = jax.random.choice(actions[0], 1)[0]
        state, _, done = step(state=state, action=a)

        if state.turn[0] > 100:
            break
    assert state.turn[0] > 100
