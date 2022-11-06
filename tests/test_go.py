import jax
import jax.numpy as jnp

from pgx.go import get_board, init, step

BOARD_SIZE = 5


def test_end_by_pass():
    state = init(BOARD_SIZE)

    _, state, _ = step(state=state, action=-1, size=BOARD_SIZE)
    assert state.passed
    assert not state.terminated
    _, state, _ = step(state=state, action=0, size=BOARD_SIZE)
    assert not state.passed
    assert not state.terminated
    _, state, _ = step(state=state, action=-1, size=BOARD_SIZE)
    assert state.passed
    assert not state.terminated
    _, state, _ = step(state=state, action=-1, size=BOARD_SIZE)
    assert state.passed
    assert state.terminated


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    state = init(size=BOARD_SIZE)
    _, state, _ = step(state=state, action=12, size=BOARD_SIZE)  # BLACK
    _, state, _ = step(state=state, action=11, size=BOARD_SIZE)  # WHITE
    _, state, _ = step(state=state, action=17, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=7, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=1, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=16, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=21, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=10, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=14, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=15, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=23, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=20, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=-1, size=BOARD_SIZE)  # pass
    _, state, _ = step(state=state, action=22, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=19, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=21, size=BOARD_SIZE)
    _, state, _ = step(state=state, action=-1, size=BOARD_SIZE)  # pass
    _, state, reward = step(state=state, action=-1, size=BOARD_SIZE)  # pass

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
    assert state.terminated
    assert (reward == jnp.array([0, 0])).all()


def test_kou():
    state = init(size=BOARD_SIZE)
    _, state, _ = step(state=state, action=2, size=BOARD_SIZE)  # BLACK
    _, state, _ = step(state=state, action=17, size=BOARD_SIZE)  # WHITE
    _, state, _ = step(state=state, action=6, size=BOARD_SIZE)  # BLACK
    _, state, _ = step(state=state, action=13, size=BOARD_SIZE)  # WHITE
    _, state, _ = step(state=state, action=8, size=BOARD_SIZE)  # BLACK
    _, state, _ = step(state=state, action=11, size=BOARD_SIZE)  # WHITE
    _, state, _ = step(state=state, action=12, size=BOARD_SIZE)  # BLACK
    _, state, _ = step(state=state, action=7, size=BOARD_SIZE)  # WHITE

    """
    ===========
    + + @ + +
    + @ + @ +
    + O @ O +
    + + O + +
    + + + + +
    ===========
    + + @ + +
    + @ O @ +
    + O + O +
    + + O + +
    + + + + +
    """
    assert state.kou == 12

    _, state1, reward = step(state=state, action=12, size=BOARD_SIZE)  # BLACK
    # ルール違反により黒の負け
    assert state1.terminated
    assert (reward == jnp.array([-1, 1])).all()

    _, state2, _ = step(state=state, action=0, size=BOARD_SIZE)  # BLACK
    # 回避した場合
    assert not state2.terminated
    assert state2.kou == -1


def test_random_play():
    import numpy as np

    state = init(size=BOARD_SIZE)
    while not state.terminated:
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = -1
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])

        _, state, _ = step(state=state, action=a, size=BOARD_SIZE)

        if state.turn > 100:
            break
    assert state.turn > 100


def test_random_play_19():
    import numpy as np

    BOARD_SIZE = 19

    state = init(size=BOARD_SIZE)
    while not state.terminated:
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = -1
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])

        _, state, _ = step(state=state, action=a, size=BOARD_SIZE)

        if state.turn > 100:
            break
    assert state.turn > 100
