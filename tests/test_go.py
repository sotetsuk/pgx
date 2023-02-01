import jax
import jax.numpy as jnp
import numpy as np

from pgx.go import get_board, init, observe, step

BOARD_SIZE = 5
j_init = jax.jit(init, static_argnums=(1,))
j_step = jax.jit(step, static_argnums=(2,))


def test_init():
    rng = jax.random.PRNGKey(0)
    curr_player, _ = j_init(rng=rng, size=BOARD_SIZE)
    assert curr_player == 1


def test_end_by_pass():
    rng = jax.random.PRNGKey(0)

    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    assert state.passed
    assert not state.terminated
    _, state, _ = j_step(state=state, action=0, size=BOARD_SIZE)
    assert not state.passed
    assert not state.terminated
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    assert state.passed
    assert not state.terminated
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    assert state.passed
    assert state.terminated


def test_step():
    """
    https://www.cosumi.net/replay/?b=You&w=COSUMI&k=0&r=0&bs=5&gr=ccbccdcbdbbadabdbecaacabecaddeaettceedbetttt
    """
    rng = jax.random.PRNGKey(0)
    curr_player, state = j_init(rng=rng, size=BOARD_SIZE)

    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=17, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=7, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=16, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=21, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=10, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=14, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=15, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=23, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=20, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # pass
    _, state, _ = j_step(state=state, action=22, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=19, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=21, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # pass
    _, state, reward = j_step(state=state, action=25, size=BOARD_SIZE)  # pass

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

    # 同点なのでコミの分黒負け
    assert (reward == jnp.array([-1, 1])).all()


def test_kou():
    rng = jax.random.PRNGKey(0)

    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=17, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=7, size=BOARD_SIZE)  # WHITE

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

    _, state1, reward = j_step(
        state=state, action=12, size=BOARD_SIZE
    )  # BLACK
    # ルール違反により黒の負け
    assert state1.terminated
    assert (reward == jnp.array([-1, 1])).all()

    _, state2, _ = j_step(state=state, action=0, size=BOARD_SIZE)  # BLACK
    # 回避した場合
    assert not state2.terminated
    assert state2.kou == -1


def test_observe():
    rng = jax.random.PRNGKey(0)
    curr_player, state = j_init(rng=rng, size=BOARD_SIZE)
    # curr_player: 1
    # player 0 is white, player 1 is black

    _, state, _ = j_step(state=state, action=0, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=4, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=7, size=BOARD_SIZE)
    # ===========
    # + O + O @
    # O @ O + +
    # + + + + +
    # + + + + +
    # + + + + +
    # fmt: off
    expected_obs_p0 = jnp.array(
        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    # fmt: on
    assert (jax.jit(observe)(state, 0, False) == expected_obs_p0).all()

    # fmt: off
    expected_obs_p1 = jnp.array(
        [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    # fmt: on
    assert (jax.jit(observe)(state, 1, False) == expected_obs_p1).all()


def test_legal_action():
    rng = jax.random.PRNGKey(0)

    # =====
    # @ + @ + @
    # + @ + @ +
    # @ + @ + @
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        False, True, False, True, False,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=0, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=4, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=10, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=14, size=BOARD_SIZE)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + @ @ @ +
    # @ O + O @
    # + @ @ @ +
    # + + + + +
    # + + + + +
    # fmt:off
    expected = jnp.array([
        False, False, False, False, False,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, True, True, True,
        True, True, True, True, True, True])
    # fmt:on
    # white 8
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=9, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 13
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=9, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # black 9
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=9, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)  # BLACK
    assert jnp.all(state.legal_action_mask == expected)

    # =====
    # + + O + +
    # + O @ O +
    # O @ + @ O
    # + O @ O +
    # + + O + +
    # fmt:off
    expected_b = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, False, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    expected_w = jnp.array([
        True, True, False, True, True,
        True, False, False, False, True,
        False, False, True, False, False,
        True, False, False, False, True,
        True, True, False, True, True, True])
    # fmt:on
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=7, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=17, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=10, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=14, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=16, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=18, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=22, size=BOARD_SIZE)  # WHITE
    assert jnp.all(state.legal_action_mask == expected_b)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)  # BLACK
    assert jnp.all(state.legal_action_mask == expected_w)

    # =====
    # + @ @ @ +
    # @ O @ + @
    # @ O @ O @
    # @ O @ O @
    # @ O O O @
    # fmt:off
    # black 24
    expected_w1 = jnp.array([
        True, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # white pass
    expected_b = jnp.array([
        True, False, False, False, True,
        False, False, False, True, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False, True])
    # black 8
    expected_w2 = jnp.array([
        False, False, False, False, False,
        False, True, False, False, False,
        False, True, False, True, False,
        False, True, False, True, False,
        False, True, True, True, False, True])
    # fmt:on
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=1, size=BOARD_SIZE)  # BLACK
    _, state, _ = j_step(state=state, action=6, size=BOARD_SIZE)  # WHITE
    _, state, _ = j_step(state=state, action=2, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=11, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=3, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=13, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=5, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=16, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=7, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=18, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=9, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=21, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=10, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=22, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=12, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=23, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=14, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=15, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=17, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=19, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=24, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    _, state, _ = j_step(state=state, action=20, size=BOARD_SIZE)
    assert jnp.all(state.legal_action_mask == expected_w1)
    _, state, _ = j_step(state=state, action=25, size=BOARD_SIZE)
    assert jnp.all(state.legal_action_mask == expected_b)
    _, state, _ = j_step(state=state, action=8, size=BOARD_SIZE)
    assert jnp.all(state.legal_action_mask == expected_w2)

    # =====
    # random
    _, state = j_init(rng=rng, size=BOARD_SIZE)
    for _ in range(100):
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = BOARD_SIZE * BOARD_SIZE
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])
        for action in actions[0][:-1]:
            _, _state, _ = j_step(state=state, action=action, size=BOARD_SIZE)
            assert not _state.terminated

        _, state, _ = j_step(state=state, action=a, size=BOARD_SIZE)


def test_random_play_5():
    rng = jax.random.PRNGKey(0)
    curr_player, state = j_init(rng=rng, size=BOARD_SIZE)
    while not state.terminated:
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = 25
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])

        _, state, _ = j_step(state=state, action=a, size=BOARD_SIZE)

        if state.turn > 100:
            break
    assert state.passed or state.turn > 100


def test_random_play_19():
    BOARD_SIZE = 19

    rng = jax.random.PRNGKey(0)
    curr_player, state = j_init(rng=rng, size=BOARD_SIZE)
    while not state.terminated:
        actions = np.where(state.legal_action_mask)
        if len(actions[0]) == 0:
            a = 19 * 19
        else:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            a = jax.random.choice(subkey, actions[0])

        _, state, _ = j_step(state=state, action=a, size=BOARD_SIZE)

        if state.turn > 100:
            break
    assert state.passed or state.turn > 100
