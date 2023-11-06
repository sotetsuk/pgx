import jax
import jax.numpy as jnp
from pgx.play2048 import Play2048, _slide_and_merge, State

env = Play2048()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
slide_and_merge = jax.jit(_slide_and_merge)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert jnp.count_nonzero(state._board == 1) == 2
    key = jax.random.PRNGKey(2)
    _, key = jax.random.split(key)  # for test compatibility
    state = init(key=key)
    assert state.legal_action_mask.shape == (4,)
    assert (state.legal_action_mask == jnp.bool_([1, 0, 1, 1])).all()


def test_slide_and_merge():
    line = jnp.int32([0, 2, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 0, 0, 0])).all()

    line = jnp.int32([0, 2, 0, 1])
    assert (slide_and_merge(line)[0] == jnp.int32([2, 1, 0, 0])).all()

    line = jnp.int32([2, 2, 2, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 3, 0, 0])).all()

    line = jnp.int32([2, 0, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 0, 0, 0])).all()

    line = jnp.int32([1, 4, 4, 5])
    assert (slide_and_merge(line)[0] == jnp.int32([1, 5, 5, 0])).all()

    board = jnp.int32([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    board_2d = board.reshape((4, 4))
    board_2d = jax.vmap(_slide_and_merge)(board_2d)[0]
    board_1d = board_2d.ravel()
    assert (
        board_1d == jnp.int32([3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    ).all()


def test_step():
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    state = init(key)
    """
    [[0 0 0 0]
     [0 0 2 0]
     [0 0 0 0]
     [0 0 2 0]]
    """
    assert (
        state._board
        == jnp.int32([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    ).all()

    key1, key2 = jax.random.split(key)
    state1 = step(state, 3, key1)  # down
    state2 = step(state, 3, key2)  # down
    assert state1._board[14] == 2
    assert state2._board[14] == 2
    assert not (state1._board == state2._board).all()


def test_legal_action():
    board = jnp.int32([0, 1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 0])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(0))
    """
    [[ 2  4  8  2]
     [ 4  8 16 32]
     [ 8 16 32 64]
     [16 32 64  0]]
    """
    assert (state.legal_action_mask == jnp.bool_([0, 0, 1, 1])).all()
    assert not state.terminated
    
    board = jnp.int32([2, 2, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(3))
    """
    [[ 8  0  0  0]
     [ 8  2  0  0]
     [ 8  0  0  0]
     [ 8  0  0  0]]
    """
    assert (state.legal_action_mask == jnp.bool_([0, 1, 1, 1])).all()
    assert not state.terminated


def test_terminated():
    board = jnp.int32([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 4, 5, 6])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(0))
    """
    [[ 2  4  8 16]
     [ 4  8 16 32]
     [ 8 16 32 64]
     [16 32 64  2]]
    """
    assert state.terminated


def test_observe():
    key = jax.random.PRNGKey(2)
    _, key = jax.random.split(key)  # due to API update
    state = init(key)
    """
    [[0 0 2 2]
     [0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    """
    obs = observe(state, 0)
    assert obs.shape == (4, 4, 31)

    assert not obs[0, 2, 0]
    assert obs[0, 2, 1]
    assert not obs[0, 2, 2]

    assert not obs[0, 3, 0]
    assert obs[0, 3, 1]
    assert not obs[0, 3, 2]


def test_api():
    import pgx
    env = pgx.make("2048")
    pgx.api_test(env, 3, use_key=True)
