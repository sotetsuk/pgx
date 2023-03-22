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
    assert jnp.count_nonzero(state.board == 1) == 2


def test_slide_and_merge():
    line = jnp.int8([0, 2, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int8([3, 0, 0, 0])).all()

    line = jnp.int8([0, 2, 0, 1])
    assert (slide_and_merge(line)[0] == jnp.int8([2, 1, 0, 0])).all()

    line = jnp.int8([2, 2, 2, 2])
    assert (slide_and_merge(line)[0] == jnp.int8([3, 3, 0, 0])).all()

    line = jnp.int8([2, 0, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int8([3, 0, 0, 0])).all()

    line = jnp.int8([1, 4, 4, 5])
    assert (slide_and_merge(line)[0] == jnp.int8([1, 5, 5, 0])).all()

    board = jnp.int8([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    board_2d = board.reshape((4, 4))
    board_2d = jax.vmap(_slide_and_merge)(board_2d)[0]
    board_1d = board_2d.ravel()
    assert (
        board_1d == jnp.int8([3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    ).all()


def test_step():
    key = jax.random.PRNGKey(0)
    state = init(key)
    """
    [[0 0 0 0]
     [0 0 2 0]
     [0 0 0 0]
     [0 0 2 0]]
    """
    assert (
        state.board
        == jnp.int8([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    ).all()

    state = step(state, 3)  # down
    """
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 2]
     [0 0 4 0]]
    """
    assert (
        state.board
        == jnp.int8([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0])
    ).all()


def test_legal_action():
    board = jnp.int8([0, 1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 0])
    state = State(board=board)
    state = step(state, 0)
    """
    [[ 2  4  8  2]
     [ 4  8 16 32]
     [ 8 16 32 64]
     [16 32 64  0]]
    """
    assert (state.legal_action_mask == jnp.bool_([0, 0, 1, 1])).all()
    assert not state.terminated


def test_terminated():
    board = jnp.int8([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 4, 5, 6])
    state = State(board=board)
    state = step(state, 0)
    """
    [[ 2  4  8 16]
     [ 4  8 16 32]
     [ 8 16 32 64]
     [16 32 64  2]]
    """
    assert state.terminated


def test_observe():
    key = jax.random.PRNGKey(2)
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


def test_random_play():
    key = jax.random.PRNGKey(0)
    done = jnp.bool_(False)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    while not done:
        legal_actions = jnp.where(state.legal_action_mask)[0]
        key, sub_key = jax.random.split(key)
        action = jax.random.choice(sub_key, legal_actions)
        state = step(state, jnp.int16(action))
        done = state.terminated


def test_api():
    import pgx

    env = pgx.make("2048")
    pgx.api_test(env, 10)
