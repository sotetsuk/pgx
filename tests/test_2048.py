import jax
import jax.numpy as jnp
from pgx._2048 import Play2048, _slide_and_merge

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
    line = jnp.array([0, 2, 0, 2])
    assert (slide_and_merge(line) == jnp.array([3, 0, 0, 0])).all()

    line = jnp.array([0, 2, 0, 1])
    assert (slide_and_merge(line) == jnp.array([2, 1, 0, 0])).all()

    line = jnp.array([2, 2, 2, 2])
    assert (slide_and_merge(line) == jnp.array([3, 3, 0, 0])).all()

    line = jnp.array([2, 0, 0, 2])
    assert (slide_and_merge(line) == jnp.array([3, 0, 0, 0])).all()

    line = jnp.array([1, 4, 4, 5])
    assert (slide_and_merge(line) == jnp.array([1, 5, 5, 0])).all()

    board = jnp.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    board_2d = board.reshape((4, 4))
    board_2d = jax.vmap(_slide_and_merge)(board_2d)
    board_1d = board_2d.ravel()
    assert (
        board_1d == jnp.array([3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    ).all()
