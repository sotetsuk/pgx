import jax
import jax.numpy as jnp
from pgx.connect_four import ConnectFour

env = ConnectFour()
init = jax.jit(env.init)
step = jax.jit(env.step)


def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1


def test_step():
    key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    for _ in range(6):
        state = step(state, 0)
    for _ in range(6):
        state = step(state, 1)

    """
    OO.....
    @@.....
    OO.....
    @@.....
    OO.....
    @@.....
    """
    # fmt: off
    assert (state.board == jnp.array(
        [1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1,
         1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1,
         1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1])).all()
    # fmt:on


def test_legal_action():
    key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    for _ in range(6):
        state = step(state, 0)
    for _ in range(6):
        state = step(state, 1)
    for _ in range(6):
        state = step(state, 6)

    assert (state.legal_action_mask == jnp.array([0, 0, 1, 1, 1, 1, 0])).all()
