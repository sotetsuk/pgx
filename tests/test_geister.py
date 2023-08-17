import jax
import jax.numpy as jnp
import numpy as np

from pgx._geister import Geister, State

env = Geister()
init = jax.jit(env.init)
#step = jax.jit(env.step)
#observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1
    num_my_good = 0
    num_my_bad = 0
    num_op_good = 0
    num_op_bad = 0
    for i in range(36):
        piece = state._board[i]
        if piece == jnp.int8(-2):
            num_op_bad += 1
        if piece == jnp.int8(-1):
            num_op_good += 1
        if piece == jnp.int8(2):
            num_my_bad += 1
        if piece == jnp.int8(1):
            num_my_good += 1
    assert num_op_bad == 4
    assert num_op_good == 4
    assert num_my_bad == 4
    assert num_my_good == 4
