import jax
import jax.numpy as jnp
import numpy as np

from pgx._geister import Geister, State, Action

env = Geister()
init = jax.jit(env.init)
step = jax.jit(env.step)
#observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert isinstance(state, State)
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


def test_step():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert isinstance(state, State)
    state = step(state, jnp.int32(43))
    assert state._board[9] == -2
    assert state._board[10] == 0
    state = step(state, jnp.int32(43))
    assert state._board[9] == -2
    assert state._board[10] == 0
    state = step(state, jnp.int32(44))
    assert state._board[8] == -2
    assert state._board[9] == 0
    assert state._num_bad_ghost[0] == 3

    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, jnp.int32(78))
    state = step(state, jnp.int32(43))
    state = step(state, jnp.int32(36))
    state = step(state, jnp.int32(8))
    state = step(state, jnp.int32(37))
    state = step(state, jnp.int32(43))
    state = step(state, jnp.int32(38))
    state = step(state, jnp.int32(8))
    state = step(state, jnp.int32(39))
    state = step(state, jnp.int32(43))
    state = step(state, jnp.int32(40))
    state = step(state, jnp.int32(8))
    state = step(state, jnp.int32(77))
    assert state.rewards[0] == 1
    assert state.terminated

    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, jnp.int32(37))
    state = step(state, jnp.int32(37))
    state = step(state, jnp.int32(38))
    state = step(state, jnp.int32(43))
    state = step(state, jnp.int32(111))
    state = step(state, jnp.int32(49))
    state = step(state, jnp.int32(117))
    state = step(state, jnp.int32(55))
    state = step(state, jnp.int32(123))
    state = step(state, jnp.int32(61))
    state = step(state, jnp.int32(129))
    state = step(state, jnp.int32(67))
    state = step(state, jnp.int32(135))
    assert state.terminated
    assert state._num_bad_ghost[0] == 0



