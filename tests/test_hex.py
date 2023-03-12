import jax

from pgx.hex import Hex, init

env = Hex()
init = jax.jit(env.init)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.curr_player == 1
