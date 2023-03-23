import jax
import jax.numpy as jnp
from pgx._khun_poker import KhunPoker, CALL, BET, FOLD, CHECK

env = KhunPoker()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.cards[0] != state.cards[1]
    assert (state.legal_action_mask == jnp.bool_([0, 1, 0, 1])).all()


def test_step():
    key = jax.random.PRNGKey(0)
    # cards = [2, 0]
    state = init(key)
    state = step(state, CHECK)
    state = step(state, CHECK)
    assert state.terminated
    assert (state.reward == jnp.float32([1, -1])).all()

    state = init(key)
    state = step(state, CHECK)
    state = step(state, BET)
    state = step(state, FOLD)
    assert state.terminated
    assert (state.reward == jnp.float32([-1, 1])).all()

    state = init(key)
    state = step(state, CHECK)
    state = step(state, BET)
    state = step(state, CALL)
    assert state.terminated
    assert (state.reward == jnp.float32([2, -2])).all()

    state = init(key)
    state = step(state, BET)
    state = step(state, FOLD)
    assert state.terminated
    assert (state.reward == jnp.float32([1, -1])).all()

    state = init(key)
    state = step(state, BET)
    state = step(state, CALL)
    assert state.terminated
    assert (state.reward == jnp.float32([2, -2])).all()


# def test_api():
#    import pgx
#
#    env = pgx.make("khun_poker")
#    pgx.api_test(env, 10)
