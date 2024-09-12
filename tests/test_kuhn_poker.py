import jax
import jax.numpy as jnp

from pgx.kuhn_poker import BET, PASS, KuhnPoker

env = KuhnPoker()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state._cards[0] != state._cards[1]
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()


def test_step():
    key = jax.random.PRNGKey(0)
    # cards = [2, 0]
    state = init(key)
    state = step(state, PASS)
    assert not state.terminated
    state = step(state, PASS)
    assert state.terminated
    assert (state.rewards == jnp.float32([1, -1])).all()

    state = init(key)
    state = step(state, PASS)
    assert not state.terminated
    state = step(state, BET)
    assert not state.terminated
    state = step(state, PASS)
    assert state.terminated
    assert (state.rewards == jnp.float32([-1, 1])).all()

    state = init(key)
    state = step(state, PASS)
    assert not state.terminated
    state = step(state, BET)
    assert not state.terminated
    state = step(state, BET)
    assert state.terminated
    assert (state.rewards == jnp.float32([2, -2])).all()

    state = init(key)
    state = step(state, BET)
    assert not state.terminated
    state = step(state, PASS)
    assert state.terminated
    assert (state.rewards == jnp.float32([1, -1])).all()

    state = init(key)
    state = step(state, BET)
    assert not state.terminated
    state = step(state, BET)
    assert state.terminated
    assert (state.rewards == jnp.float32([2, -2])).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)
    # cards = [2, 0]
    state = init(key)
    state = step(state, PASS)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, PASS)
    assert state.terminated

    state = init(key)
    state = step(state, PASS)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, BET)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, PASS)
    assert state.terminated

    state = init(key)
    state = step(state, PASS)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, BET)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, BET)
    assert state.terminated

    state = init(key)
    state = step(state, BET)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, PASS)
    assert state.terminated

    state = init(key)
    state = step(state, BET)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()
    state = step(state, BET)
    assert state.terminated


def test_observation():
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    state = init(key)
    """
    Player 0: K
    Player 1: J
    """
    state = step(state, BET)  # Player 0 bets 1 chip
    obs = observe(state, 0)
    assert (obs == jnp.bool_([0, 0, 1, 0, 1, 1, 0])).all()

    obs = observe(state, 1)
    assert (obs == jnp.bool_([1, 0, 0, 1, 0, 0, 1])).all()


def test_api():
    import pgx

    env = pgx.make("kuhn_poker")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
