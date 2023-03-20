import jax
import jax.numpy as jnp
from pgx.othello import Othello

env = Othello()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 0


def test_step():
    key = jax.random.PRNGKey(0)
    state = init(key)
    state = step(state, 19)
    state = step(state, 18)
    state = step(state, 26)
    state = step(state, 20)
    state = step(state, 21)
    state = step(state, 34)
    state = step(state, 17)
    # fmt: off
    expected = jnp.int8([
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 1, 1, -1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
    # fmt:on
    assert jnp.all(state.board == expected)


def test_terminated():
    # wipe out
    key = jax.random.PRNGKey(0)
    state = init(key)
    print(state.current_player)
    for i in [37, 43, 34, 29, 52, 45, 38, 44]:
        state = step(state, i)
        assert not state.terminated
    state = step(state, 20)
    assert state.terminated
    assert (state.reward == jnp.float32([1.0, -1.0])).all()

    # cannot put
    state = init(key)
    print(state.current_player)
    for i in [37, 29, 18, 44, 53, 46, 30, 60, 62, 38]:
        state = step(state, i)
        assert not state.terminated
    state = step(state, 39)
    assert state.terminated
    assert (state.reward == jnp.float32([1.0, -1.0])).all()


def test_random_play():
    key = jax.random.PRNGKey(0)
    done = jnp.bool_(False)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    rewards = jnp.int16([0.0, 0.0])
    while not done:
        legal_actions = jnp.where(state.legal_action_mask)[0]
        key, sub_key = jax.random.split(key)
        action = jax.random.choice(sub_key, legal_actions)
        state = step(state, jnp.int16(action))
        done = state.terminated
        rewards += state.reward
