import jax
import jax.numpy as jnp
from pgx._leduc_holdem import LeducHoldem, CALL, RAISE, FOLD

env = LeducHoldem()
init = jax.jit(env.init)
step = jax.jit(env.step)
# observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 0
    assert (state.legal_action_mask == jnp.bool_([1, 1, 0, 0])).all()


def test_step():
    key = jax.random.PRNGKey(1)

    state = init(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, CALL)
    state = step(state, RAISE)  # +2(3)
    state = step(state, CALL)

    # second round
    state = step(state, CALL)
    state = step(state, RAISE)  # +4(7)
    state = step(state, RAISE)  # +4(11)
    assert not state.terminated
    state = step(state, CALL)
    assert state.terminated
    assert (state.reward == jnp.float32([11, -11])).all()

    state = init(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, CALL)
    state = step(state, RAISE)  # +2(3)
    state = step(state, CALL)

    # second round
    state = step(state, CALL)
    state = step(state, RAISE)  # +4(7)
    state = step(state, RAISE)  # +4(11)
    assert not state.terminated
    state = step(state, FOLD)
    assert state.terminated
    assert (state.reward == jnp.float32([-7, 7])).all()


def test_random_play():
    N = 100
    key = jax.random.PRNGKey(0)
    for _ in range(N):
        done = jnp.bool_(False)
        key, sub_key = jax.random.split(key)
        state = init(sub_key)
        while not done:
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, sub_key = jax.random.split(key)
            action = jax.random.choice(sub_key, legal_actions)
            state = step(state, action)
            done = state.terminated


# def test_api():
#    import pgx
#
#    env = pgx.make("leduc_holdem")
#    pgx.api_test(env, 10)
