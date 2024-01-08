import jax
import jax.numpy as jnp

from pgx.tic_tac_toe import TicTacToe

env = TicTacToe()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1


def test_step():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1
    assert state._x.color == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state._x.board == jnp.int32([-1, -1, -1, -1, -1, -1, -1, -1, -1])
    )
    assert not state.terminated
    # -1 -1 -1
    # -1 -1 -1
    # -1 -1 -1

    action = jnp.int32(4)
    state = step(state, action)
    assert state.current_player == 0
    assert state._x.color == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state._x.board == jnp.int32([-1, -1, -1, -1, 0, -1, -1, -1, -1])
    )
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    # -1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int32(0)
    state = step(state, action)
    assert state.current_player == 1
    assert state._x.color == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._x.board == jnp.int32([1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int32(1)
    state = step(state, action)
    assert state.current_player == 0
    assert state._x.color == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._x.board == jnp.int32([1, 0, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int32(8)
    state = step(state, action)
    assert state.current_player == 1
    assert state._x.color == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 0], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._x.board == jnp.int32([1, 0, -1, -1, 0, -1, -1, -1, 1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1  1

    action = jnp.int32(7)
    state = step(state, action)
    assert state.current_player == 0
    assert state._x.color == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._x.board == jnp.int32([1, 0, -1, -1, 0, -1, -1, 0, 1]))
    assert jnp.all(state.rewards == jnp.int32([-1, 1]))  # fmt: ignore
    assert state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1  0  1


def test_observe():
    state = init(jax.random.PRNGKey(1))
    obs = observe(state, state.current_player)
    init_obs = jnp.zeros([3, 3, 2])
    assert (obs == init_obs).all()

    state = step(state, 1)
    assert state.current_player == 0
    obs = observe(state, 0)
    assert (obs == init_obs.at[0, 1, 1].set(1)).all(), obs
    obs = observe(state, 1)
    assert (obs == init_obs.at[0, 1, 0].set(1)).all(), obs



def test_api():
    import pgx
    env = pgx.make("tic_tac_toe")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
