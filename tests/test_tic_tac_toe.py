import jax
import jax.numpy as jnp

from pgx.tic_tac_toe import init, step


def test_init():
    rng = jax.random.PRNGKey(0)
    curr_player, state = init(rng=rng)
    assert curr_player == 1


def test_step():
    rng = jax.random.PRNGKey(0)
    curr_player, state = init(rng=rng)
    assert curr_player == 1
    assert state.turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([-1, -1, -1, -1, -1, -1, -1, -1, -1]))
    assert not state.terminated
    # -1 -1 -1
    # -1 -1 -1
    # -1 -1 -1

    action = jnp.int8(4)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state.turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([-1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    # -1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(0)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 1
    assert state.turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([0, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(1)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state.turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(8)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 1
    assert state.turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 0], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, 1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1  1

    action = jnp.int8(7)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state.turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 0, 0], jnp.bool_))  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, 0, 1]))
    assert jnp.all(rewards == jnp.int16([-1, 1]))  # fmt: ignore
    assert state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1  0  1
