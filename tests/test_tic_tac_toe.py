import jax
import jax.numpy as jnp

from pgx.tic_tac_toe import init, step, _win_check

init = jax.jit(init)
step = jax.jit(step)


def test_init():
    rng = jax.random.PRNGKey(0)
    curr_player, state = init(rng=rng)
    assert curr_player == 1


def test_step():
    rng = jax.random.PRNGKey(0)
    curr_player, state = init(rng=rng)
    assert curr_player == 1
    assert state._turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([-1, -1, -1, -1, -1, -1, -1, -1, -1]))
    assert not state.terminated
    # -1 -1 -1
    # -1 -1 -1
    # -1 -1 -1

    action = jnp.int8(4)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state._turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([-1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    # -1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(0)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 1
    assert state._turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([0, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(1)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state._turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 1], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(8)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 1
    assert state._turn == 0
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 0], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, 1]))
    assert jnp.all(rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1  1

    action = jnp.int8(7)
    curr_player, state, rewards = step(state, action)
    assert curr_player == 0
    assert state._turn == 1
    assert jnp.all(state.legal_action_mask == jnp.array([0, 0, 1, 1, 0, 1, 1, 0, 0], jnp.bool_))  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, 0, 1]))
    assert jnp.all(rewards == jnp.int16([-1, 1]))  # fmt: ignore
    assert state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1  0  1


def test_random_play():
    N = 1000
    key = jax.random.PRNGKey(0)
    for i in range(N):
        done = jnp.bool_(False)
        key, sub_key = jax.random.split(key)
        curr_player, state = init(sub_key)
        rewards = jnp.int16([0., 0.])
        while not done:
            assert jnp.all(rewards == 0), state._board
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, sub_key = jax.random.split(key)
            action = jax.random.choice(sub_key, legal_actions)
            curr_player, state, rewards = step(state, action)
            done = state.terminated


def test_win_check():
    board = jnp.int8(
        [-1, -1, -1,
         -1, -1, -1,
         -1, -1, -1]
    )
    turn = jnp.int8(1)
    assert not _win_check(board, turn)

    board = jnp.int8(
        [1, -1, -1,
         -1, 1, -1,
         0, -1, 0]
    )
    turn = jnp.int8(1)
    assert not _win_check(board, turn)

    board = jnp.int8(
        [1, -1, -1,
         -1, 1, -1,
         -1, -1, 1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, -1, 1,
         -1, 1, -1,
         1, -1, -1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [1, 1, 1,
         -1, -1, -1,
         -1, -1, -1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, -1, -1,
          1, 1, 1,
         -1, -1, -1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, -1, -1,
         -1, -1, -1,
         1, 1, 1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [1, -1, -1,
         1, -1, -1,
         1, -1, -1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, 1, -1,
         -1, 1, -1,
         -1, 1, -1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, -1, 1,
         -1, -1, 1,
         -1, -1, 1]
    )
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8(
        [-1, 0, -1,
         -1, 0, -1,
         -1, 0, -1]
    )
    turn = jnp.int8(0)
    assert _win_check(board, turn)
