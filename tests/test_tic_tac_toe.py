import jax
import jax.numpy as jnp

from pgx.tic_tac_toe import _win_check, init, observe, step

init = jax.jit(init)
step = jax.jit(step)
observe = jax.jit(observe)


def test_init():
    rng = jax.random.PRNGKey(0)
    state = init(rng=rng)
    assert state.curr_player == 1


def test_step():
    rng = jax.random.PRNGKey(0)
    state = init(rng=rng)
    assert state.curr_player == 1
    assert state.turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state.board == jnp.int8([-1, -1, -1, -1, -1, -1, -1, -1, -1])
    )
    assert not state.terminated
    # -1 -1 -1
    # -1 -1 -1
    # -1 -1 -1

    action = jnp.int8(4)
    state = step(state, action)
    assert state.curr_player == 0
    assert state.turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state.board == jnp.int8([-1, -1, -1, -1, 0, -1, -1, -1, -1])
    )
    assert jnp.all(state.reward == 0)  # fmt: ignore
    assert not state.terminated
    # -1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(0)
    state = step(state, action)
    assert state.curr_player == 1
    assert state.turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.reward == 0)  # fmt: ignore
    assert not state.terminated
    #  1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(1)
    state = step(state, action)
    assert state.curr_player == 0
    assert state.turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.reward == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(8)
    state = step(state, action)
    assert state.curr_player == 1
    assert state.turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 0], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, 1]))
    assert jnp.all(state.reward == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1  1

    action = jnp.int8(7)
    state = step(state, action)
    assert state.curr_player == 0
    assert state.turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state.board == jnp.int8([1, 0, -1, -1, 0, -1, -1, 0, 1]))
    assert jnp.all(state.reward == jnp.int16([-1, 1]))  # fmt: ignore
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
        state = init(sub_key)
        rewards = jnp.int16([0.0, 0.0])
        while not done:
            assert jnp.all(rewards == 0), state.board
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, sub_key = jax.random.split(key)
            action = jax.random.choice(sub_key, legal_actions)
            state = step(state, action)
            done = state.terminated
            rewards += state.reward


def test_win_check():
    board = jnp.int8([-1, -1, -1, -1, -1, -1, -1, -1, -1])
    turn = jnp.int8(1)
    assert not _win_check(board, turn)

    board = jnp.int8([1, -1, -1, -1, 1, -1, 0, -1, 0])
    turn = jnp.int8(1)
    assert not _win_check(board, turn)

    board = jnp.int8([1, -1, -1, -1, 1, -1, -1, -1, 1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, -1, 1, -1, 1, -1, 1, -1, -1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([1, 1, 1, -1, -1, -1, -1, -1, -1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, -1, -1, 1, 1, 1, -1, -1, -1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, -1, -1, -1, -1, -1, 1, 1, 1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([1, -1, -1, 1, -1, -1, 1, -1, -1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, 1, -1, -1, 1, -1, -1, 1, -1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, -1, 1, -1, -1, 1, -1, -1, 1])
    turn = jnp.int8(1)
    assert _win_check(board, turn)

    board = jnp.int8([-1, 0, -1, -1, 0, -1, -1, 0, -1])
    turn = jnp.int8(0)
    assert _win_check(board, turn)


def test_observe():
    state = init(jax.random.PRNGKey(1))
    obs = observe(state, state.curr_player)
    init_obs = jnp.zeros(27).at[:9].set(1)
    assert (obs == init_obs).all()

    state = step(state, 0)
    obs = observe(state, 0)
    assert (obs == init_obs.at[0].set(0).at[18].set(1)).all(), obs
    obs = observe(state, 1)
    assert (obs == init_obs.at[0].set(0).at[9].set(1)).all(), obs
