import jax
import jax.numpy as jnp

from pgx.tic_tac_toe import _win_check, TicTacToe

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
    assert state._turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state._board == jnp.int8([-1, -1, -1, -1, -1, -1, -1, -1, -1])
    )
    assert not state.terminated
    # -1 -1 -1
    # -1 -1 -1
    # -1 -1 -1

    action = jnp.int8(4)
    state = step(state, action)
    assert state.current_player == 0
    assert state._turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(
        state._board == jnp.int8([-1, -1, -1, -1, 0, -1, -1, -1, -1])
    )
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    # -1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(0)
    state = step(state, action)
    assert state.current_player == 1
    assert state._turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 1, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, -1, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1 -1 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(1)
    state = step(state, action)
    assert state.current_player == 0
    assert state._turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, -1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1 -1

    action = jnp.int8(8)
    state = step(state, action)
    assert state.current_player == 1
    assert state._turn == 0
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([0, 0, 1, 1, 0, 1, 1, 1, 0], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, -1, 1]))
    assert jnp.all(state.rewards == 0)  # fmt: ignore
    assert not state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1 -1  1

    action = jnp.int8(7)
    state = step(state, action)
    assert state.current_player == 0
    assert state._turn == 1
    assert jnp.all(
        state.legal_action_mask
        == jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], jnp.bool_)
    )  # fmt: ignore
    assert jnp.all(state._board == jnp.int8([1, 0, -1, -1, 0, -1, -1, 0, 1]))
    assert jnp.all(state.rewards == jnp.int16([-1, 1]))  # fmt: ignore
    assert state.terminated
    #  1  0 -1
    # -1  0 -1
    # -1  0  1


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
    obs = observe(state, state.current_player)
    init_obs = jnp.zeros([3, 3, 2])
    assert (obs == init_obs).all()

    state = step(state, 1)
    assert state.current_player == 0
    obs = observe(state, 0)
    assert (obs == init_obs.at[0, 1, 1].set(1)).all(), obs
    obs = observe(state, 1)
    assert (obs == init_obs.at[0, 1, 0].set(1)).all(), obs


def test_init_with_first_player():
    import pgx
    from pgx.experimental.utils import act_randomly
    env = pgx.make("tic_tac_toe")
    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    first_player = jnp.arange(10) % 2

    state1 = jax.jit(jax.vmap(env.init))(keys)
    state2 = jax.jit(jax.vmap(env.init_with_first_player))(keys, first_player)  # type: ignore
    assert (state1.observation == state2.observation).all()
    assert (state1.current_player != state2.current_player).any()
    assert (state2.current_player == jnp.arange(10) % 2).all()
    action = jax.jit(act_randomly)(jax.random.PRNGKey(0), state2)
    state = jax.jit(jax.vmap(env.step))(state2, action)
    assert (state.current_player == (jnp.arange(10) + 1) % 2).all()


def test_api():
    import pgx
    env = pgx.make("tic_tac_toe")
    pgx.v1_api_test(env, 10)
