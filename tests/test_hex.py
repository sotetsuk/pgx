import jax
import jax.numpy as jnp
from pgx.hex import Hex

env = Hex()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key=key)
    assert state.current_player == 0


def test_merge():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, 0)
    state = step(state, 11)
    state = step(state, 1)
    state = step(state, 12)
    state = step(state, 3)
    state = step(state, 13)
    state = step(state, 2)
    state = step(state, 22)
    # fmt: off
    expected = jnp.int32([
          3,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,
        -23, -23, -23,   0,   0,   0,   0,   0,   0,   0,   0,
        -23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])
    # fmt:on
    assert jnp.all(state._board == expected)


def test_swap():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert ~state.legal_action_mask[-1]
    state = step(state, 1)
    state.save_svg("tests/assets/hex/swap_01.svg")
    assert (state._board != 0).sum() == 1
    assert state._board[1] == -2
    assert state.legal_action_mask[-1]
    state = step(state, 121)  # swap!
    state.save_svg("tests/assets/hex/swap_02.svg")
    assert (state._board != 0).sum() == 1
    assert state._board[11] == -12
    assert ~state.legal_action_mask[-1]

    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, 0)
    state.save_svg("tests/assets/hex/swap_03.svg")
    assert (state._board != 0).sum() == 1
    assert state._board[0] == -1
    assert state.legal_action_mask[-1]
    state = step(state, 121)  # swap!
    state.save_svg("tests/assets/hex/swap_04.svg")
    assert (state._board != 0).sum() == 1
    assert state._board[0] == -1
    assert ~state.legal_action_mask[-1]

    key = jax.random.PRNGKey(0)
    state = init(key=key)
    state = step(state, 1)
    state = step(state, 121)  # swap!
    for i in range(10):
        state = step(state, 0 + i)
        state = step(state, 12 + i)
    state.save_svg("tests/assets/hex/swap_05.svg")
    assert state.terminated


def test_terminated():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert not state.terminated
    for i in range(11):
        state = step(state, i * 11)
        state = step(state, i * 11 + 1)
    assert state.terminated

    state = init(key=key)
    for i in range(10):
        state = step(state, i)
        state = step(state, i + 11)
    state = step(state, 120)
    state = step(state, 21)
    assert state.terminated

    state = init(key=key)
    for i in range(10):
        state = step(state, i)
        state = step(state, i + 11)
    state = step(state, 10)
    assert not state.terminated


def test_reward():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1
    assert (state.rewards == jnp.float32([0.0, 0.0])).all()

    for i in range(10):
        state = step(state, i * 11)
        state = step(state, i * 11 + 1)
    state = step(state, 110)
    assert (state.rewards == jnp.float32([-1.0, 1.0])).all()

    state = init(key=key)
    for i in range(10):
        state = step(state, i)
        state = step(state, i + 11)
    state = step(state, 120)
    state = step(state, 21)
    assert (state.rewards == jnp.float32([1.0, -1.0])).all()


def test_observe():
    """
    @ O . . .
     . . . . .
      . . . . .
    """
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key=key)
    assert state.current_player == 0
    assert state.observation[:, :, 0].sum() == 0
    assert state.observation[:, :, 1].sum() == 0
    assert state.observation[:, :, 2].sum() == 0
    assert state.observation[:, :, 3].sum() == 0
    assert state.observation.sum() == 0
    assert (jnp.zeros((11, 11, 4)) == observe(state, 0)).all()
    assert (state.observation[:, :, -1] == 0).all()
    state = step(state, 0)
    assert (observe(state, 0)[:, :, 2] == 0).all()
    assert (observe(state, 1)[:, :, 2] == 1).all()
    assert (state.observation[:, :, -1] == 1).all()
    state = step(state, 1)
    assert (
        jnp.zeros((11, 11, 4), dtype=jnp.bool_).at[0, 0, 0].set(True).at[0, 1, 1].set(True)
        == observe(state, 0)
    ).all()
    assert (
            jnp.zeros((11, 11, 4), dtype=jnp.bool_).at[0, 1, 0].set(True).at[0, 0, 1].set(True).at[:, :, 2].set(True)
        == observe(state, 1)
    ).all()
    assert (state.observation[:, :, -1] == 0).all()


def test_api():
    import pgx
    env = pgx.make("hex")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
