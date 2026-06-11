import jax
import jax.numpy as jnp
from pgx.othello import Othello

env = Othello()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def _init_with_current_player(player: int):
    """Return an initial state whose current_player is `player`.

    The starting player is decided by an RNG coin flip in `_init`, and the exact
    flip outcome for a given key is not stable across JAX versions. Searching keys
    keeps these tests deterministic and version-agnostic without depending on a
    specific `jax.random` implementation.
    """
    for seed in range(1000):
        state = init(jax.random.PRNGKey(seed))
        if int(state.current_player) == player:
            return state
    raise AssertionError(f"no key produced current_player={player}")


def test_init():
    # Both starting players are reachable, and current_player is always 0 or 1.
    players = {int(_init_with_current_player(p).current_player) for p in (0, 1)}
    assert players == {0, 1}


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
    expected = jnp.int32([
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 1, 1, -1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
    # fmt:on
    assert jnp.all(state._x.board == expected)


def test_terminated():
    # wipe out
    state = _init_with_current_player(0)
    for i in [37, 43, 34, 29, 52, 45, 38, 44]:
        state = step(state, i)
        assert not state.terminated
    state = step(state, 20)
    assert state.terminated
    assert (state.rewards == jnp.float32([1.0, -1.0])).all()


def test_legal_action():
    # cannot put
    state = _init_with_current_player(0)
    assert state.current_player == 0
    for i in [37, 29, 18, 44, 53, 46, 30, 60, 62, 38, 39]:
        state = step(state, i)
    assert ~state.legal_action_mask[:64].any()
    assert state.legal_action_mask[64]

    state = step(state, 64)
    assert ~state.legal_action_mask[:64].any()
    assert state.legal_action_mask[64]
    assert not state.terminated

    state = step(state, 64)
    assert state.terminated


def test_observe():
    state = _init_with_current_player(0)
    assert state.current_player == 0

    obs = observe(state, state.current_player)
    assert obs.shape == (8, 8, 2)

    state = step(state, 37)
    """
    ........
    ........
    ........
    ...O@...
    ...@@@..
    ........
    ........
    """
    obs = observe(state, 0)
    assert obs[3, 4, 0]
    assert obs[4, 3, 0]
    assert obs[4, 4, 0]
    assert obs[4, 5, 0]
    assert obs[3, 3, 1]
    assert not obs[0, 0, 0]

    state = step(state, 29)
    """
    ........
    ........
    ........
    ...OOO..
    ...@@@..
    ........
    ........
    """
    obs = observe(state, 1)
    assert obs[3, 3, 0]
    assert obs[3, 4, 0]
    assert obs[3, 5, 0]
    assert obs[4, 3, 1]
    assert obs[4, 4, 1]
    assert obs[4, 5, 1]
    assert not obs[0, 0, 0]


def test_api():
    import pgx

    env = pgx.make("othello")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
