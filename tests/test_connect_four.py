import jax
import jax.numpy as jnp
from pgx.connect_four import ConnectFour

env = ConnectFour()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1


def test_step():
    key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    for _ in range(6):
        state = step(state, 0)
    for _ in range(6):
        state = step(state, 1)

    """
    OO.....
    @@.....
    OO.....
    @@.....
    OO.....
    @@.....
    """
    # fmt: off
    assert (state.board == jnp.array(
        [1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1,
         1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1,
         1, 1, -1, -1, -1, -1, -1,
         0, 0, -1, -1, -1, -1, -1])).all()
    # fmt:on


def test_legal_action():
    key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    for _ in range(6):
        state = step(state, 0)
    for _ in range(6):
        state = step(state, 1)
    for _ in range(6):
        state = step(state, 6)

    assert (state.legal_action_mask == jnp.array([0, 0, 1, 1, 1, 1, 0])).all()


def test_win_check():
    key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    assert state.current_player == 0

    for _ in range(3):
        state = step(state, 0)
        state = step(state, 1)
    state = step(state, 0)
    assert state.terminated
    assert (state.reward == jnp.array([1.0, -1.0])).all()

    state = init(sub_key)
    for i in range(3):
        state = step(state, i)
        state = step(state, i)
    state = step(state, 3)
    assert state.terminated
    assert (state.reward == jnp.array([1.0, -1.0])).all()

    state = init(sub_key)
    for i in [1, 2, 2, 3, 3, 4, 3, 4, 4, 6, 4]:
        state = step(state, i)
    """
    .......
    .......
    ....@..
    ...@@..
    ..@@O..
    .@OOO.O
    """
    assert state.terminated
    assert (state.reward == jnp.array([1.0, -1.0])).all()


def test_random_play():
    key = jax.random.PRNGKey(0)
    done = jnp.bool_(False)
    key, sub_key = jax.random.split(key)
    state = init(sub_key)
    while not done:
        legal_actions = jnp.where(state.legal_action_mask)[0]
        key, sub_key = jax.random.split(key)
        action = jax.random.choice(sub_key, legal_actions)
        state = step(state, action)
        done = state.terminated


def test_observe():
    key = jax.random.PRNGKey(0)
    state = init(key)
    obs = observe(state, state.current_player)
    assert obs.shape == (6, 7, 2)

    state = step(state, 1)
    obs = observe(state, state.current_player)
    assert obs[:, :, 1].sum() == 1
    assert (obs[:, :, 1] == jnp.bool_(
        [[False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, True,  False, False, False, False, False]]
    )).all()
    assert obs[:, :, 0].sum() == 0
    obs = observe(state, state.current_player - 1)
    assert obs[:, :, 0].sum() == 1
    assert (obs[:, :, 0] == jnp.bool_(
        [[False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False],
         [False, True,  False, False, False, False, False]]
    )).all()
    assert obs[:, :, 1].sum() == 0

def test_api():
    import pgx
    env = pgx.make("connect_four")
    pgx.api_test(env, 10)
