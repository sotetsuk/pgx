import jax
import jax.numpy as jnp
from pgx.leduc_holdem import LeducHoldem, CALL, RAISE, FOLD, TRUE, State

env = LeducHoldem()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

def make_test_state(key):
    # due to API update
    _, key = jax.random.split(key)
    rng1, rng2, rng3 = jax.random.split(key, 3)
    current_player = jnp.int32(jax.random.bernoulli(rng1))
    init_card = jax.random.permutation(
        rng2, jnp.int32([0, 0, 1, 1, 2, 2]), independent=True
    )
    return State(  # type:ignore
        _first_player=current_player,
        current_player=current_player,
        _cards=init_card[:3],
        legal_action_mask=jnp.bool_([1, 1, 0]),
        _chips=jnp.ones(2, dtype=jnp.int32),
    )



def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert state.current_player == 0
    assert (state.legal_action_mask == jnp.bool_([1, 1, 0])).all()


def test_step():
    key = jax.random.PRNGKey(1)

    # =================
    state = init(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, RAISE)
    state = step(state, FOLD)
    assert state.terminated
    assert (state.rewards == jnp.float32([-1, 1])).all()

    # =================
    state = init(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, RAISE)  # +2(3)
    state = step(state, RAISE)  # +2(5)
    state = step(state, FOLD)
    assert state.terminated
    assert (state.rewards == jnp.float32([3, -3])).all()

    # =================
    state = init(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, CALL)
    state = step(state, RAISE)  # +2(3)
    state = step(state, RAISE)  # +2(5)
    state = step(state, FOLD)
    assert state.terminated
    assert (state.rewards == jnp.float32([-3, 3])).all()

    # =================
    state = make_test_state(key)
    assert state.current_player == 1
    # cards = [1 0]
    # player 1 is the first

    # first round
    state = step(state, CALL)
    state = step(state, RAISE)  # +2(3)
    state = step(state, CALL)

    # second round
    assert state.current_player == 1
    state = step(state, CALL)
    state = step(state, RAISE)  # +4(7)
    state = step(state, RAISE)  # +4(11)
    assert not state.terminated
    state = step(state, CALL)
    assert state.terminated
    assert (state.rewards == jnp.float32([11, -11])).all()

    # =================
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
    assert (state.rewards == jnp.float32([-7, 7])).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert (state.legal_action_mask == jnp.bool_([1, 1, 0])).all()

    state = step(state, CALL)
    assert (state.legal_action_mask == jnp.bool_([1, 1, 0])).all()
    state = step(state, RAISE)
    assert (state.legal_action_mask == jnp.bool_([1, 1, 1])).all()
    state = step(state, RAISE)

    # cannot raise
    assert (state.legal_action_mask == jnp.bool_([1, 0, 1])).all()

    state = step(state, CALL)

    # second round
    assert (state.legal_action_mask == jnp.bool_([1, 1, 0])).all()
    state = step(state, RAISE)
    assert (state.legal_action_mask == jnp.bool_([1, 1, 1])).all()
    state = step(state, RAISE)

    # cannot raise
    assert (state.legal_action_mask == jnp.bool_([1, 0, 1])).all()


def test_draw():
    key = jax.random.PRNGKey(0)
    state = make_test_state(key)
    # cards = [1 1]
    state = step(state, RAISE)
    state = step(state, RAISE)
    state = step(state, CALL)

    state = step(state, RAISE)
    state = step(state, RAISE)
    assert not state.terminated
    state = step(state, CALL)
    assert state.terminated
    assert (state.rewards == jnp.float32([0, 0])).all()


def test_observe():
    key = jax.random.PRNGKey(5)
    state = make_test_state(key)
    """
    player 1 is the First.
    cards = [0 1]
    public cards = None
    chips = [1 1]
    round = 0
    """
    obs = observe(state, 1)
    assert (
        obs
        == jnp.zeros(34, dtype=jnp.bool_)
        .at[1]
        .set(TRUE)
        .at[6 + 1]
        .set(TRUE)
        .at[20 + 1]
        .set(TRUE)
    ).all()

    state = step(state, CALL)
    state = step(state, CALL)

    # public card: 1
    state = step(state, RAISE)  # +4(5)
    state = step(state, RAISE)  # +4(9)

    obs = observe(state, 0)
    assert (
        obs
        == jnp.zeros(34, dtype=jnp.bool_)
        .at[0]
        .set(TRUE)
        .at[3 + 1]
        .set(TRUE)
        .at[6 + 9]
        .set(TRUE)
        .at[20 + 5]
        .set(TRUE)
    ).all()
    obs = observe(state, 1)
    assert (
        obs
        == jnp.zeros(34, dtype=jnp.bool_)
        .at[1]
        .set(TRUE)
        .at[3 + 1]
        .set(TRUE)
        .at[6 + 5]
        .set(TRUE)
        .at[20 + 9]
        .set(TRUE)
    ).all()


def test_api():
    import pgx
    env = pgx.make("leduc_holdem")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
