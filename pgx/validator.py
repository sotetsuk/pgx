import jax

import pgx
from pgx.utils import act_randomly


def validate_init(init_fn):
    rng = jax.random.PRNGKey(38292)
    state = init_fn(rng)
    assert not (
        state.rng == jax.random.PRNGKey(0)  # default
    ).all(), f"Non-default rng must be set to state.rng: got {state.rng}"


def validate_step(init_fn, step_fn, N=100):
    rng = jax.random.PRNGKey(38201)
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        state = init_fn(rng)
        while True:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
            state = step_fn(state, action)

            validate_curr_player(state)

            if state.terminated:
                break


def validate_illegal_actions(state: pgx.State):
    ...


def validate_curr_player(state: pgx.State):
    if state.terminated:
        assert (
            state.curr_player == -1
        ), f"curr_player must be -1 when terminated but got {state.curr_player}"
    else:
        assert (
            state.curr_player >= 0
        ), f"curr_player must be positivie before terminated but got {state.curr_player}"
