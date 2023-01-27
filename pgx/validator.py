import jax


def validate_init(init_fn):

    # rng must be set to state.rng
    rng = jax.random.PRNGKey(38292)
    state = init_fn(rng)
    assert not (
        state.rng == jax.random.PRNGKey(0)  # default
    ).all(), f"Non-default rng must be set to state.rng: got {state.rng}"
