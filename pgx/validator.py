import jax


def validate_init(init_fn):

    # rng must be set to state.rng
    rng = jax.random.PRNGKey(38292)
    state = init_fn(rng)
    assert (
        state.rng == rng
    ).all(), (
        f"rng must be set to state.rng: got {state.rng} but expected {rng}"
    )
