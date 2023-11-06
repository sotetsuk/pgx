# I personally don't prefer env = Wrapper(env) style.
import jax
import jax.numpy as jnp

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    There are several concerns before staging this wrapper:

    1. Final state (observation)
    When auto-reset happens, the termianl (or truncated) state/observation is
    replaced by initial state/observation, It's ok if it's termination.
    However, when truncation happens, value of truncated state/observation
    might be used by agents. So its must be stored somewhere.
    For example,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and
    terminates within reasonable # of steps.
    Note that chess, shogi, and Go have `max_termination_steps` as AlphaZero.
    So, this implementation is enough (so far).

    2. Performance:
    Might harm the performance as it always generates new state.
    Memory usage might be doubled. Need to check.
    """

    def wrapped_step_fn(state, action):
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn
