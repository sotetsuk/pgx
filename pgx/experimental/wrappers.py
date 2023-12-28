from typing import Optional

import jax
import jax.numpy as jnp

from pgx._src.types import Array, PRNGKey
from pgx.core import State

FALSE = jnp.bool_(False)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    We have a concern about the final state before staging this wrapper:

    When auto-reset happens, the termianl (or truncated) state/observation is
    replaced by initial state/observation, It's ok if it's termination.
    However, when truncation happens, value of truncated state/observation
    might be used by agents (by bootstrap). So it must be stored somewhere.
    For example,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and
    terminates within reasonable # of steps.
    Note that chess, shogi, and Go have `max_termination_steps` as AlphaZero.
    So, this implementation is enough (so far).

    2. Performance
    """

    def wrapped_step_fn(state: State, action: Array, key: Optional[PRNGKey] = None):
        assert key is not None, (
            "v2.0.0 changes the signature of auto reset. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step_fn(state, action)\n"
            "  * >= v2.0.0: step_fn(state, action, key)\n\n"
            "Note that codes under pgx.experimental are subject to change without notice."
        )

        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action, key1)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(key2).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn
