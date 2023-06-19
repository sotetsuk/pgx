import jax
import jax.numpy as jnp
import pgx


FALSE = jnp.bool_(False)


class Wrapper:

    def __init__(self, env: pgx.Env):
        self.env: pgx.Env = env

    def init(self, key: jax.random.KeyArray) -> pgx.State:
        return self.env.init(key)

    def step(self, state: pgx.State, action: jnp.ndarray) -> pgx.State:
        return self.env.step(state, action)


class AutoReset(Wrapper):

    def step(self, state: pgx.State, action: jnp.ndarray) -> pgx.State:
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = self.env.step(state, action)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: self.env.init(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state
