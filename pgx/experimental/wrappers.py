import jax
import jax.numpy as jnp
import pgx
from typing import Tuple


FALSE = jnp.bool_(False)


class Wrapper(pgx.Env):

    def __init__(self, env: pgx.Env):
        self.env: pgx.Env = env

    def init(self, key: jax.random.KeyArray) -> pgx.State:
        return self.env.init(key)

    def step(self, state: pgx.State, action: jnp.ndarray) -> pgx.State:
        return self.env.step(state, action)

    def observe(self, state: pgx.State, player_id: jnp.ndarray) -> jnp.ndarray:
        return self.env.observe(state, player_id)

    @property
    def id(self) -> pgx.EnvId:
        return self.env.id

    @property
    def version(self) -> str:
        return self.env.version

    @property
    def num_players(self) -> int:
        return self.env.num_players

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.env.observation_shape


class AutoReset(Wrapper):
    """AutoReset wrapper resets the state to the initial state immediately just after termination or truncation.
    Note that the state before reset is required when truncation occurs,
    but Pgx does not have such an environment at present, so it is not a practical problem.
    """

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
