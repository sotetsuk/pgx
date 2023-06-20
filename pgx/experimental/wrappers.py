from typing import Tuple

import jax
import jax.numpy as jnp

import pgx

FALSE = jnp.bool_(False)


class Wrapper(pgx.Env):
    def __init__(self, env: pgx.Env):
        self.env: pgx.Env = env

    def _init(self, key: jax.random.KeyArray) -> pgx.State:
        """Implement game-specific init function here."""
        return self.env._init(key)

    def _step(self, state, action) -> pgx.State:
        """Implement game-specific step function here."""
        return self.env._step(state, action)

    def _observe(self, state: pgx.State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Implement game-specific observe function here."""
        return self.env._observe(state, player_id)

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


class ToSingle(Wrapper):
    """Flatten rewards to (batch_size,) assuming only <player_id=0> plays."""

    def init(self, key: jax.random.KeyArray) -> pgx.State:
        state = self.env.init(key)
        return state.replace(rewards=state.rewards[:, 0])  # type: ignore

    def step(self, state: pgx.State, action: jnp.ndarray) -> pgx.State:
        state = self.env.step(state, action)
        return state.replace(rewards=state.rewards[:, 0])  # type: ignore


class SpecifyFirstPlayer(Wrapper):

    def __init__(self, env: pgx.Env):
        super().__init__(env)
        assert self.num_players == 2, "SpecifyFirstPlayer is only for two-player game."

    def init_with_first_player(
        self, key: jax.random.KeyArray, first_player_id: jnp.ndarray
    ) -> pgx.State:
        """Special init function for two-player perfect information game.
        Args:
            key: pseudo-random generator key in JAX
            first_player_id: zero or one
        Returns:
            State: initial state of environment
        """
        state = self.init(key=key)
        return state.replace(current_player=jnp.int8(first_player_id))  # type: ignore
