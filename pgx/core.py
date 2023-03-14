import abc
from typing import Literal, Tuple, get_args

import jax
import jax.numpy as jnp

from pgx.flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pgx environments are strictly versioned like OpenAI Gym.
# One can check the version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) we do not provide older versions (as with OpenAI Gym),
# (2) it is tedious to remember or rewrite version numbers, and
# (3) we do not want to slow down development for fear of inconveniencing users.
EnvId = Literal[
    "tic_tac_toe",
    "go-19x19",
    "shogi",
    "backgammon",
    "minatar/asterix",
]


@dataclass
class State:
    steps: jnp.ndarray
    curr_player: jnp.ndarray
    observation: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    legal_action_mask: jnp.ndarray
    # NOTE: _rng_key is
    #   - used for stochastic env and auto reset
    #   - updated only when actually used
    #   - supposed NOT to be used by agent
    _rng_key: jax.random.KeyArray

    def _repr_html_(self) -> str:
        from pgx.visualizer import Visualizer

        v = Visualizer()
        return v._to_dwg_from_states(states=self).tostring()


class Env(abc.ABC):
    def __init__(
        self, *, auto_reset: bool = False, max_truncation_steps: int = -1
    ):
        self.auto_reset = jnp.bool_(auto_reset)
        self.max_truncation_steps = jnp.int32(max_truncation_steps)

    def init(self, key: jax.random.KeyArray) -> State:
        key, subkey = jax.random.split(key)
        state = self._init(subkey)
        state = state.replace(_rng_key=key)  # type: ignore
        observation = self.observe(state, state.curr_player)
        return state.replace(observation=observation)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        is_illegal = ~state.legal_action_mask[action]
        curr_player = state.curr_player

        # Auto reset
        state = jax.lax.cond(
            self.auto_reset & (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                steps=jnp.int32(0),
                terminated=FALSE,
                truncated=FALSE,
                reward=jnp.zeros_like(state.reward),
            ),
            lambda: state,
        )

        # increment step count
        state = state.replace(steps=state.steps + 1)  # type: ignore

        # If the state is already terminated, environment does not take usual step, but
        # return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: self._step_if_terminated(state),
            lambda: self._step(state, action),
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, curr_player),
            lambda: state,
        )

        # Time limit
        state = jax.lax.cond(
            ~state.terminated
            & (0 <= self.max_truncation_steps)
            & (self.max_truncation_steps <= state.steps),
            lambda: state.replace(truncated=TRUE),  # type: ignore
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
            ),
            lambda: state,
        )

        # Auto reset
        state = jax.lax.cond(
            self.auto_reset & (state.terminated | state.truncated),
            lambda: self.init(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                reward=state.reward,
            ),
            lambda: state,
        )

        observation = self.observe(state, state.curr_player)
        return state.replace(observation=observation)  # type: ignore

    @abc.abstractmethod
    def _init(self, key: jax.random.KeyArray) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def reward_range(self) -> Tuple[float, float]:
        """Note that min reward must be <= 0."""
        ...

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self.observe(state, state.curr_player)
        return obs.shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of legal_action_mask"""
        state = self.init(jax.random.PRNGKey(0))
        return state.legal_action_mask.shape

    @staticmethod
    def _step_if_terminated(state: State) -> State:
        return state.replace(  # type: ignore
            reward=jnp.zeros_like(state.reward),
        )

    def _step_with_illegal_action(
        self, state: State, loser: jnp.ndarray
    ) -> State:
        min_reward = self.reward_range[0]
        reward = (
            jnp.ones_like(state.reward)
            * (-1 * min_reward)
            * (self.num_players - 1)
        )
        reward = reward.at[loser].set(min_reward)
        return state.replace(reward=reward, terminated=TRUE)  # type: ignore


def make(env_id: EnvId):
    if env_id == "tic_tac_toe":
        from pgx.tic_tac_toe import TicTacToe

        return TicTacToe()
    elif env_id == "shogi":
        from pgx.shogi import Shogi

        return Shogi()
    elif env_id == "go-19x19":
        from pgx.go import Go

        return Go(size=19, komi=7.5)
    elif env_id == "backgammon":
        from pgx.backgammon import Backgammon

        return Backgammon()
    elif env_id == "minatar/asterix":
        from pgx.minatar.asterix import MinAtarAsterix

        return MinAtarAsterix()
    else:
        available_envs = "\n".join(get_args(EnvId))
        raise ValueError(
            f"Wrong env_id is passed. Available ids are: \n{available_envs}"
        )
