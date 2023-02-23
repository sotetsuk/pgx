import abc
from typing import Literal, Tuple

import jax
import jax.numpy as jnp

from pgx.flax.struct import dataclass

TRUE = jnp.bool_(True)


EnvId = Literal[
    "tic_tac_toe/v0",
    "go-19x19/v0",
    "shogi/v0",
    "backgammon/v0",
    "minatar/asterix/v0",
]


@dataclass
class State:
    curr_player: jnp.ndarray
    observation: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    legal_action_mask: jnp.ndarray


class Env(abc.ABC):
    def __init__(self):
        ...

    def init(self, key: jax.random.KeyArray) -> State:
        state = self._init(key)
        observation = self.observe(state, state.curr_player)
        return state.replace(observation=observation)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        # TODO: legal_action_mask周りの挙動
        #  - set legal_action_mask = all True
        #    - Typical usage of legal_action_mask is normalizing action probability
        #    - all zero mask will raise zero division error
        #  - ends with negative reward if illegal action is taken

        is_illegal = ~state.legal_action_mask[action]
        curr_player = state.curr_player
        # If the state is already terminated, environment does not take usual step, but
        # return the same state with zero-rewards for all players
        state = jax.lax.cond(
            state.terminated,
            lambda: self._step_if_terminated(state),
            lambda: self._step(state, action),
        )
        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, curr_player),
            lambda: state,
        )
        # All legal_action_mask elements are TRUE (**not FALSE**) at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(  # type: ignore
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
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
