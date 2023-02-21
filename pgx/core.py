import abc
from typing import Literal, Tuple

import jax
import jax.numpy as jnp

from pgx.flax.struct import dataclass

EnvId = Literal[
    "tic_tac_toe/v0",
    "go/v0",
    "shogi/v0" "suzume_jong/v0",
    "minatar/breakout/v0",
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
        #    - Typical usage of legal_action_mask is nomralizing action probability
        #    - all zero mask will raise zero division error
        #  - ends with negative reward if illegal action is taken
        state = jax.lax.cond(
            state.terminated,
            self._step_if_terminated,
            self._step,
            state,
            action,
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
    def _step_if_terminated(state: State, action: jnp.ndarray) -> State:
        return state.replace(reward=jnp.zeros_like(state.reward))  # type: ignore

    @staticmethod
    def _step_with_illegal_action(state: State, action: jnp.ndarray) -> State:
        # TODO: implement me
        return state
