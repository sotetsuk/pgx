import abc
from typing import Literal, Tuple

import jax
import jax.numpy as jnp

from pgx.flax.struct import dataclass

EnvId = Literal[
    "tic_tac_toe/v0",
    "shogi/v0" "minatar/breakout/v0",
    "suzume_jong/v0",
    "go/v0",
]


@dataclass
class State:
    curr_player: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    legal_action_mask: jnp.ndarray


class Env(abc.ABC):
    def __init__(self):
        ...

    @abc.abstractmethod
    def init(self, rng: jax.random.KeyArray) -> State:
        ...

    def step(self, state: State, action: jnp.ndarray) -> State:
        # TODO: curr_player周りの挙動
        #  - set curr_player = -1 if already terminated
        # TODO: legal_action_mask周りの挙動
        #  - set legal_action_mask = all False if already terminated  # or all True?
        #  - ends with negative reward if illegal action is taken
        return jax.lax.cond(
            state.terminated,
            lambda: self._step_if_terminated(state, action),
            lambda: self._step(state, action),
        )

    @abc.abstractmethod
    def _step(self, state, action) -> State:
        ...

    @abc.abstractmethod
    def observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        ...

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        state = self.init(jax.random.PRNGKey(0))
        obs = self.observe(state, state.curr_player)
        return obs.shape

    @property
    def num_actions(self) -> int:
        state = self.init(jax.random.PRNGKey(0))
        return state.legal_action_mask.shape[0]

    @staticmethod
    def _step_if_terminated(state: State, action: jnp.ndarray) -> State:
        return state.replace(reward=jnp.zeros_like(state.reward))  # type: ignore

    @staticmethod
    def _step_with_illegal_action(state: State, action: jnp.ndarray) -> State:
        # TODO: implement me
        return state
