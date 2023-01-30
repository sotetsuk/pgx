import abc
from typing import Literal, Tuple

import jax.numpy as jnp
import jax.random
from flax.struct import dataclass

EnvId = Literal[
    "tic_tac_toe/v0",
    "minatar/breakout/v0",
    "suzume_jong/v0",
    "go/v0",
]


@dataclass
class State:
    rng: jax.random.KeyArray
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

    @abc.abstractmethod
    def step(self, state: State, action: jnp.ndarray) -> State:
        # TODO: legal_action_mask周りの挙動
        ...

    @abc.abstractmethod
    def observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        ...

    @property
    @abc.abstractmethod
    def num_players(self):
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
