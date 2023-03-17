# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Literal, Optional, Tuple, get_args

import jax
import jax.numpy as jnp

from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pgx environments are strictly versioned like OpenAI Gym.
# One can check the version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) we do not provide older versions (as with OpenAI Gym),
# (2) it is tedious to remember or rewrite version numbers, and
# (3) we do not want to slow down development for fear of inconveniencing users.
EnvId = Literal[
    "hex",
    "tic_tac_toe",
    "go-19x19",
    "shogi",
    "backgammon",
    "minatar/asterix",
]


@dataclass
class State:
    steps: jnp.ndarray
    current_player: jnp.ndarray
    observation: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray  # so far, not used as all Pgx environments are finite horizon
    legal_action_mask: jnp.ndarray
    # NOTE: _rng_key is
    #   - used for stochastic env and auto reset
    #   - updated only when actually used
    #   - supposed NOT to be used by agent
    _rng_key: jax.random.KeyArray

    def _repr_html_(self) -> str:
        from pgx._visualizer import Visualizer

        v = Visualizer()
        return v.get_dwg(states=self).tostring()

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        """
        color_theme: Default(None) is "light"
        scale: change image size. Default(None) is 1.0
        """
        from pgx._visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.save_svg(self, filename)


class Env(abc.ABC):
    def __init__(self):
        ...

    def init(self, key: jax.random.KeyArray) -> State:
        key, subkey = jax.random.split(key)
        state = self._init(subkey)
        state = state.replace(_rng_key=key)  # type: ignore
        observation = self.observe(state, state.current_player)
        return state.replace(observation=observation)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(reward=jnp.zeros_like(state.reward)),  # type: ignore
            lambda: self._step(state.replace(steps=state.steps + 1), action),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
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

        observation = self.observe(state, state.current_player)
        return state.replace(observation=observation)  # type: ignore

    def observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        obs = self._observe(state, player_id)
        return jax.lax.stop_gradient(obs)

    @abc.abstractmethod
    def _init(self, key: jax.random.KeyArray) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        ...

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of legal_action_mask"""
        state = self.init(jax.random.PRNGKey(0))
        return state.legal_action_mask.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(
        self, state: State, loser: jnp.ndarray
    ) -> State:
        penalty = self._illegal_action_penalty
        reward = (
            jnp.ones_like(state.reward)
            * (-1 * penalty)
            * (self.num_players - 1)
        )
        reward = reward.at[loser].set(penalty)
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
    elif env_id == "hex":
        from pgx.hex import Hex

        return Hex()
    else:
        available_envs = "\n".join(get_args(EnvId))
        raise ValueError(
            f"Wrong env_id is passed. Available ids are: \n{available_envs}"
        )
