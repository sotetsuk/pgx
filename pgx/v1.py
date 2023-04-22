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

from pgx._src.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pgx environments are versioned like OpenAI Gym or Brax.
# OpenAI Gym forces user to specify version (e.g., `MountainCar-v0`); while Brax does not (e.g., `ant`)
# We follow the way of Brax. One can check the environment version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) In game domain, performance measure is not the score in environment but
#     the comparison to other agents (i.e., environment version is less important),
# (2) we do not provide older versions (as with OpenAI Gym), and
# (3) it is tedious to remember and write version numbers.
EnvId = Literal[
    "2048",
    "animal_shogi",
    "backgammon",
    "bridge_bidding",
    "chess",
    "connect_four",
    "go-9x9",
    "go-19x19",
    "hex",
    "kuhn_poker",
    "leduc_holdem",
    # "mahjong",
    "minatar/asterix",
    "minatar/breakout",
    "minatar/freeway",
    "minatar/seaquest",
    "minatar/space_invaders",
    "othello",
    "shogi",
    "sparrow_mahjong",
    "tic_tac_toe",
]


@dataclass
class State(abc.ABC):
    """Base state class of all Pgx game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (jnp.ndarray): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        observation (jnp.ndarray): observation for the current state.
            `Env.observe` is called to compute.
        reward (jnp.ndarray): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.reward` is zero for all players.
        terminated (jnp.ndarray): denotes that the state is termianl state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminates within a limited number of states (following AlphaGo).
        truncated (jnp.ndarray): so far, not used as all Pgx environments are finite horizon
        legal_action_mask (jnp.ndarray): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

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
    _step_count: jnp.ndarray

    @property
    @abc.abstractmethod
    def env_id(self) -> EnvId:
        """Environment id (e.g. "go-19x19")"""
        ...

    def _repr_html_(self) -> str:
        return self.to_svg()

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        """Return SVG string. Useful for visualization in notebook.

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            str: SVG string
        """
        from pgx._src.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        """Save the entire state (not observation) to a file.
        The filename must end with `.svg`

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            None
        """
        from pgx._src.visualizer import save_svg

        save_svg(self, filename, color_theme=color_theme, scale=scale)


class Env(abc.ABC):
    """Environment class API.

    !!! example "Example usage"

        ```py
        env: Env = pgx.make("tic_tac_toe")
        state = env.init(jax.random.PRNGKey(0))
        action = jax.random.int32(4)
        state = env.step(state, action)
        ```

    """

    def __init__(self):
        ...

    def init(self, key: jax.random.KeyArray) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX

        Returns:
            State: initial state of environment

        """
        key, subkey = jax.random.split(key)
        state = self._init(subkey)
        state = state.replace(_rng_key=key)  # type: ignore
        observation = self.observe(state, state.current_player)
        return state.replace(observation=observation)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(reward=jnp.zeros_like(state.reward)),  # type: ignore
            lambda: self._step(state.replace(_step_count=state._step_count + 1), action),  # type: ignore
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
        """Observation function."""
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
    def id(self) -> EnvId:
        """Environment id."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.init(jax.random.PRNGKey(0))
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

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


def available_games() -> Tuple[EnvId, ...]:
    return get_args(EnvId)


def make(env_id: EnvId):  # noqa: C901
    if env_id == "2048":
        from pgx.play2048 import Play2048

        return Play2048()
    elif env_id == "animal_shogi":
        from pgx.animal_shogi import AnimalShogi

        return AnimalShogi()
    elif env_id == "backgammon":
        from pgx.backgammon import Backgammon

        return Backgammon()
    elif env_id == "bridge_bidding":
        from pgx.bridge_bidding import BridgeBidding

        return BridgeBidding()
    elif env_id == "chess":
        from pgx.chess import Chess

        return Chess()
    elif env_id == "connect_four":
        from pgx.connect_four import ConnectFour

        return ConnectFour()
    elif env_id == "go-9x9":
        from pgx.go import Go

        return Go(size=9, komi=7.5)
    elif env_id == "go-19x19":
        from pgx.go import Go

        return Go(size=19, komi=7.5)
    elif env_id == "hex":
        from pgx.hex import Hex

        return Hex()
    elif env_id == "kuhn_poker":
        from pgx.kuhn_poker import KuhnPoker

        return KuhnPoker()
    elif env_id == "leduc_holdem":
        from pgx.leduc_holdem import LeducHoldem

        return LeducHoldem()
    elif env_id == "minatar/asterix":
        from pgx.minatar.asterix import MinAtarAsterix

        return MinAtarAsterix()
    elif env_id == "minatar/breakout":
        from pgx.minatar.breakout import MinAtarBreakout

        return MinAtarBreakout()
    elif env_id == "minatar/freeway":
        from pgx.minatar.freeway import MinAtarFreeway

        return MinAtarFreeway()
    elif env_id == "minatar/seaquest":
        from pgx.minatar.seaquest import MinAtarSeaquest

        return MinAtarSeaquest()
    elif env_id == "minatar/space_invaders":
        from pgx.minatar.space_invaders import MinAtarSpaceInvaders

        return MinAtarSpaceInvaders()
    elif env_id == "othello":
        from pgx.othello import Othello

        return Othello()
    elif env_id == "shogi":
        from pgx.shogi import Shogi

        return Shogi()
    elif env_id == "sparrow_mahjong":
        from pgx.sparrow_mahjong import SparrowMahjong

        return SparrowMahjong()
    elif env_id == "tic_tac_toe":
        from pgx.tic_tac_toe import TicTacToe

        return TicTacToe()
    else:
        available_envs = "\n".join(available_games())
        raise ValueError(
            f"Wrong env_id is passed. Available ids are: \n{available_envs}"
        )
