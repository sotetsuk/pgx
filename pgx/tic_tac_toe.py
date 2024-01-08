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

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class GameState:
    _turn: Array = jnp.int32(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    _board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((3, 3, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(9, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "tic_tac_toe"


class TicTacToe(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "tic_tac_toe"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: PRNGKey) -> State:
    current_player = jnp.int32(jax.random.bernoulli(rng))
    return State(current_player=current_player)  # type:ignore


def _step_game_state(state: GameState, action: Array) -> GameState:
    board = state._board.at[action].set(state._turn)
    idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
    won = (board[idx] == state._turn).all(axis=1).any()
    winner = jax.lax.select(won, state._turn, -1)
    return state.replace(  # type: ignore
        _board=state._board.at[action].set(state._turn),
        _turn=(state._turn + 1) % 2,
        winner=winner,
    )


def _step(state: State, action: Array) -> State:
    x = _step_game_state(state._x, action)
    state = state.replace(current_player=(state.current_player + 1) % 2, _x=x)  # type: ignore
    legal_action_mask = _legal_action_mask(x)
    terminated = _is_terminal(x)
    rewards = _returns(x)
    should_flip = state.current_player == state._x._turn
    rewards = jax.lax.select(should_flip, rewards, jnp.flip(rewards))
    rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
    return state.replace(legal_action_mask=legal_action_mask, rewards=rewards, terminated=terminated)  # type: ignore


def _legal_action_mask(state: GameState) -> Array:
    return state._board < 0


def _is_terminal(state: GameState) -> Array:
    return (state.winner >= 0) | jnp.all(state._board != -1)


def _returns(state: GameState) -> Array:
    return jax.lax.select(
        state.winner >= 0,
        jnp.float32([-1, -1]).at[state.winner].set(1),
        jnp.zeros(2, jnp.float32),
    )


def _observe(state: State, player_id: Array) -> Array:
    curr_color = state._x._turn
    my_color = jax.lax.select(player_id == state.current_player, curr_color, 1 - curr_color)
    return _observe_game_state(state._x, my_color)


def _observe_game_state(state: GameState, color: Array) -> Array:
    @jax.vmap
    def plane(i):
        return (state._board == i).reshape((3, 3))

    # flip if player_id is opposite
    x = jax.lax.select(color == 0, jnp.int32([0, 1]), jnp.int32([1, 0]))
    return jnp.stack(plane(x), -1)
