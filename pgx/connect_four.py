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
    # 6x7 board
    # [[ 0,  1,  2,  3,  4,  5,  6],
    #  [ 7,  8,  9, 10, 11, 12, 13],
    #  [14, 15, 16, 17, 18, 19, 20],
    #  [21, 22, 23, 24, 25, 26, 27],
    #  [28, 29, 30, 31, 32, 33, 34],
    #  [35, 36, 37, 38, 39, 40, 41]]
    _board: Array = -jnp.ones(42, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((6, 7, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(7, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "connect_four"


class ConnectFour(core.Env):
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
        return "connect_four"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _make_win_cache():
    idx = []
    # Vertical
    for i in range(3):
        for j in range(7):
            a = i * 7 + j
            idx.append([a, a + 7, a + 14, a + 21])
    # Horizontal
    for i in range(6):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 1, a + 2, a + 3])

    # Diagonal
    for i in range(3):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 8, a + 16, a + 24])
    for i in range(3):
        for j in range(3, 7):
            a = i * 7 + j
            idx.append([a, a + 6, a + 12, a + 18])
    return jnp.int32(idx)


IDX = _make_win_cache()


def _init(rng: PRNGKey) -> State:
    current_player = jnp.int32(jax.random.bernoulli(rng))
    return State(current_player=current_player)  # type:ignore


def _step_game_state(state: GameState, action: Array) -> GameState:
    board2d = state._board.reshape(6, 7)
    num_filled = (board2d[:, action] >= 0).sum()
    board2d = board2d.at[5 - num_filled, action].set(state._turn)
    won = _win_check(board2d.flatten(), state._turn)
    winner = jax.lax.select(won, state._turn, -1)
    return state.replace(  # type: ignore
        _turn=1 - state._turn,
        _board=board2d.flatten(),
        winner=winner,
    )


def _step(state: State, action: Array) -> State:
    x = _step_game_state(state._x, action)
    state = state.replace(  # type: ignore
        current_player=1 - state.current_player,
        _x=x,
    )
    terminated = is_terminal(state._x)
    rewards = returns(state._x)
    should_flip = state.current_player != state._x._turn
    rewards = jax.lax.select(should_flip, jnp.flip(rewards), rewards)
    rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
    return state.replace(  # type: ignore
        legal_action_mask=legal_action_mask(state._x),
        rewards=rewards,
        terminated=terminated,
    )


def returns(state: GameState) -> Array:
    return jax.lax.cond(
        state.winner >= 0,
        lambda: jnp.float32([-1, -1]).at[state.winner].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )


def is_terminal(state: GameState) -> Array:
    board2d = state._board.reshape(6, 7)
    return (state.winner >= 0) | jnp.all((board2d >= 0).sum(axis=0) == 6)


def legal_action_mask(state: GameState) -> Array:
    board2d = state._board.reshape(6, 7)
    return ((board2d >= 0).sum(axis=0) < 6)


def _win_check(board, turn) -> Array:
    return ((board[IDX] == turn).all(axis=1)).any()


def _observe(state: State, player_id: Array) -> Array:
    turns = jnp.int32([state._x._turn, 1 - state._x._turn])
    turns = jax.lax.cond(
        player_id == state.current_player,
        lambda: turns,
        lambda: jnp.flip(turns),
    )

    def make(turn):
        return state._x._board.reshape(6, 7) == turn

    return jnp.stack(jax.vmap(make)(turns), -1)
