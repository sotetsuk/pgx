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
from pgx._flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((6, 7, 2), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(7, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Connect Four specific ---
    turn: jnp.ndarray = jnp.int8(0)
    # 6x7 board
    # [[ 0,  1,  2,  3,  4,  5,  6],
    #  [ 7,  8,  9, 10, 11, 12, 13],
    #  [14, 15, 16, 17, 18, 19, 20],
    #  [21, 22, 23, 24, 25, 26, 27],
    #  [28, 29, 30, 31, 32, 33, 34],
    #  [35, 36, 37, 38, 39, 40, 41]]
    board: jnp.ndarray = -jnp.ones(42, jnp.int8)  # -1 (empty), 0, 1
    blank_row: jnp.ndarray = jnp.full(7, 5)


class ConnectFour(core.Env):
    def __init__(
        self,
    ):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "Connect Four"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _make_win_cache():
    idx = []
    # 縦
    for i in range(3):
        for j in range(7):
            a = i * 7 + j
            idx.append([a, a + 7, a + 14, a + 21])
    # 横
    for i in range(6):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 1, a + 2, a + 3])

    # 斜め
    for i in range(3):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 8, a + 16, a + 24])
    for i in range(3):
        for j in range(3, 7):
            a = i * 7 + j
            idx.append([a, a + 6, a + 12, a + 18])
    return jnp.int8(idx)


IDX = _make_win_cache()


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(current_player=current_player)  # type:ignore


def _step(state: State, action: jnp.ndarray) -> State:
    board = state.board
    row = state.blank_row[action]
    blank_row = state.blank_row.at[action].set(row - 1)
    board = board.at[_to_idx(row, action)].set(state.turn)
    won = _win_check(board, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    return state.replace(  # type: ignore
        current_player=1 - state.current_player,
        legal_action_mask=blank_row >= 0,
        turn=1 - state.turn,
        board=board,
        blank_row=blank_row,
        terminated=won | jnp.all(blank_row == -1),
        reward=reward,
    )


def _to_idx(row, col):
    return row * 7 + col


def _win_check(board, turn) -> jnp.ndarray:
    return ((board[IDX] == turn).all(axis=1)).any()


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    turns = jnp.int8([state.turn, 1 - state.turn])
    turns = jax.lax.cond(
        player_id == state.current_player,
        lambda: turns,
        lambda: jnp.flip(turns),
    )

    def make(turn):
        return state.board.reshape(6, 7) == turn

    return jnp.stack(jax.vmap(make)(turns), -1)
