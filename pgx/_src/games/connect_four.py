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

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    color: Array = jnp.int32(0)
    # 6x7 board
    # [[ 0,  1,  2,  3,  4,  5,  6],
    #  [ 7,  8,  9, 10, 11, 12, 13],
    #  [14, 15, 16, 17, 18, 19, 20],
    #  [21, 22, 23, 24, 25, 26, 27],
    #  [28, 29, 30, 31, 32, 33, 34],
    #  [35, 36, 37, 38, 39, 40, 41]]
    board: Array = -jnp.ones(42, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        board2d = state.board.reshape(6, 7)
        num_filled = (board2d[:, action] >= 0).sum()
        board2d = board2d.at[5 - num_filled, action].set(state.color)
        won = ((board2d.flatten()[IDX] == state.color).all(axis=1)).any()
        winner = jax.lax.select(won, state.color, -1)
        return state._replace(  # type: ignore
            color=1 - state.color,
            board=board2d.flatten(),
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        def make(turn):
            return state.board.reshape(6, 7) == turn

        turns = jax.lax.select(color == 0, jnp.int32([0, 1]), jnp.int32([1, 0]))
        return jnp.stack(jax.vmap(make)(turns), -1)

    def legal_action_mask(self, state: GameState) -> Array:
        board2d = state.board.reshape(6, 7)
        return (board2d >= 0).sum(axis=0) < 6

    def is_terminal(self, state: GameState) -> Array:
        board2d = state.board.reshape(6, 7)
        return (state.winner >= 0) | jnp.all((board2d >= 0).sum(axis=0) == 6)

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )


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
