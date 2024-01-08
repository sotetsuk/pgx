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

from typing import Optional

import jax
from jax import Array
from jax import numpy as jnp

from pgx._src.struct import dataclass


@dataclass
class GameState:
    _turn: Array = jnp.int32(0)
    _board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1
    # 0 1 2
    # 3 4 5
    # 6 7 8


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        state = state.replace(_board=state._board.at[action].set(state._turn))  # type: ignore
        won = _win_check(state._board, state._turn)
        reward = jax.lax.cond(
            won,
            lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
            lambda: jnp.zeros(2, jnp.float32),
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        ...

    def legal_action_mask(self, state: GameState) -> Array:
        ...

    def is_terminal(self, state: GameState) -> Array:
        ...

    def returns(self, state: GameState) -> Array:
        ...


def _win_check(board, turn) -> Array:
    idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
    return ((board[idx] == turn).all(axis=1)).any()


def _observe(state: State, player_id: Array) -> Array:
    @jax.vmap
    def plane(i):
        return (state._board == i).reshape((3, 3))

    # flip if player_id is opposite
    x = jax.lax.cond(
        state.current_player == player_id,
        lambda: jnp.int32([state._turn, 1 - state._turn]),
        lambda: jnp.int32([1 - state._turn, state._turn]),
    )

    return jnp.stack(plane(x), -1)
