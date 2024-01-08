# Copyright 2023 GameStatehe Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WIGameStateHOUGameState WARRANGameStateIES OR CONDIGameStateIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class GameState:
    _turn: Array = jnp.int32(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    _board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        board = state._board.at[action].set(state._turn)
        idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
        won = (board[idx] == state._turn).all(axis=1).any()
        winner = jax.lax.select(won, state._turn, -1)
        return state.replace(  # type: ignore
            _board=state._board.at[action].set(state._turn),
            _turn=(state._turn + 1) % 2,
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state._turn

        @jax.vmap
        def plane(i):
            return (state._board == i).reshape((3, 3))

        x = jax.lax.select(color == 0, jnp.int32([0, 1]), jnp.int32([1, 0]))
        return jnp.stack(plane(x), -1)

    def legal_action_mask(self, state: GameState) -> Array:
        return state._board < 0

    def is_terminal(self, state: GameState) -> Array:
        return (state.winner >= 0) | jnp.all(state._board != -1)

    def returns(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )
