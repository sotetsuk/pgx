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

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array


FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


class GameState(NamedTuple):
    # 0(black), 1(white)
    step_count: Array = jnp.int32(0)
    # 11x11 board
    # [[  0,  1,  2,  ...,  8,  9, 10],
    #  [ 11,  12, 13, ..., 19, 20, 21],
    #  .
    #  .
    #  .
    #  [110, 111, 112, ...,  119, 120]]
    board: Array = jnp.zeros(11 * 11, jnp.int32)  # <0(oppo), 0(empty), 0<(self)
    terminated: Array = FALSE

    @property
    def color(self) -> Array:
        return self.step_count % 2


class Game:
    def __init__(self, size: int = 11):
        self.size = size

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        return jax.lax.cond(
            action != self.size * self.size,
            lambda: partial(_step, size=self.size)(state, action),
            lambda: partial(_swap, size=self.size)(state),
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        return _observe(state, color, self.size)

    def legal_action_mask(self, state: GameState) -> Array:
        return jnp.append(state.board == 0, state.step_count == 1)

    def is_terminal(self, state: GameState) -> Array:
        return state.terminated

    # def rewards(self, state: GameState) -> Array:
    #     ...


def _is_terminal(state: GameState, size: int) -> Array:
    top, bottom = jax.lax.cond(
        state.color == 0,
        lambda: (state.board[::size], state.board[size - 1 :: size]),
        lambda: (state.board[:size], state.board[-size:]),
    )

    def check_same_id_exist(_id):
        return (_id < 0) & (_id == bottom).any()

    return jax.vmap(check_same_id_exist)(top).any()


def _step(state: GameState, action: Array, size: int) -> GameState:
    set_place_id = action + 1
    board = state.board.at[action].set(set_place_id)
    neighbour = _neighbour(action, size)

    def merge(i, b):
        adj_pos = neighbour[i]
        return jax.lax.cond(
            (adj_pos >= 0) & (b[adj_pos] > 0),
            lambda: jnp.where(b == b[adj_pos], set_place_id, b),
            lambda: b,
        )

    board = jax.lax.fori_loop(0, 6, merge, board)
    
    state = state._replace(
        step_count=state.step_count + 1,
        board=board * -1,
    )

    terminated = _is_terminal(state, size)
    return state._replace(terminated=terminated)


def _swap(state: GameState, size: int) -> GameState:
    ix = jnp.nonzero(state.board, size=1)[0]
    row = ix // size
    col = ix % size
    swapped_ix = col * size + row
    set_place_id = swapped_ix + 1
    board = state.board.at[ix].set(0).at[swapped_ix].set(set_place_id)
    return state._replace(
        step_count=state.step_count + 1,
        board=board * -1,
    )


def _observe(state: GameState, color: Optional[Array] = None, size: int = 11) -> Array:
    if color is None:
        color = state.color

    board = jax.lax.select(color == state.color, state.board, -state.board)
    board = board.reshape((size, size))

    my_board = board * 1 > 0
    opp_board = board * -1 > 0
    ones = jnp.ones_like(my_board)
    color = color * ones
    can_swap = (state.step_count == 1) * ones

    return jnp.stack([my_board, opp_board, color, can_swap], 2, dtype=jnp.bool_)


def _neighbour(xy, size):
    """
        (x,y-1)   (x+1,y-1)
    (x-1,y)    (x,y)    (x+1,y)
       (x-1,y+1)   (x,y+1)
    """
    x = xy // size
    y = xy % size
    xs = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)
