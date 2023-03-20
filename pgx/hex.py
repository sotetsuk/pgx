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

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((11, 11, 2), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(11 * 11, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Hex specific ---
    size: jnp.ndarray = jnp.int8(11)
    # 0(black), 1(white)
    turn: jnp.ndarray = jnp.int8(0)
    # 11x11 board
    # [[  0,  1,  2,  ...,  8,  9, 10],
    #  [ 11,  12, 13, ..., 19, 20, 21],
    #  .
    #  .
    #  .
    #  [110, 111, 112, ...,  119, 120]]
    board: jnp.ndarray = -jnp.zeros(
        11 * 11, jnp.int32
    )  # <0(oppo), 0(empty), 0<(self)


class Hex(core.Env):
    def __init__(
        self,
        *,
        size: int = 11,
    ):
        super().__init__()
        assert isinstance(size, int)
        self.size = size

    def _init(self, key: jax.random.KeyArray) -> State:
        return partial(_init, size=self.size)(rng=key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return partial(_step, size=self.size)(state, action)

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return partial(_observe, size=self.size)(state, player_id)

    @property
    def name(self) -> str:
        return f"Hex ({self.size}x{self.size})"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray, size: int) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(size=size, current_player=current_player)  # type:ignore


def _step(state: State, action: jnp.ndarray, size: int) -> State:
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
    won = _is_game_end(board, size, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    legal_action_mask = board == 0
    state = state.replace(  # type:ignore
        current_player=1 - state.current_player,
        turn=1 - state.turn,
        board=board * -1,
        reward=reward,
        terminated=won,
        legal_action_mask=legal_action_mask,
    )

    return state


def _observe(state: State, player_id: jnp.ndarray, size) -> jnp.ndarray:
    board = jax.lax.cond(
        player_id == state.current_player,
        lambda: state.board.reshape((size, size)),
        lambda: (state.board * -1).reshape((size, size)),
    )

    def make(color):
        return board * color > 0

    return jnp.stack(jax.vmap(make)(jnp.int8([1, -1])), 2)


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


def _is_game_end(board, size, turn):
    top, bottom = jax.lax.cond(
        turn == 0,
        lambda: (board[:size], board[-size:]),
        lambda: (board[::size], board[size - 1 :: size]),
    )

    def check_same_id_exist(_id):
        return (_id > 0) & (_id == bottom).any()

    return jax.vmap(check_same_id_exist)(top).any()


def _get_abs_board(state):
    return jax.lax.cond(
        state.turn == 0, lambda: state.board, lambda: state.board * -1
    )
