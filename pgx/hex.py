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
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((11, 11, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(11 * 11 + 1, dtype=jnp.bool_).at[-1].set(FALSE)
    _step_count: Array = jnp.int32(0)
    # --- Hex specific ---
    _size: Array = jnp.int32(11)
    # 0(black), 1(white)
    _turn: Array = jnp.int32(0)
    # 11x11 board
    # [[  0,  1,  2,  ...,  8,  9, 10],
    #  [ 11,  12, 13, ..., 19, 20, 21],
    #  .
    #  .
    #  .
    #  [110, 111, 112, ...,  119, 120]]
    _board: Array = jnp.zeros(11 * 11, jnp.int32)  # <0(oppo), 0(empty), 0<(self)

    @property
    def env_id(self) -> core.EnvId:
        return "hex"


class Hex(core.Env):
    def __init__(self, *, size: int = 11):
        super().__init__()
        assert isinstance(size, int)
        self.size = size

    def _init(self, key: PRNGKey) -> State:
        return partial(_init, size=self.size)(rng=key)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return jax.lax.cond(
            action != self.size * self.size,
            lambda: partial(_step, size=self.size)(state, action),
            lambda: partial(_swap, size=self.size)(state),
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return partial(_observe, size=self.size)(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "hex"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: PRNGKey, size: int) -> State:
    current_player = jnp.int32(jax.random.bernoulli(rng))
    return State(_size=size, current_player=current_player)  # type:ignore


def _step(state: State, action: Array, size: int) -> State:
    set_place_id = action + 1
    board = state._board.at[action].set(set_place_id)
    neighbour = _neighbour(action, size)

    def merge(i, b):
        adj_pos = neighbour[i]
        return jax.lax.cond(
            (adj_pos >= 0) & (b[adj_pos] > 0),
            lambda: jnp.where(b == b[adj_pos], set_place_id, b),
            lambda: b,
        )

    board = jax.lax.fori_loop(0, 6, merge, board)
    won = _is_game_end(board, size, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    state = state.replace(  # type:ignore
        current_player=1 - state.current_player,
        _turn=1 - state._turn,
        _board=board * -1,
        rewards=reward,
        terminated=won,
        legal_action_mask=state.legal_action_mask.at[:-1].set(board == 0).at[-1].set(state._step_count == 1),
    )

    return state


def _swap(state: State, size: int) -> State:
    ix = jnp.nonzero(state._board, size=1)[0]
    row = ix // size
    col = ix % size
    swapped_ix = col * size + row
    set_place_id = swapped_ix + 1
    board = state._board.at[ix].set(0).at[swapped_ix].set(set_place_id)
    return state.replace(  # type: ignore
        current_player=1 - state.current_player,
        _turn=1 - state._turn,
        _board=board * -1,
        legal_action_mask=state.legal_action_mask.at[:-1].set(board == 0).at[-1].set(FALSE),
    )


def _observe(state: State, player_id: Array, size) -> Array:
    board = jax.lax.select(
        player_id == state.current_player,
        state._board.reshape((size, size)),
        -state._board.reshape((size, size)),
    )

    my_board = board * 1 > 0
    opp_board = board * -1 > 0
    ones = jnp.ones_like(my_board)
    color = jax.lax.select(player_id == state.current_player, state._turn, 1 - state._turn)
    color = color * ones
    can_swap = state.legal_action_mask[-1] * ones

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
    return jax.lax.cond(state._turn == 0, lambda: state._board, lambda: state._board * -1)
