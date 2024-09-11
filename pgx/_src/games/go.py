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
from jax import Array
from jax import numpy as jnp

ZOBRIST_BOARD = jax.random.randint(
    jax.random.PRNGKey(12345), shape=(3, 19 * 19, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32
)


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)
    # ids of representative stone (smallest) in the connected stones
    board: Array = jnp.zeros(19 * 19, dtype=jnp.int32)  # b > 0, w < 0, empty = 0
    board_history: Array = jnp.full((8, 19 * 19), 2, dtype=jnp.int32)  # for obs
    num_captured: Array = jnp.zeros(2, dtype=jnp.int32)  # [b, w]
    consecutive_pass_count: Array = jnp.int32(0)
    ko: Array = jnp.int32(-1)  # by SSK
    is_psk: Array = jnp.bool_(False)
    hash: Array = jnp.zeros(2, dtype=jnp.uint32)
    hash_history: Array = jnp.zeros((19 * 19 * 2, 2), dtype=jnp.uint32)

    @property
    def color(self) -> Array:
        return self.step_count % 2

    @property
    def size(self) -> Array:
        return jnp.sqrt(self.board.shape[-1]).astype(jnp.int32)


class Game:
    def __init__(
        self, size: int = 19, komi: float = 7.5, history_length: int = 8, max_termination_steps: Optional[int] = None
    ):
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = size * size * 2 if max_termination_steps is None else max_termination_steps

    def init(self) -> GameState:
        return GameState(
            board=jnp.zeros(self.size**2, dtype=jnp.int32),
            board_history=jnp.full((self.history_length, self.size**2), 2, dtype=jnp.int32),
            hash_history=jnp.zeros((self.max_termination_steps, 2), dtype=jnp.uint32),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        state = state._replace(ko=jnp.int32(-1))
        # update state
        state = jax.lax.cond(
            (action < self.size * self.size),
            lambda: _apply_action(state, action, self.size),
            lambda: _apply_pass(state),
        )
        # update board history
        board_history = jnp.roll(state.board_history, self.size**2)
        board_history = board_history.at[0].set(jnp.clip(state.board, -1, 1).astype(jnp.int32))
        state = state._replace(board_history=board_history)
        # check PSK
        hash = _compute_hash(state)
        state = state._replace(hash=hash, hash_history=state.hash_history.at[state.step_count].set(hash))
        state = state._replace(is_psk=_is_psk(state))
        # increment turns
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        my_color_sign, _ = _colors(color)

        def _make(i):
            c = jnp.int32([1, -1])[i % 2] * my_color_sign
            return state.board_history[i // 2] == c

        log = jax.vmap(_make)(jnp.arange(self.history_length * 2))
        color = jnp.full_like(log[0], color)  # black=0, white=1
        return jnp.vstack([log, color]).transpose().reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Array:
        """Logic is highly inspired by OpenSpiel's Go implementation"""
        is_empty = state.board == 0
        my_color, opp_color = _colors(state.color)
        num_pseudo, idx_sum, idx_squared_sum = _count(state, self.size)
        chain_ix = jnp.abs(state.board) - 1
        in_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
        has_liberty = (state.board * my_color > 0) & ~in_atari
        kills_opp = (state.board * opp_color > 0) & in_atari

        @jax.vmap
        def is_neighbor_ok(xy):
            neighbors = _neighbour(xy, self.size)
            on_board = neighbors != -1
            _has_empty = is_empty[neighbors]
            _has_liberty = has_liberty[neighbors]
            _kills_opp = kills_opp[neighbors]
            return (on_board & (_has_empty | _kills_opp | _has_liberty)).any()

        neighbor_ok = is_neighbor_ok(jnp.arange(self.size**2))
        mask = is_empty & neighbor_ok
        mask = jax.lax.select(state.ko == -1, mask, mask.at[state.ko].set(False))
        return jnp.append(mask, True)  # pass is always legal

    def is_terminal(self, state: GameState) -> Array:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Array:
        scores = _count_point(state, self.size)
        is_black_win = scores[0] - self.komi > scores[1]
        rewards = jax.lax.select(is_black_win, jnp.float32([1, -1]), jnp.float32([-1, 1]))
        to_play = state.color
        rewards = jax.lax.select(state.is_psk, jnp.float32([-1, -1]).at[to_play].set(1.0), rewards)
        rewards = jax.lax.select(self.is_terminal(state), rewards, jnp.zeros(2, dtype=jnp.float32))
        return rewards


def _apply_pass(state: GameState) -> GameState:
    return state._replace(consecutive_pass_count=state.consecutive_pass_count + 1)


def _apply_action(state: GameState, action, size) -> GameState:
    state = state._replace(consecutive_pass_count=0)
    my_color, opp_color = _colors(state.color)

    # remove killed stones
    neighbours = _neighbour(action, size)
    chain_id = state.board[neighbours]
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    chain_ix = jnp.abs(chain_id) - 1
    is_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
    single_liberty = (idx_squared_sum[chain_ix] // idx_sum[chain_ix]) - 1
    is_killed = (neighbours != -1) & (chain_id * opp_color > 0) & is_atari & (single_liberty == action)
    surrounded_stones = (state.board[:, None] == chain_id) & (is_killed[None, :])
    num_captured = jnp.count_nonzero(surrounded_stones)
    ko_ix = jnp.nonzero(is_killed, size=1)[0][0]
    ko_may_occur = _ko_may_occur(state, action, size)
    state = state._replace(
        board=jnp.where(surrounded_stones.any(axis=-1), 0, state.board),
        num_captured=state.num_captured.at[state.color].add(num_captured),
        ko=jax.lax.select(ko_may_occur & (num_captured == 1), neighbours[ko_ix], -1),
    )

    # set stone
    state = state._replace(board=state.board.at[action].set((action + 1) * my_color))

    # merge neighbours
    is_my_chain = state.board[neighbours] * my_color > 0
    should_merge = (neighbours != -1) & is_my_chain
    new_id = state.board[action]
    tgt_ids = state.board[neighbours]
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), 9999))
    smallest_id = jnp.minimum(jnp.abs(new_id), smallest_id) * my_color
    mask = (state.board == new_id) | (should_merge[None, :] & (state.board[:, None] == tgt_ids[None, :])).any(axis=-1)
    state = state._replace(board=jnp.where(mask, smallest_id, state.board))

    return state


def _count(state: GameState, size):
    board = jnp.abs(state.board)
    is_empty = board == 0
    idx_sum = jnp.where(is_empty, jnp.arange(1, size**2 + 1), 0)
    idx_squared_sum = jnp.where(is_empty, jnp.arange(1, size**2 + 1) ** 2, 0)

    def _count_neighbor(xy):
        neighbors = _neighbour(xy, size)
        on_board = neighbors != -1
        return (
            jnp.where(on_board, is_empty[neighbors], 0).sum(),
            jnp.where(on_board, idx_sum[neighbors], 0).sum(),
            jnp.where(on_board, idx_squared_sum[neighbors], 0).sum(),
        )

    idx = jnp.arange(size**2)
    num_pseudo, idx_sum, idx_squared_sum = jax.vmap(_count_neighbor)(idx)

    def _num_pseudo(x):
        return jnp.where(board == x + 1, num_pseudo, 0).sum()

    def _idx_sum(x):
        return jnp.where(board == x + 1, idx_sum, 0).sum()

    def _idx_squared_sum(x):
        return jnp.where(board == x + 1, idx_squared_sum, 0).sum()

    return jax.vmap(_num_pseudo)(idx), jax.vmap(_idx_sum)(idx), jax.vmap(_idx_squared_sum)(idx)


def _colors(color):
    return jnp.int32([[1, -1], [-1, 1]])[color]  # (my_color, opp_color)


def _ko_may_occur(state: GameState, xy: int, size: int) -> Array:
    neighbours = _neighbour(xy, size)
    on_board = neighbours != -1
    _, opp_color = _colors(state.color)
    is_occupied_by_opp = state.board[neighbours] * opp_color > 0
    return (~on_board | is_occupied_by_opp).all()


def _neighbour(xy, size):
    dx, dy = jnp.int32([-1, +1, 0, 0]), jnp.int32([0, 0, -1, +1])
    xs, ys = xy // size + dx, xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)


def _compute_hash(state: GameState):
    board = jnp.clip(state.board, -1, 1)
    to_reduce = ZOBRIST_BOARD[board, jnp.arange(board.shape[-1])]
    hash = jax.lax.reduce(to_reduce, 0, jax.lax.bitwise_xor, (0,))
    return hash


def _is_psk(state: GameState):
    not_passed = state.consecutive_pass_count == 0
    has_same_hash = (state.hash == state.hash_history).all(axis=-1).sum() > 1
    return not_passed & has_same_hash


def _count_point(state: GameState, size):
    def calc_point(c):
        return _count_ji(state, c, size) + jnp.count_nonzero(state.board * c > 0)

    return jax.vmap(calc_point)(jnp.int32([1, -1]))


def _count_ji(state: GameState, color: int, size: int):
    board = jnp.clip(state.board * color, -1, 1)  # 1 = mine, -1 = opponent's
    adj_mat = jax.vmap(partial(_neighbour, size=size))(jnp.arange(size**2))  # (size**2, 4)

    def fill_opp(x):
        b, _ = x
        # True if empty and any of neighbours is opponent
        mask = (b == 0) & ((adj_mat != -1) & (b[adj_mat] == -1)).any(axis=1)
        return jnp.where(mask, -1, b), mask.any()

    board, _ = jax.lax.while_loop(lambda x: x[1], fill_opp, (board, True))
    return (board == 0).sum()
