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
from jax import Array, lax
from jax import numpy as jnp

ZOBRIST_BOARD = jax.random.randint(jax.random.PRNGKey(12345), (3, 19 * 19, 2), 0, 2**31 - 1, jnp.uint32)


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)
    # ids of representative stone (smallest) in the connected stones
    board: Array = jnp.zeros(19 * 19, dtype=jnp.int32)  # b > 0, w < 0, empty = 0
    board_history: Array = jnp.full((8, 19 * 19), 2, dtype=jnp.int32)  # for obs
    num_captured: Array = jnp.zeros(2, dtype=jnp.int32)  # (b, w)
    consecutive_pass_count: Array = jnp.int32(0)
    ko: Array = jnp.int32(-1)  # by SSK
    is_psk: Array = jnp.bool_(False)
    hash_history: Array = jnp.zeros((19 * 19 * 2, 2), dtype=jnp.uint32)

    @property
    def color(self) -> Array:
        return self.step_count % 2


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
        state = lax.cond(
            (action < self.size * self.size),
            lambda: _apply_action(state, action, self.size),
            lambda: _apply_pass(state),
        )
        # update board history
        board_history = jnp.roll(state.board_history, self.size**2)
        board_history = board_history.at[0].set(jnp.clip(state.board, -1, 1).astype(jnp.int32))
        state = state._replace(board_history=board_history)
        # check PSK
        hash_ = _compute_hash(state)
        state = state._replace(hash_history=state.hash_history.at[state.step_count].set(hash_))
        state = state._replace(is_psk=_is_psk(state))
        # increment turns
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        my_sign, _ = _signs(color)

        def _make(i):
            c = jnp.int32([1, -1])[i % 2] * my_sign
            return state.board_history[i // 2] == c

        log = jax.vmap(_make)(jnp.arange(self.history_length * 2))
        color = jnp.full_like(log[0], color)  # b = 0, w = 1
        return jnp.vstack([log, color]).transpose().reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Array:
        # some logic is inspired by OpenSpiel's Go implementation
        is_empty = state.board == 0
        my_sign, opp_sign = _signs(state.color)
        num_pseudo, idx_sum, idx_squared_sum = _count(state, self.size)
        chain_ix = jnp.abs(state.board) - 1
        in_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
        has_liberty = (state.board * my_sign > 0) & ~in_atari
        can_kill = (state.board * opp_sign > 0) & in_atari

        def is_adj_ok(xy):
            adj_ixs = _adj_ixs(xy, self.size)
            on_board = adj_ixs != -1
            return (on_board & (is_empty[adj_ixs] | can_kill[adj_ixs] | has_liberty[adj_ixs])).any()

        mask = is_empty & jax.vmap(is_adj_ok)(jnp.arange(self.size**2))
        mask = lax.select(state.ko == -1, mask, mask.at[state.ko].set(False))
        return jnp.append(mask, True)  # pass is always legal

    def is_terminal(self, state: GameState) -> Array:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Array:
        scores = _count_scores(state, self.size)
        is_black_win = scores[0] - self.komi > scores[1]
        rewards = lax.select(is_black_win, jnp.float32([1, -1]), jnp.float32([-1, 1]))
        to_play = state.color
        rewards = lax.select(state.is_psk, jnp.float32([-1, -1]).at[to_play].set(1.0), rewards)
        rewards = lax.select(self.is_terminal(state), rewards, jnp.zeros(2, dtype=jnp.float32))
        return rewards


def _apply_pass(state: GameState) -> GameState:
    return state._replace(consecutive_pass_count=state.consecutive_pass_count + 1)


def _apply_action(state: GameState, action, size) -> GameState:
    state = state._replace(consecutive_pass_count=0)
    my_sign, opp_sign = _signs(state.color)

    # remove killed stones
    adj_ixs = _adj_ixs(action, size)
    adj_ids = state.board[adj_ixs]
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    chain_ix = jnp.abs(adj_ids) - 1
    is_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
    single_liberty = (idx_squared_sum[chain_ix] // idx_sum[chain_ix]) - 1
    is_killed = (adj_ixs != -1) & (adj_ids * opp_sign > 0) & is_atari & (single_liberty == action)
    surrounded_stones = (state.board[:, None] == adj_ids) & (is_killed[None, :])
    num_captured = jnp.count_nonzero(surrounded_stones)
    ko_ix = jnp.nonzero(is_killed, size=1)[0][0]
    ko_may_occur = ((adj_ixs == -1) | (state.board[adj_ixs] * opp_sign > 0)).all()
    state = state._replace(
        board=jnp.where(surrounded_stones.any(axis=-1), 0, state.board),
        num_captured=state.num_captured.at[state.color].add(num_captured),
        ko=lax.select(ko_may_occur & (num_captured == 1), adj_ixs[ko_ix], -1),
    )

    # set stone
    state = state._replace(board=state.board.at[action].set((action + 1) * my_sign))

    # merge adjacent chains
    is_my_chain = state.board[adj_ixs] * my_sign > 0
    should_merge = (adj_ixs != -1) & is_my_chain
    new_id = state.board[action]
    tgt_ids = state.board[adj_ixs]
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), 9999))
    smallest_id = jnp.minimum(jnp.abs(new_id), smallest_id) * my_sign
    mask = (state.board == new_id) | (should_merge[None, :] & (state.board[:, None] == tgt_ids[None, :])).any(axis=-1)
    state = state._replace(board=jnp.where(mask, smallest_id, state.board))

    return state


def _count(state: GameState, size):
    board = jnp.abs(state.board)
    is_empty = board == 0
    idx_sum = jnp.where(is_empty, jnp.arange(1, size**2 + 1), 0)
    idx_squared_sum = jnp.where(is_empty, jnp.arange(1, size**2 + 1) ** 2, 0)

    def _count_neighbor(xy):
        adj_ixs = _adj_ixs(xy, size)
        on_board = adj_ixs != -1
        return (
            jnp.where(on_board, is_empty[adj_ixs], 0).sum(),
            jnp.where(on_board, idx_sum[adj_ixs], 0).sum(),
            jnp.where(on_board, idx_squared_sum[adj_ixs], 0).sum(),
        )

    idx = jnp.arange(size**2)
    num_pseudo, idx_sum, idx_squared_sum = jax.vmap(_count_neighbor)(idx)

    def count_all(x):
        return (
            jnp.where(board == x + 1, num_pseudo, 0).sum(),
            jnp.where(board == x + 1, idx_sum, 0).sum(),
            jnp.where(board == x + 1, idx_squared_sum, 0).sum(),
        )

    return jax.vmap(count_all)(idx)


def _signs(color):
    return jnp.int32([[1, -1], [-1, 1]])[color]  # (my_sign, opp_sign)


def _adj_ixs(xy, size):
    dx, dy = jnp.int32([-1, +1, 0, 0]), jnp.int32([0, 0, -1, +1])
    xs, ys = xy // size + dx, xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)  # -1 if out of board


def _compute_hash(state: GameState):
    board = jnp.clip(state.board, -1, 1)
    to_reduce = ZOBRIST_BOARD[board, jnp.arange(board.shape[-1])]
    return lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))


def _is_psk(state: GameState):
    not_passed = state.consecutive_pass_count == 0
    curr_hash = state.hash_history[state.step_count]
    has_same_hash = (curr_hash == state.hash_history).all(axis=-1).sum() > 1
    return not_passed & has_same_hash


def _count_scores(state: GameState, size):
    def calc_point(c):
        return _count_ji(state, c, size) + jnp.count_nonzero(state.board * c > 0)

    return jax.vmap(calc_point)(jnp.int32([1, -1]))


def _count_ji(state: GameState, color: int, size: int):
    board = jnp.clip(state.board * color, -1, 1)  # my stone: 1, opp stone: -1
    adj_mat = jax.vmap(_adj_ixs, in_axes=(0, None))(jnp.arange(size**2), size)  # (size**2, 4)

    def fill_opp(x):
        b, _ = x
        # true if empty and adjacent to opponent's stone
        mask = (b == 0) & ((adj_mat != -1) & (b[adj_mat] == -1)).any(axis=1)
        return jnp.where(mask, -1, b), mask.any()

    board, _ = lax.while_loop(lambda x: x[1], fill_opp, (board, True))
    return (board == 0).sum()
