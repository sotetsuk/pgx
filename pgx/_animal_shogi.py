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
from typing import Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
BISHOP = jnp.int8(1)
ROOK = jnp.int8(2)
KING = jnp.int8(3)
GOLD = jnp.int8(4)
#  5: OPP_PAWN
#  6: OPP_ROOK
#  7: OPP_BISHOP
#  8: OPP_KING
#  9: OPP_GOLD
INIT_BOARD = jnp.int8([7, -1, -1, 1, 8, 5, 0, 3, 6, -1, -1, 2])  # (12,)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(132, dtype=jnp.bool_)  # (132,)
    observation: jnp.ndarray = jnp.zeros(1, dtype=jnp.bool_)  # TODO: fix me
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Animal Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD  # (12,)
    hand: jnp.ndarray = jnp.zeros((2, 3), dtype=jnp.int8)


# Implements AlphaZero like action:
# 132 =
#   [Move] 12 (from_) * 8 (direction) +
#   [Drop] 12 (to) * 3 (piece_type)
@dataclass
class Action:
    is_drop: jnp.ndarray = FALSE
    piece: jnp.ndarray = jnp.int8(0)
    from_: jnp.ndarray = jnp.int8(0)
    to: jnp.ndarray = jnp.int8(0)
    is_promotion: jnp.ndarray = FALSE

    @staticmethod
    def _from_label(a: jnp.ndarray):
        return Action()  # TODO: write me


def _step(state: State, action: jnp.ndarray):
    a = Action._from_label(action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))

    state = _flip(state)

    legal_action_mask = _legal_action_mask(state)  # TODO: fix me
    terminated = ~legal_action_mask.any()
    # fmt: off
    reward = jax.lax.select(
        terminated,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # fmt: on
    return state.replace(  # type: ignore
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        reward=reward,
    )


def _step_move(state: State, action: Action) -> State:
    # remove piece from the original position
    board = state.board.at[action.from_].set(EMPTY)
    # capture the opponent if exists
    captured = board[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state.hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by % 5, and
        #   (2) filtering promoted piece by x % 4
        lambda: state.hand.at[0, (captured % 5) % 4].add(1),
    )
    # promote piece (PAWN to GOLD)
    piece = jax.lax.select(action.is_promotion, action.piece + 4, action.piece)
    # set piece to the target position
    board = board.at[action.to].set(piece)
    # apply piece moves
    return state.replace(board=board, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    board = state.board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state.replace(board=board, hand=hand)  # type: ignore


def _legal_action_mask(state: State):
    return jnp.ones(132, dtype=jnp.bool_)

def _flip(state):
    empty_mask = state.board == EMPTY
    board = (state.board + 5) % 10
    board = jnp.where(empty_mask, EMPTY, board)
    board = board[::-1]
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        turn=(state.turn + 1) % 2,
        board=board,
        hand=state.hand[::-1],
    )


def _can_move(piece, from_, to):
    def can_move(piece, from_, to):
        """Can <piece> move from <from_> to <to>?"""
        x0, y0 = from_ // 4, from_ % 4
        x1, y1 = to // 4, to % 4
        dx = x1 - x0
        dy = y1 - y0
        is_neighbour = ((dx != 0) | (dy != 0)) & (jnp.abs(dx) <= 1) & (jnp.abs(dy) <= 1)
        return jax.lax.switch(
            piece,
            [
                lambda: (dx == 0) & (dy == -1),  # PAWN
                lambda: is_neighbour & ((dx == dy) | (dx == -dy)),  # BISHOP
                lambda: is_neighbour & ((dx == 0) | (dy == 0)),  # ROOK
                lambda: is_neighbour,  # KING
                lambda: is_neighbour & ((dx != 0) | (dy != +1))  # GOLD
            ]
        )

    # fmt: off
    # CAN_MOVE[piece, from_, to] = Can <piece> move from <from_> to <to>?
    CAN_MOVE = jax.jit(jax.vmap(jax.vmap(jax.vmap(
        can_move, (None, None, 0)), (None, 0, None)), (0, None, None))
    )(jnp.arange(5), jnp.arange(12), jnp.arange(12))
    # fmt: on

    return CAN_MOVE[piece, from_, to]
