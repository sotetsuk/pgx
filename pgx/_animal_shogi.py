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

from pgx._flax.serialization import from_bytes
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# -1: EMPTY
#  0: PAWN
#  1: ROOK
#  2: BISHOP
#  3: KING
#  4: GOLD
#  5: OPP_PAWN
#  6: OPP_ROOK
#  7: OPP_BISHOP
#  8: OPP_KING
#  9: OPP_GOLD
INIT_BOARD = jnp.int8([7, -1, -1, 1, 8, 5, 0, 3, 6, -1, -1, 2])  # (12,)


@dataclass
class JaxAnimalShogiState:
    current_player: jnp.ndarray = jnp.int8(0)
    # --- Animal Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD
    hand: jnp.ndarray = jnp.zeros((2, 3), dtype=jnp.int8)


# Implements AlphaZero like action:
# 132 =
#   [Move] 12 (from_) * 8 (to) +
#   [Drop] 12 (to) * 3 (piece_type)
@dataclass
class Action:
    is_drop: jnp.ndarray = FALSE
    piece: jnp.ndarray = jnp.int8(0)
    from_: jnp.ndarray = jnp.int8(0)
    to: jnp.ndarray = jnp.int8(0)
    is_promotion: jnp.ndarray = FALSE


def _can_move(piece, from_, to):
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
    _can_move, (None, None, 0)), (None, 0, None)), (0, None, None))
)(jnp.arange(5), jnp.arange(12), jnp.arange(12))
# fmt: on
