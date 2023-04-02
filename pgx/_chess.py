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

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
KNIGHT = jnp.int8(1)
BISHOP = jnp.int8(2)
ROOK = jnp.int8(3)
QUEEN = jnp.int8(4)
KING = jnp.int8(5)
# OPP_PAWN = 6
# OPP_KNIGHT = 7
# OPP_BISHOP = 8
# OPP_ROOK = 9
# OPP_QUEEN = 10
# OPP_KING = 11

INIT_BOARD = jnp.int8([
    9,  7,  8, 11, 10,  8,  7,  9,
    6,  6,  6,  6,  6,  6,  6,  6,
   -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,
    3,  1,  2,  5,  4,  2,  1,  3
])


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(73 * 64, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((8, 8, 119), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Chess specific ---
    turn: jnp.ndarray = jnp.int8(0)
    # 空白,WPawn,WKnight,WBishop,WRook,WQueen,WKing,BPawn,BKnight,BBishop,BRook,BQueen,BKing
    # の順で駒がどの位置にあるかをone_hotで記録
    # 左上からFENと同じ形式で埋めていく
    board: jnp.ndarray = INIT_BOARD
