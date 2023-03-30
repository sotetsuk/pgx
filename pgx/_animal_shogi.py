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


# 指し手のdataclass
@dataclass
class JaxAnimalShogiAction:
    # 上の3つは移動と駒打ちで共用
    # 下の3つは移動でのみ使用
    # 駒打ちかどうか
    is_drop: jnp.ndarray = FALSE
    # piece: 動かした(打った)駒の種類
    piece: jnp.ndarray = jnp.int32(0)
    # final: 移動後の座標
    to: jnp.ndarray = jnp.int32(0)
    # 移動前の座標
    from_: jnp.ndarray = jnp.int32(0)
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: jnp.ndarray = jnp.int32(0)
    # is_promote: 駒を成るかどうかの判定
    is_promote: jnp.ndarray = FALSE


INIT_BOARD = jnp.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=jnp.bool_,
)



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
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD
    hand: jnp.ndarray = jnp.zeros((2, 3), dtype=jnp.int8)

