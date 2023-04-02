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


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    turn: jnp.ndarray = jnp.int8(0)
    # 空白,WPawn,WKnight,WBishop,WRook,WQueen,WKing,BPawn,BKnight,BBishop,BRook,BQueen,BKing
    # の順で駒がどの位置にあるかをone_hotで記録
    board: jnp.ndarray = jnp.zeros(64, dtype=jnp.int8)
