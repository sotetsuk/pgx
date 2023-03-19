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

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    steps: jnp.ndarray = jnp.int32(0)
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(27, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # ---
    turn: jnp.ndarray = jnp.int8(0)
    # 8x8 board
    # [[ 0,  1,  2,  3,  4,  5,  6,  7],
    #  [ 8,  9, 10, 11, 12, 13, 14, 15],
    #  [16, 17, 18, 19, 20, 21, 22, 23],
    #  [24, 25, 26, 27, 28, 29, 30, 31],
    #  [32, 33, 34, 35, 36, 37, 38, 39],
    #  [40, 41, 42, 43, 44, 45, 46, 47],
    #  [48, 49, 50, 51, 52, 53, 54, 55],
    #  [56, 57, 58, 59, 60, 61, 62, 63]]
    board: jnp.ndarray = -jnp.ones(64, jnp.int8)  # -1 (empty), 0, 1


def step(board, action):
    my = board > 0
    opp = board < 0
    pos = jnp.zeros(64, dtype=jnp.bool_).at[action].set(TRUE)

    rev = jnp.zeros(64, dtype=jnp.bool_)
    tmp = line_left(pos, opp)
    rev = jax.lax.cond(
        (jnp.roll(tmp, 1) & my).any(), lambda: rev | tmp, lambda: rev
    )
    tmp = line_right(pos, opp)
    rev = jax.lax.cond(
        (jnp.roll(tmp, -1) & my).any(), lambda: rev | tmp, lambda: rev
    )
    my ^= pos | rev
    opp ^= rev

    return jnp.where(jnp.int8(opp), -1, jnp.int8(my))


def line_left(pos, opp):
    # fmt: off
    mask = opp & jnp.array([
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
    ], dtype=jnp.bool_)
    # fmt: on
    result = mask & jnp.roll(pos, 1)
    result |= opp & jnp.roll(result, 1)
    result |= opp & jnp.roll(result, 1)
    result |= opp & jnp.roll(result, 1)
    result |= opp & jnp.roll(result, 1)
    result |= opp & jnp.roll(result, 1)
    return result


def line_right(pos, opp):
    # fmt: off
    mask = opp & jnp.array([
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
    ], dtype=jnp.bool_)
    # fmt: on
    result = mask & jnp.roll(pos, -1)
    result |= opp & jnp.roll(result, -1)
    result |= opp & jnp.roll(result, -1)
    result |= opp & jnp.roll(result, -1)
    result |= opp & jnp.roll(result, -1)
    result |= opp & jnp.roll(result, -1)
    return result


# fmt: off
board = jnp.array([
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, -1, 0, 0, 0,
        0, 0, 0, -1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=jnp.int8)
"""
pos = jnp.array([
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=jnp.bool_)
"""
# fmt:on
action = 29
board = jax.jit(step)(board, action)
print(board.reshape((8, 8)))
