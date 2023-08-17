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

import pgx.v1 as v1
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

EMPTY = jnp.int8(0)
GOOD_GHOST = jnp.int8(1)
BAD_GHOST = jnp.int8(2)


# board index (white view)
# 6  5 11 17 23 29 35
# 5  4 10 16 22 28 34
# 4  3  9 15 21 27 33
# 3  2  8 14 20 26 32
# 2  1  7 13 19 25 31
# 1  0  6 12 18 24 30
#    a  b  c  d  e  f
# board index (flipped black view)
# 6  0  6 12 18 24 30
# 5  1  7 13 19 25 31
# 4  2  8 14 20 26 32
# 3  3  9 15 21 27 33
# 2  4 10 16 22 28 34
# 1  5 11 17 23 29 35
#    a  b  c  d  e  f
# fmt: off
INIT_BOARD = jnp.int8([
    0,  0,  0,  0,  0,  0,
    1,  2,  0,  0, -2, -1,
    1,  2,  0,  0, -2, -1,
    1,  2,  0,  0, -2, -1,
    1,  2,  0,  0, -2, -1,
    0,  0,  0,  0,  0,  0,
])


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((6, 6, 5), dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(144, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- geister specific ---
    _turn: jnp.ndarray = jnp.int8(0)
    _board: jnp.ndarray = INIT_BOARD
    # -2(opponent bad ghost), -1 (opponennt good ghost), 0(empty), 1 (my good ghost), 2(my bad ghost)

    @property
    def env_id(self) -> v1.EnvId:
        return "geister"


# DOWN, UP, LEFT, RIGHT
DISTANCE = jnp.int8([-1, 1, -6, 6])


# Action
# directions(4:DOWN, UP, LEFT, RIGHT) × positions(36)
class Action:
    from_: jnp.ndarray = jnp.int8(-1)
    to: jnp.ndarray = jnp.int8(-1)

    @staticmethod
    def _from_label(label: jnp.ndarray):
        direction, from_ = label // 36, label % 36
        return Action(  # type: ignore
            from_=from_,
            to=from_ + DISTANCE[direction],
        )

    def _to_label(self):
        distance = self.to - self.from_
        direction = -1
        if distance == -1:
            direction = 0
        elif distance == 1:
            direction = 1
        elif distance == -6:
            direction = 2
        elif distance == 6:
            direction = 3
        return jnp.int32(self.from_) + jnp.int32(36 * direction)


class Geister(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        return state
    #     assert isinstance(state, State)
    #     state = _step(state, action)
    #     state = jax.lax.cond(
    #         MAX_TERMINATION_STEPS <= state._step_count,
    #         # end with tie
    #         lambda: state.replace(terminated=TRUE),  # type: ignore
    #         lambda: state,
    #     )
    #     return state  # type: ignore

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((6, 6, 5), dtype=jnp.bool_)
    #     assert isinstance(state, State)
    #     return _observe(state, player_id)

    @property
    def id(self) -> v1.EnvId:
        # need new id
        return "geister"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(current_player=current_player)  # type:ignore
