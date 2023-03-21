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
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(16, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32(0.0)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(4, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- 2048 specific ---
    turn: jnp.ndarray = jnp.int8(0)
    # 4x4 board
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    board: jnp.ndarray = jnp.zeros(16, jnp.int8)
    #  Board is expressed as a power of 2.
    # e.g.
    # [[ 0,  0,  1,  1],
    #  [ 1,  0,  1,  2],
    #  [ 3,  3,  6,  7],
    #  [ 3,  6,  7,  9]]
    # means
    # [[ 0,  0,  2,  2],
    #  [ 2,  0,  2,  4],
    #  [ 8,  8, 64,128],
    #  [ 8, 64,128,512]]


class Play2048(core.Env):
    def __init__(
        self,
    ):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "2048"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 1


def _init(rng: jax.random.KeyArray) -> State:
    return State()


def _step(state, action):
    ...


def _observe(state, player_id) -> jnp.ndarray:
    ...


# only for debug
def show(state):
    board = jnp.array([0 if i == 0 else 2**i for i in state.board])
    print(board.reshape((4, 4)))
