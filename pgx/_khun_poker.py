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
    observation: jnp.ndarray = jnp.zeros((8, 8, 2), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(4, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Khun poker specific ---
    card: jnp.ndarray = jnp.int8([-1, -1])
    # [(player 0),(player 1)]
    last_action: jnp.ndarray = jnp.int8(-1)
    # 0(Call)  1(Bet)  2(Fold)  3(Check)


class KhunPoker(core.Env):
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
        return "KhunPoker"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2 = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(rng1))
    init_card = jax.random.choice(
        rng2, jnp.int8([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
    )
    return State(
        current_player=current_player,
        card=init_card,
        legal_action_mask=jnp.bool_([0, 0, 1, 1]),
    )


def _step(state, action):
    ...


def _observe(state, player_id) -> jnp.ndarray:
    ...
