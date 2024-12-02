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
from pgx._src.games.hex import Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((11, 11, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(11 * 11 + 1, dtype=jnp.bool_).at[-1].set(FALSE)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "hex"


class Hex(core.Env):
    def __init__(self, *, size: int = 11):
        super().__init__()
        assert isinstance(size, int)
        self.size = size
        self._game = Game(size=size)

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        return State(_x=self._game.init(), current_player=current_player)  # type:ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        terminated = self._game.is_terminal(x)
        reward = jax.lax.cond(
            terminated,
            lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
            lambda: jnp.zeros(2, jnp.float32),
        )

        return state.replace(  # type:ignore
            current_player=1 - state.current_player,
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=reward,
            terminated=terminated,
            _x=x,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = jax.lax.select(player_id == state.current_player, state._x.color, 1 - state._x.color)
        return self._game.observe(state._x, color)

    @property
    def id(self) -> core.EnvId:
        return "hex"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
