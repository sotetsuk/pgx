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
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
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
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        x = self._game.init()
        return State(current_player=_player_order[x.color], _player_order=_player_order, _x=x)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        return state.replace(  # type:ignore
            current_player=state._player_order[x.color],
            legal_action_mask=self._game.legal_action_mask(x),
            terminated=self._game.is_terminal(x),
            rewards=self._game.rewards(x)[state._player_order],
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
