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

from typing import Optional

import jax
from jax import numpy as jnp

import pgx._src.games.go as go
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(19 * 19 + 1, dtype=jnp.bool_)
    observation: Array = jnp.zeros((19, 19, 17), dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    _x: go.GameState = go.GameState()

    @property
    def _size(self) -> int:
        return int(jnp.sqrt(self._x.board.shape[-1]).astype(jnp.int32))

    @property
    def env_id(self) -> core.EnvId:
        return f"go_{self._size}x{self._size}"  # type: ignore


class Go(core.Env):
    def __init__(
        self,
        *,
        size: int = 19,
        komi: float = 7.5,
        history_length: int = 8,
        max_terminal_steps: Optional[int] = None,
    ):
        super().__init__()
        assert isinstance(size, int)
        self._game = go.Game(
            size=size, komi=komi, history_length=history_length, max_termination_steps=max_terminal_steps
        )

    def _init(self, key: PRNGKey) -> State:
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        x = self._game.init()
        size = self._game.size
        return State(  # type:ignore
            current_player=_player_order[x.color],
            legal_action_mask=jnp.ones(size * size + 1, dtype=jnp.bool_),
            _player_order=_player_order,
            _x=x,
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        return state.replace(  # type:ignore
            current_player=state._player_order[x.color],
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=self._game.rewards(x)[state._player_order],
            terminated=self._game.is_terminal(x),
            _x=x,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        curr_color = state._x.color
        my_turn = jax.lax.select(player_id == state.current_player, curr_color, 1 - curr_color)
        return self._game.observe(state._x, my_turn)

    @property
    def id(self) -> core.EnvId:
        return f"go_{int(self._game.size)}x{int(self._game.size)}"  # type: ignore

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 2
