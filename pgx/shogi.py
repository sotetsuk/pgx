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
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx._src.games.shogi import MAX_TERMINATION_STEPS, GameState, Game, _observe, INIT_LEGAL_ACTION_MASK


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # (27 * 81,)
    observation: Array = jnp.zeros((119, 9, 9), dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "shogi"


class Shogi(core.Env):

    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        state = State()
        current_player = jnp.int32(jax.random.bernoulli(key))
        return state.replace(current_player=current_player)  # type: ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        # Note: Assume that illegal action is already filtered by Env.step
        x = self._game.step(state._x, action)
        state = state.replace(  # type: ignore
            current_player=(state.current_player + 1) % 2,
            _x=x,
        )
        del x
        terminated = ~state._x.legal_action_mask.any()
        # fmt: off
        reward = jax.lax.select(
            terminated,
            jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )
        # fmt: on
        state = state.replace(  # type: ignore
            legal_action_mask=state._x.legal_action_mask,
            terminated=terminated,
            rewards=reward,
        )
        state = jax.lax.cond(
            (MAX_TERMINATION_STEPS <= state._x.step_count),
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state._x, flip=state.current_player == player_id)

    @property
    def id(self) -> core.EnvId:
        return "shogi"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2