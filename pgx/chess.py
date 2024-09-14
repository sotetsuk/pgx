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

import warnings

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.chess import INIT_LEGAL_ACTION_MASK, Game, GameState, _flip
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # 64 * 73 = 4672
    observation: Array = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "chess"

    @staticmethod
    def _from_fen(fen: str):
        from pgx.experimental.chess import from_fen

        warnings.warn(
            "State._from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
            DeprecationWarning,
        )
        return from_fen(fen)

    def _to_fen(self) -> str:
        from pgx.experimental.chess import to_fen

        warnings.warn(
            "State._to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
            DeprecationWarning,
        )
        return to_fen(self)


class Chess(core.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()

    def _init(self, key: PRNGKey) -> State:
        x = GameState()
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        state = State(  # type: ignore
            current_player=_player_order[x.color],
            _player_order=_player_order,
            _x=x,
        )
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self.game.step(state._x, action)
        state = state.replace(  # type: ignore
            _x=x,
            legal_action_mask=x.legal_action_mask,
            terminated=self.game.is_terminal(x),
            rewards=self.game.rewards(x)[state._player_order],
            current_player=state._player_order[x.color],
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = jax.lax.select(state.current_player == player_id, state._x.color, 1 - state._x.color)
        x = jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x))
        return self.game.observe(x, color)

    @property
    def id(self) -> core.EnvId:
        return "chess"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def num_players(self) -> int:
        return 2


def _from_fen(fen: str):
    from pgx.experimental.chess import from_fen

    warnings.warn(
        "_from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
        DeprecationWarning,
    )
    return from_fen(fen)


def _to_fen(state: State):
    from pgx.experimental.chess import to_fen

    warnings.warn(
        "_to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
        DeprecationWarning,
    )
    return to_fen(state)
