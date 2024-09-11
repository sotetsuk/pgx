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
from pgx._src.games.chess import GameState, _observe, _rewards, _step, _is_terminated, _flip
from pgx._src.games.chess_utils import INIT_LEGAL_ACTION_MASK
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

INIT_ZOBRIST_HASH = jnp.uint32([1172276016, 1112364556])
MAX_TERMINATION_STEPS = 512  # from AZ paper

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int32(0)
PAWN = jnp.int32(1)
KNIGHT = jnp.int32(2)
BISHOP = jnp.int32(3)
ROOK = jnp.int32(4)
QUEEN = jnp.int32(5)
KING = jnp.int32(6)
# OPP_PAWN = -1
# OPP_KNIGHT = -2
# OPP_BISHOP = -3
# OPP_ROOK = -4
# OPP_QUEEN = -5
# OPP_KING = -6


# board index (white view)
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
# board index (flipped black view)
# 8  0  8 16 24 32 40 48 56
# 7  1  9 17 25 33 41 49 57
# 6  2 10 18 26 34 42 50 58
# 5  3 11 19 27 35 43 51 59
# 4  4 12 20 28 36 44 52 60
# 3  5 13 21 29 37 45 53 61
# 2  6 14 22 30 38 46 54 62
# 1  7 15 23 31 39 47 55 63
#    a  b  c  d  e  f  g  h
# fmt: off
INIT_BOARD = jnp.int32([
    4, 1, 0, 0, 0, 0, -1, -4,
    2, 1, 0, 0, 0, 0, -1, -2,
    3, 1, 0, 0, 0, 0, -1, -3,
    5, 1, 0, 0, 0, 0, -1, -5,
    6, 1, 0, 0, 0, 0, -1, -6,
    3, 1, 0, 0, 0, 0, -1, -3,
    2, 1, 0, 0, 0, 0, -1, -2,
    4, 1, 0, 0, 0, 0, -1, -4
])
# fmt: on

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
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

    def _init(self, key: PRNGKey) -> State:
        x = GameState()
        _player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
        state = State(  # type: ignore
            current_player=_player_order[x.turn],
            _player_order=_player_order,
            _x=x,
        )
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = _step(state._x, action)
        state = state.replace(  # type: ignore
            _x=x,
            legal_action_mask=x.legal_action_mask,
            terminated=_is_terminated(x),
            rewards=_rewards(x)[state._player_order],
            current_player=state._player_order[x.turn],
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = jax.lax.select(state.current_player == player_id, state._x.turn, 1 - state._x.turn)
        x = jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x))
        return _observe(x, color)

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
