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
from pgx._src.shogi_utils import (
    INIT_LEGAL_ACTION_MASK,
    _from_sfen,
    _to_sfen,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx._src.games.shogi import MAX_TERMINATION_STEPS, GameState, Game, _observe, _flip


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

    @staticmethod
    def _from_board(turn, piece_board: Array, hand: Array):
        """Mainly for debugging purpose.
        terminated, reward, and current_player are not changed"""
        state = State(_x=GameState(turn=turn, board=piece_board, hand=hand))  # type: ignore
        # fmt: off
        state = jax.lax.cond(turn % 2 == 1, lambda: state.replace(_x=_flip(state._x)), lambda: state)  # type: ignore
        # fmt: on
        return state.replace(legal_action_mask=Game().legal_action_mask(state._x))  # type: ignore

    @staticmethod
    def _from_sfen(sfen):
        turn, pb, hand, step_count = _from_sfen(sfen)
        return jax.jit(State._from_board)(turn, pb, hand).replace(_step_count=jnp.int32(step_count))  # type: ignore

    def _to_sfen(self):
        state = self if self._x.turn % 2 == 0 else self.replace(_x=_flip(self._x))  # type: ignore
        return _to_sfen(state)


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
        legal_action_mask = self._game.legal_action_mask(state._x)
        terminated = ~legal_action_mask.any()
        # fmt: off
        reward = jax.lax.select(
            terminated,
            jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )
        # fmt: on
        state = state.replace(  # type: ignore
            legal_action_mask=legal_action_mask,
            terminated=terminated,
            rewards=reward,
        )
        state = jax.lax.cond(
            (MAX_TERMINATION_STEPS <= state._step_count),
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