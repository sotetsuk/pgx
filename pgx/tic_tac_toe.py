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


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((3, 3, 2), dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Tic-tac-toe specific ---
    _turn: jnp.ndarray = jnp.int8(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    _board: jnp.ndarray = -jnp.ones(9, jnp.int8)  # -1 (empty), 0, 1

    @property
    def env_id(self) -> v1.EnvId:
        return "tic_tac_toe"


class TicTacToe(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> v1.EnvId:
        return "tic_tac_toe"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(current_player=current_player)  # type:ignore


def _step(state: State, action: jnp.ndarray) -> State:
    state = state.replace(_board=state._board.at[action].set(state._turn))  # type: ignore
    won = _win_check(state._board, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        legal_action_mask=state._board < 0,
        rewards=reward,
        terminated=won | jnp.all(state._board != -1),
        _turn=(state._turn + 1) % 2,
    )


def _win_check(board, turn) -> jnp.ndarray:
    idx = jnp.int8([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
    return ((board[idx] == turn).all(axis=1)).any()


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    @jax.vmap
    def plane(i):
        return (state._board == i).reshape((3, 3))

    # flip if player_id is opposite
    x = jax.lax.cond(
        state.current_player == player_id,
        lambda: jnp.int8([state._turn, 1 - state._turn]),
        lambda: jnp.int8([1 - state._turn, state._turn]),
    )

    return jnp.stack(plane(x), -1)
