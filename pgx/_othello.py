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
# fmt:off
LR_MASK = jnp.array([
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.bool_)
UD_MASK = jnp.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.bool_)
# fmt:on
SIDE_MASK = LR_MASK & UD_MASK


@dataclass
class State(core.State):
    steps: jnp.ndarray = jnp.int32(0)
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(63, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # ---
    turn: jnp.ndarray = jnp.int8(0)
    # 8x8 board
    # [[ 0,  1,  2,  3,  4,  5,  6,  7],
    #  [ 8,  9, 10, 11, 12, 13, 14, 15],
    #  [16, 17, 18, 19, 20, 21, 22, 23],
    #  [24, 25, 26, 27, 28, 29, 30, 31],
    #  [32, 33, 34, 35, 36, 37, 38, 39],
    #  [40, 41, 42, 43, 44, 45, 46, 47],
    #  [48, 49, 50, 51, 52, 53, 54, 55],
    #  [56, 57, 58, 59, 60, 61, 62, 63]]
    board: jnp.ndarray = jnp.zeros(64, jnp.int8)  # -1(opp), 0(empty), 1(self)


class Othello(core.Env):
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
        return "Connect Four"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(
        current_player=current_player,
        board=jnp.zeros(64, dtype=jnp.int8)
        .at[27]
        .set(1)
        .at[36]
        .set(1)
        .at[28]
        .set(-1)
        .at[35]
        .set(-1),
    )  # type:ignore


def _step(state, action):
    action = jnp.int8(action)
    board = state.board
    my = board > 0
    opp = board < 0
    pos = jnp.zeros(64, dtype=jnp.bool_).at[action].set(TRUE)

    shifts = jnp.array([1, -1, 8, -8])
    masks = jnp.array([LR_MASK, LR_MASK, UD_MASK, UD_MASK])

    def _shift(i, rev):
        tmp = check_line(pos, opp, shifts[i], masks[i])
        return jax.lax.cond(
            (jnp.roll(tmp, shifts[i]) & my).any(),
            lambda: rev | tmp,
            lambda: rev,
        )

    rev = jax.lax.fori_loop(0, 4, _shift, jnp.zeros(64, dtype=jnp.bool_))

    # TODO
    my ^= pos | rev
    opp ^= rev

    return state.replace(board=jnp.where(jnp.int8(opp), -1, jnp.int8(my)))


def check_line(pos, opp, shift, mask):
    result = opp & mask
    result = result & jnp.roll(pos, shift)
    result |= opp & jnp.roll(result, shift)
    result |= opp & jnp.roll(result, shift)
    result |= opp & jnp.roll(result, shift)
    result |= opp & jnp.roll(result, shift)
    result |= opp & jnp.roll(result, shift)
    return result


def _observe(state, player_id) -> jnp.ndarray:
    ...
