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
    legal_action_mask: jnp.ndarray = jnp.ones(64 + 1, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Othello specific ---
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
    passed: jnp.ndarray = FALSE


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
        return "Othello"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


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


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(
        current_player=current_player,
        board=jnp.zeros(64, dtype=jnp.int8)
        .at[28]
        .set(1)
        .at[35]
        .set(1)
        .at[27]
        .set(-1)
        .at[36]
        .set(-1),
        legal_action_mask=jnp.zeros(64 + 1, dtype=jnp.bool_)
        .at[19]
        .set(TRUE)
        .at[26]
        .set(TRUE)
        .at[37]
        .set(TRUE)
        .at[44]
        .set(TRUE),
    )  # type:ignore


def _step(state, action):
    board = state.board
    my = board > 0
    opp = board < 0

    # updates at out-of-bounds indices will be skipped:
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    # Therefore, there is no need to ignore the path action
    pos = jnp.zeros(64, dtype=jnp.bool_).at[action].set(TRUE)

    shifts = jnp.array([1, -1, 8, -8, 7, -7, 9, -9])
    masks = jnp.array(
        [
            LR_MASK,
            LR_MASK,
            UD_MASK,
            UD_MASK,
            SIDE_MASK,
            SIDE_MASK,
            SIDE_MASK,
            SIDE_MASK,
        ]
    )

    def _shift(i, rev):
        tmp = _check_line(pos, opp, shifts[i], masks[i])
        return jax.lax.cond(
            (jnp.roll(tmp, shifts[i]) & my).any(),
            lambda: rev | tmp,
            lambda: rev,
        )

    rev = jax.lax.fori_loop(0, 8, _shift, jnp.zeros(64, dtype=jnp.bool_))
    my ^= pos | rev
    opp ^= rev
    emp = ~(my | opp)

    def _make_legal(i, legal):
        # NOT _check_line(my, opp, shifts[i], masks[i])
        # because this generates a legal action for the next turn
        tmp = _check_line(opp, my, shifts[i], masks[i])
        tmp = jnp.roll(tmp, shifts[i]) & emp
        return legal | tmp

    legal_action = jax.lax.fori_loop(
        0, 8, _make_legal, jnp.zeros(64, dtype=jnp.bool_)
    )

    reward, terminated = jax.lax.cond(
        (
            (jnp.count_nonzero(my | opp) == 64)
            | ~opp.any()
            | (state.passed & (action == 64))
        ),
        lambda: (_get_reward(my, opp, state.current_player), TRUE),
        lambda: (jnp.zeros(2, jnp.float32), FALSE),
    )

    return state.replace(
        current_player=1 - state.current_player,
        turn=1 - state.turn,
        legal_action_mask=state.legal_action_mask.at[:64]
        .set(legal_action)
        .at[64]
        .set(~legal_action.any()),
        reward=reward,
        terminated=terminated,
        board=-jnp.where(jnp.int8(opp), -1, jnp.int8(my)),
        passed=action == 64,
    )


def _check_line(pos, opp, shift, mask):
    _opp = opp & mask
    result = _opp & jnp.roll(pos, shift)
    result |= _opp & jnp.roll(result, shift)
    result |= _opp & jnp.roll(result, shift)
    result |= _opp & jnp.roll(result, shift)
    result |= _opp & jnp.roll(result, shift)
    result |= _opp & jnp.roll(result, shift)
    return result


def _get_reward(my, opp, curr_player):
    my = jnp.count_nonzero(my)
    opp = jnp.count_nonzero(opp)
    winner = jax.lax.cond(
        my > opp, lambda: curr_player, lambda: 1 - curr_player
    )
    return jax.lax.cond(
        my == opp,
        lambda: jnp.zeros(2, jnp.float32),
        lambda: jnp.float32([-1, -1]).at[winner].set(1),
    )


def _observe(state, player_id) -> jnp.ndarray:
    board = jax.lax.cond(
        player_id == state.current_player,
        lambda: state.board.reshape((8, 8)),
        lambda: (state.board * -1).reshape((8, 8)),
    )

    def make(color):
        return board * color > 0

    return jnp.stack(jax.vmap(make)(jnp.int8([1, -1])), -1)


def _get_abs_board(state):
    return jax.lax.cond(
        state.turn == 0, lambda: state.board, lambda: state.board * -1
    )
