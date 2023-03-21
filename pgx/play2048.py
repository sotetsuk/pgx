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
ZERO = jnp.int8(0)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(16, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(4, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- 2048 specific ---
    turn: jnp.ndarray = jnp.int8(0)
    # 4x4 board
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    board: jnp.ndarray = jnp.zeros(16, jnp.int8)
    #  Board is expressed as a power of 2.
    # e.g.
    # [[ 0,  0,  1,  1],
    #  [ 1,  0,  1,  2],
    #  [ 3,  3,  6,  7],
    #  [ 3,  6,  7,  9]]
    # means
    # [[ 0,  0,  2,  2],
    #  [ 2,  0,  2,  4],
    #  [ 8,  8, 64,128],
    #  [ 8, 64,128,512]]


class Play2048(core.Env):
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
        return "2048"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 1


def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2 = jax.random.split(rng)
    board = _add_random_num(jnp.zeros((4, 4), jnp.int8), rng1)
    board = _add_random_num(board, rng2)
    return State(board=board.ravel())  # type:ignore


def _step(state: State, action):
    """action: 0(left), 1(up), 2(right), 3(down)"""
    board_2d = state.board.reshape((4, 4))
    board_2d = jax.lax.switch(
        action,
        [
            lambda: board_2d,
            lambda: jnp.rot90(board_2d, 1),
            lambda: jnp.rot90(board_2d, 2),
            lambda: jnp.rot90(board_2d, 3),
        ],
    )
    board_2d, reward = jax.vmap(_slide_and_merge)(board_2d)
    board_2d = jax.lax.switch(
        action,
        [
            lambda: board_2d,
            lambda: jnp.rot90(board_2d, -1),
            lambda: jnp.rot90(board_2d, -2),
            lambda: jnp.rot90(board_2d, -3),
        ],
    )

    _rng_key, sub_key = jax.random.split(state._rng_key)
    board_2d = _add_random_num(board_2d, sub_key)

    legal_action = jax.vmap(_can_slide_left)(
        jnp.array(
            [
                board_2d,
                jnp.rot90(board_2d, 1),
                jnp.rot90(board_2d, 2),
                jnp.rot90(board_2d, 3),
            ]
        )
    )

    return state.replace(  # type:ignore
        _rng_key=_rng_key,
        board=board_2d.ravel(),
        reward=jnp.float32([reward.sum()]),
        legal_action_mask=legal_action.ravel(),
        terminated=~legal_action.any(),
    )


def _observe(state: State, player_id) -> jnp.ndarray:
    obs = jnp.zeros((16, 31), dtype=jnp.bool_)
    obs = jax.lax.fori_loop(
        0, 16, lambda i, obs: obs.at[i, state.board[i]].set(TRUE), obs
    )
    return obs.reshape((4, 4, 31))


def _add_random_num(board_2d, key):
    """Add 2 or 4 to the empty space on the board.
    2 appears 90% of the time, and 4 appears 10% of the time.
    cf. https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js#L71
    """
    key, sub_key = jax.random.split(key)
    pos = jax.random.choice(key, jnp.arange(16), p=(board_2d.ravel() == 0))
    set_num = jax.random.choice(
        sub_key, jnp.int8([1, 2]), p=jnp.array([0.9, 0.1])
    )
    board_2d = board_2d.at[pos // 4, pos % 4].set(set_num)
    return board_2d


def _slide_and_merge(line):
    """[2 2 2 2] -> [4 4 0 0]"""
    line = _slide_left(line)
    line, reward = _merge(line)
    line = _slide_left(line)
    return line, reward


def _merge(line):
    """[2 2 2 2] -> [4 0 4 0]"""
    line, reward = jax.lax.cond(
        (line[0] != 0) & (line[0] == line[1]),
        lambda: (
            line.at[0].set(line[0] + 1).at[1].set(ZERO),
            2 ** (line[0] + 1),
        ),
        lambda: (line, ZERO),
    )
    line, reward = jax.lax.cond(
        (line[1] != 0) & (line[1] == line[2]),
        lambda: (
            line.at[1].set(line[1] + 1).at[2].set(ZERO),
            reward + 2 ** (line[1] + 1),
        ),
        lambda: (line, reward),
    )
    line, reward = jax.lax.cond(
        (line[2] != 0) & (line[2] == line[3]),
        lambda: (
            line.at[2].set(line[2] + 1).at[3].set(ZERO),
            reward + 2 ** (line[2] + 1),
        ),
        lambda: (line, reward),
    )
    return line, reward


def _slide_left(line):
    """[0 2 0 2] -> [2 2 0 0]"""
    line = jax.lax.cond(
        (line[2] == 0),
        lambda: line.at[2:].set(jnp.roll(line[2:], -1)),
        lambda: line,
    )
    line = jax.lax.cond(
        (line[1] == 0),
        lambda: line.at[1:].set(jnp.roll(line[1:], -1)),
        lambda: line,
    )
    line = jax.lax.cond(
        (line[0] == 0),
        lambda: jnp.roll(line, -1),
        lambda: line,
    )
    return line


def _can_slide_left(board_2d):
    def _can_slide(line):
        """Judge if it can be moved to the left."""
        can_slide = (line[:3] == 0).any()
        can_slide |= (
            (line[0] == line[1]) | (line[1] == line[2]) | (line[2] == line[3])
        )
        can_slide &= ~(line == 0).all()
        return can_slide

    can_slide = jax.vmap(_can_slide)(board_2d).any()
    return can_slide


# only for debug
def show(state):
    board = jnp.array([0 if i == 0 else 2**i for i in state.board])
    print(board.reshape((4, 4)))
