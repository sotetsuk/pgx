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


from functools import partial

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._shogi_utils import *
from pgx._shogi_utils import _flip, _from_sfen, _to_sfen



@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(81 * 81, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((119, 9, 9), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,) 後手のときにはflipする
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)  # 後手のときにはflipする

    @staticmethod
    def _from_board(turn, piece_board: jnp.ndarray, hand: jnp.ndarray):
        """Mainly for debugging purpose.
        terminated, reward, and current_player are not changed"""
        state = State(turn=turn, piece_board=piece_board, hand=hand)  # type: ignore
        # fmt: off
        state = jax.lax.cond(turn % 2 == 1, lambda: _flip(state), lambda: state)
        # fmt: on
        return state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore

    @staticmethod
    def _from_sfen(sfen):
        turn, pb, hand = _from_sfen(sfen)
        return State._from_board(turn, pb, hand)

    def _to_sfen(self):
        return _to_sfen(self)


class Shogi(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        state = _init_board()
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        return state.replace(current_player=current_player)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        # Note: Assume that illegal action is already filtered by Env.step
        return _step(state, action)

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "Shogi"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def _init_board():
    """Initialize Shogi State.
    >>> s = _init_board()
    >>> s.piece_board.reshape((9, 9))
    Array([[15, -1, 14, -1, -1, -1,  0, -1,  1],
           [16, 18, 14, -1, -1, -1,  0,  5,  2],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [21, -1, 14, -1, -1, -1,  0, -1,  7],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [16, 19, 14, -1, -1, -1,  0,  4,  2],
           [15, -1, 14, -1, -1, -1,  0, -1,  1]], dtype=int8)
    >>> jnp.rot90(s.piece_board.reshape((9, 9)), k=3)
    Array([[15, 16, 17, 20, 21, 20, 17, 16, 15],
           [-1, 19, -1, -1, -1, -1, -1, 18, -1],
           [14, 14, 14, 14, 14, 14, 14, 14, 14],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [-1,  4, -1, -1, -1, -1, -1,  5, -1],
           [ 1,  2,  3,  6,  7,  6,  3,  2,  1]], dtype=int8)
    """
    state = State()
    return state.replace(  # type: ignore
        legal_action_mask=_legal_action_mask(state)
    )



def _legal_action_mask(state: State):
    mask = jax.vmap(partial(_is_legal_move, board=state.piece_board))(move=jnp.arange(81 * 81))
    return mask


def _is_legal_move(board: jnp.ndarray, move: jnp.ndarray):
    from_, to = move // 81, move % 81
    # destination is my piece
    is_illegal = (PAWN <= board[to]) & (board[to] < OPP_PAWN)
    # piece cannot move like that
    piece = board[from_]
    is_illegal |= ~CAN_MOVE[piece, from_, to]
    # there is an obstacle between from_ and to
    i = _to_large_piece_ix(piece)
    is_illegal |= ((i >= 0) & (BETWEEN[i, from_, to, :] & (board != EMPTY)).any())

    # actually move
    board = board.at[from_].set(EMPTY).at[to].set(piece)

    # suicide move （王手放置、自殺手）
    king_pos = jnp.nonzero(board == KING, size=1)[0][0]

    # 大駒
    @jax.vmap
    def can_capture_king(f):
        p = _flip_piece(board[f])  # 敵の大駒
        i = _to_large_piece_ix(p)  # 敵の大駒のix
        return ((i >= 0) &  # 敵の大駒かつ
                (CAN_MOVE[p, king_pos, f]) &  # 移動可能で
                ((BETWEEN[i, king_pos, f, :] & (board != EMPTY)).sum() == 0))  # 障害物なし

    is_illegal |= can_capture_king(jnp.arange(81)).any()
    return ~is_illegal


def _flip_piece(piece):
    return jax.lax.select(piece >= 0, (piece + 14) % 28, piece)

def _to_large_piece_ix(piece):
    # fmt: off
    ixs = (-jnp.ones(28, dtype=jnp.int8)) \
            .at[LANCE].set(0) \
            .at[BISHOP].set(1) \
            .at[ROOK].set(2) \
            .at[HORSE].set(1) \
            .at[DRAGON].set(2)
    # fmt: on
    return jax.lax.select(piece >= 0, ixs[piece], jnp.int8(-1))


def _observe(state: State, player_id: jnp.ndarray):
    return jnp.zeros_like(state.observation)


def _step(state, action):
    return state