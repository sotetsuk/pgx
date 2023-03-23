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


@dataclass
class Action:
    is_drop: jnp.ndarray
    piece: jnp.ndarray
    to: jnp.ndarray
    # --- Optional (only for move action) ---
    from_: jnp.ndarray = jnp.int8(0)
    is_promotion: jnp.ndarray = FALSE

    @staticmethod
    def make_move(piece, from_, to, is_promotion=FALSE):
        return Action(
            is_drop=FALSE,
            piece=piece,
            from_=from_,
            to=to,
            is_promotion=is_promotion,
        )

    @staticmethod
    def make_drop(piece, to):
        return Action(is_drop=TRUE, piece=piece, to=to)

    @staticmethod
    def _from_dlshogi_action(state: State, action: jnp.ndarray):
        action = jnp.int32(action)
        direction, to = jnp.int8(action // 81), jnp.int8(action % 81)
        is_drop = direction >= 20
        is_promotion = (10 <= direction) & (direction < 20)
        # LEGAL_FROM_IDX[UP, 19] = [20, 21, ... -1]
        legal_from_idx = LEGAL_FROM_IDX[direction % 10, to]  # (81,)
        from_cand = state.piece_board[legal_from_idx]  # (8,)
        mask = (
            (legal_from_idx >= 0)
            & (PAWN <= from_cand)
            & (from_cand < OPP_PAWN)
        )
        i = jnp.nonzero(mask, size=1)[0][0]
        from_ = legal_from_idx[i]
        piece = jax.lax.cond(
            is_drop,
            lambda: direction - 20,
            lambda: state.piece_board[from_],
        )
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promotion=is_promotion)  # type: ignore


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
    return state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore


def _step(state: State, action: jnp.ndarray):
    action = Action._from_dlshogi_action(state, action)
    # apply move/drop action
    state = jax.lax.cond(
        action.is_drop, _step_drop, _step_move, *(state, action)
    )
    # flip state
    state = _flip(state)
    state = state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        turn=(state.turn + 1) % 2,
        legal_action_mask=_legal_action_mask(state),
    )
    return state


def _step_move(state: State, action: Action) -> State:
    pb = state.piece_board
    # remove piece from the original position
    pb = pb.at[action.from_].set(EMPTY)
    # capture the opponent if exists
    captured = pb[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state.hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by (x + 14) % 28, and
        #   (2) filtering promoted piece by x % 8
        lambda: state.hand.at[0, ((captured + 14) % 28) % 8].add(1),
    )
    # promote piece
    piece = jax.lax.select(action.is_promotion, action.piece + 8, action.piece)
    # set piece to the target position
    pb = pb.at[action.to].set(piece)
    # apply piece moves
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    pb = state.piece_board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _legal_action_mask(state: State):
    @jax.vmap
    def is_legal(action):
        a = Action._from_dlshogi_action(state, action)
        return jax.lax.cond(
            a.from_ < 0,  # TODO: fix me. a is invalid. all LEGAL_FROM_IDX == -1,
            lambda: FALSE,
            lambda: jax.lax.cond(
                a.is_drop,
                lambda: _is_legal_drop(state.piece_board, state.hand, a.piece, a.to),
                lambda: _is_legal_move(state.piece_board, a.from_ * 81 + a.to, a.is_promotion)
            )
        )

    return is_legal(jnp.arange(27 * 81))


def _is_legal_drop(board: jnp.ndarray, hand: jnp.ndarray, piece: jnp.ndarray, to: jnp.ndarray):
    # destination is not empty
    is_illegal = board[to] != EMPTY
    # don't have the piece
    is_illegal |= hand[0, piece] <= 0

    # actually drop
    board = board.at[to].set(piece)

    # suicide move
    king_pos = jnp.nonzero(board == KING, size=1)[0][0]
    _apply = jax.vmap(partial(can_major_capture_king, board=board, king_pos=king_pos))
    is_illegal |= _apply(f=jnp.arange(81)).any()  # TODO: 実際には81ではなくqueen movesだけで十分
    # captured by neighbours (王の周囲から)
    _apply = jax.vmap(partial(can_neighbour_capture_king, board=board, king_pos=king_pos))
    is_illegal |= _apply(f=NEIGHBOURS[king_pos]).any()

    return ~is_illegal


def _is_legal_move(board: jnp.ndarray, move: jnp.ndarray, is_promotion: jnp.ndarray):
    from_, to = move // 81, move % 81
    # source is not my piece
    piece = board[from_]
    is_illegal = ~((PAWN <= piece) & (piece < OPP_PAWN))
    # destination is my piece
    is_illegal |= (PAWN <= board[to]) & (board[to] < OPP_PAWN)
    # piece cannot move like that
    is_illegal |= ~CAN_MOVE[piece, from_, to]
    # there is an obstacle between from_ and to
    i = _major_piece_ix(piece)
    is_illegal |= ((i >= 0) & (BETWEEN[i, from_, to, :] & (board != EMPTY)).any())

    # actually move
    board = board.at[from_].set(EMPTY).at[to].set(piece)

    # suicide move （王手放置、自殺手）
    king_pos = jnp.nonzero(board == KING, size=1)[0][0]
    # captured by large piece (大駒)
    _apply = jax.vmap(partial(can_major_capture_king, board=board, king_pos=king_pos))
    is_illegal |= _apply(f=jnp.arange(81)).any()  # TODO: 実際には81ではなくqueen movesだけで十分
    # captured by neighbours (王の周囲から)
    _apply = jax.vmap(partial(can_neighbour_capture_king, board=board, king_pos=king_pos))
    is_illegal |= _apply(f=NEIGHBOURS[king_pos]).any()

    # TODO:
    is_illegal |= is_promotion

    return ~is_illegal

def can_major_capture_king(board, king_pos, f):
    p = _flip_piece(board[f])  # 敵の大駒
    i = _major_piece_ix(p)  # 敵の大駒のix
    return ((i >= 0) &  # 敵の大駒かつ
            (CAN_MOVE[p, king_pos, f]) &  # 移動可能で
            ((BETWEEN[i, king_pos, f, :] & (board != EMPTY)).sum() == 0))  # 障害物なし

def can_neighbour_capture_king(board, king_pos, f):
    p = _flip_piece(board[f])
    return (f >= 0) & (PAWN <= p) & (p < OPP_PAWN) & CAN_MOVE[p, king_pos, f]

def _flip_piece(piece):
    return jax.lax.select(piece >= 0, (piece + 14) % 28, piece)

def _major_piece_ix(piece):
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
