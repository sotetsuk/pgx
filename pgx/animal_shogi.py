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
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


EMPTY = jnp.int32(-1)
PAWN = jnp.int32(0)
BISHOP = jnp.int32(1)
ROOK = jnp.int32(2)
KING = jnp.int32(3)
GOLD = jnp.int32(4)
#  5: OPP_PAWN
#  6: OPP_BISHOP
#  7: OPP_ROOK
#  8: OPP_KING
#  9: OPP_GOLD
# fmt: off
INIT_BOARD = jnp.int32(
    [6, -1, -1, 2,
     8, 5, 0, 3,
     7, -1, -1, 1]
)  # (12,)
# fmt: on

ZOBRIST_KEY = jax.random.PRNGKey(23279)
ZOBRIST_SIDE = jax.random.randint(ZOBRIST_KEY, shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_KEY, ZOBRIST_SUBKEY = jax.random.split(ZOBRIST_KEY)
ZOBRIST_BOARD = jax.random.randint(
    ZOBRIST_SUBKEY,
    shape=(12, 10, 2),
    minval=0,
    maxval=2**31 - 1,
    dtype=jnp.uint32,
)
ZOBRIST_KEY, ZOBRIST_SUBKEY = jax.random.split(ZOBRIST_KEY)
ZOBRIST_HAND = jax.random.randint(
    ZOBRIST_SUBKEY,
    shape=(2, 3, 3, 2),
    minval=0,
    maxval=2**31 - 1,
    dtype=jnp.uint32,
)


MAX_TERMINATION_STEPS = 256


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(132, dtype=jnp.bool_)  # (132,)
    observation: Array = jnp.zeros((4, 3, 22), dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Animal Shogi specific ---
    _turn: Array = jnp.int32(0)
    _board: Array = INIT_BOARD  # (12,)
    _hand: Array = jnp.zeros((2, 3), dtype=jnp.int32)
    _zobrist_hash: Array = jnp.uint32([233882788, 593924309])
    _hash_history: Array = (
        jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(jnp.uint32([233882788, 593924309]))
    )
    _board_history: Array = (-jnp.ones((8, 12), dtype=jnp.int32)).at[0, :].set(INIT_BOARD)
    _hand_history: Array = jnp.zeros((8, 6), dtype=jnp.int32)
    _rep_history: Array = jnp.zeros((8,), dtype=jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
        return "animal_shogi"


@dataclass
class Action:
    is_drop: Array = FALSE
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    drop_piece: Array = jnp.int32(-1)

    @staticmethod
    def _from_label(a: Array):
        # Implements AlphaZero like action label:
        # 132 labels =
        #   [Move] 8 (direction) * 12 (from_) +
        #   [Drop] 3 (piece_type) * 12 (to)
        x, sq = jnp.int32(a // 12), jnp.int32(a % 12)
        is_drop = 8 <= x
        return jax.lax.cond(
            is_drop,
            lambda: Action(is_drop=TRUE, to=sq, drop_piece=x - 8),  # type: ignore
            lambda: Action(is_drop=FALSE, from_=sq, to=_to(sq, x)),  # type: ignore
        )


class AnimalShogi(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        state = State(current_player=current_player)  # type: ignore
        state = state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        state = _step(state, action)
        state = jax.lax.cond(
            MAX_TERMINATION_STEPS <= state._step_count,
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "animal_shogi"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: Array):
    a = Action._from_label(action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))
    is_try = (state._board[jnp.int32([0, 4, 8])] == KING).any()

    # update board/hand history
    board_history = jnp.roll(state._board_history, 8)
    board_history = board_history.at[0].set(state._board)
    state = state.replace(_board_history=board_history)  # type:ignore
    hand_history = jnp.roll(state._hand_history, 8)
    hand_history = hand_history.at[0].set(state._hand.flatten())
    state = state.replace(_hand_history=hand_history)  # type:ignore

    state = _flip(state)
    state = state.replace(  # type: ignore
        _hash_history=state._hash_history.at[state._step_count].set(state._zobrist_hash),
    )

    rep = (state._hash_history == state._zobrist_hash).any(axis=1).sum().astype(jnp.int32) - 1
    is_rep_draw = rep >= 2
    legal_action_mask = _legal_action_mask(state)  # TODO: fix me
    terminated = (~legal_action_mask.any()) | is_try | is_rep_draw
    # fmt: off
    reward = jax.lax.select(
        terminated & ~is_rep_draw,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # rep history
    rep_history = jnp.roll(state._rep_history, 1)
    rep_history = rep_history.at[0].set(rep)
    # fmt: on
    return state.replace(  # type: ignore
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        rewards=reward,
        _rep_history=rep_history,
    )


def _observe(state: State, player_id: Array) -> Array:
    # player_id's color
    color = jax.lax.select(state.current_player == player_id, state._turn, 1 - state._turn)

    state = jax.lax.cond(
        state.current_player == player_id,
        lambda: state,
        lambda: _flip(state),
    )

    def make(i):
        def is_piece(p, state):
            return state._board_history[i] == p

        def num_hand(p, n, state):
            return state._hand_history[i, p] >= n

        # fmt: off
        piece_feat = jax.vmap(partial(is_piece, state=state))(jnp.arange(10))  # (10, 12)
        hand_feat = jax.vmap(jax.vmap(
            partial(num_hand, state=state), (None, 0)), (0, None)
        )(jnp.arange(6), jnp.arange(1, 3)).flatten().reshape(12, 1)  # (12, 1)
        hand_feat = jnp.tile(hand_feat, reps=(1, 12))  # (12, 12)
        # fmt: on

        ones = jnp.ones((1, 12), dtype=jnp.float32)
        rep0 = ones * (state._rep_history[i] == 0)
        rep1 = ones * (state._rep_history[i] == 1)
        return jnp.vstack((piece_feat, hand_feat, rep0, rep1))  # (24, 12)

    obs = jax.vmap(make)(jnp.arange(8)).reshape(-1, 12).astype(jnp.float32)  # (192 = 8 * 24, 12)
    ones = jnp.ones((1, 12), dtype=jnp.float32)
    obs = jnp.vstack([obs, ones * color, ones * state._step_count / MAX_TERMINATION_STEPS])  # (193, 12)
    obs = obs.reshape(-1, 3, 4).transpose((2, 1, 0))  # channel last
    return jnp.flip(obs, axis=1)


def _step_move(state: State, action: Action) -> State:
    piece = state._board[action.from_]
    # remove piece from the original position
    board = state._board.at[action.from_].set(EMPTY)
    zb_from_ = jax.lax.select(state._turn == 0, action.from_, 11 - action.from_)
    zb_piece = jax.lax.select(state._turn == 0, piece, (piece + 5) % 10)
    zobrist_hash = state._zobrist_hash ^ ZOBRIST_BOARD[zb_from_, zb_piece]
    # capture the opponent if exists
    captured = board[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state._hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by % 5, and
        #   (2) filtering promoted piece by x % 4
        lambda: state._hand.at[0, (captured % 5) % 4].add(1),
    )
    zobrist_hash = jax.lax.select(
        captured == EMPTY,
        zobrist_hash,
        zobrist_hash
        ^ ZOBRIST_BOARD[
            zb_from_,
            jax.lax.select(state._turn == 0, captured, (captured + 5) % 10),
        ],
    )
    num_hand = hand[0, (captured % 5) % 4]
    zobrist_hash ^= ZOBRIST_HAND[state._turn, (captured % 5) % 4, num_hand - 1]
    zobrist_hash ^= ZOBRIST_HAND[state._turn, (captured % 5) % 4, num_hand]
    # promote piece (PAWN to GOLD)
    is_promotion = (action.from_ % 4 == 1) & (piece == PAWN)
    piece = jax.lax.select(is_promotion, GOLD, piece)
    # set piece to the target position
    board = board.at[action.to].set(piece)
    zb_to_ = jax.lax.select(state._turn == 0, action.to, 11 - action.to)
    zb_piece = jax.lax.select(state._turn == 0, piece, (piece + 5) % 10)
    zobrist_hash ^= ZOBRIST_BOARD[zb_to_, zb_piece]
    # apply piece moves
    return state.replace(_board=board, _hand=hand, _zobrist_hash=zobrist_hash)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    board = state._board.at[action.to].set(action.drop_piece)
    zb_to_ = jax.lax.select(state._turn == 0, action.to, 11 - action.to)
    zb_piece = jax.lax.select(state._turn == 0, action.drop_piece, (action.drop_piece + 5) % 10)
    zobrist_hash = state._zobrist_hash ^ ZOBRIST_BOARD[zb_to_, zb_piece]
    # remove piece from hand
    hand = state._hand.at[0, action.drop_piece].add(-1)
    num_hand = state._hand[0, action.drop_piece]
    zobrist_hash ^= ZOBRIST_HAND[state._turn, action.drop_piece, num_hand + 1]
    zobrist_hash ^= ZOBRIST_HAND[state._turn, action.drop_piece, num_hand]
    return state.replace(_board=board, _hand=hand, _zobrist_hash=zobrist_hash)  # type: ignore


def _legal_action_mask(state: State):
    def is_legal(label: Array):
        action = Action._from_label(label)
        return jax.lax.cond(action.is_drop, is_legal_drop, is_legal_move, action)

    def is_legal_move(action: Action):
        piece = state._board[action.from_]
        ok = (PAWN <= piece) & (piece <= GOLD)
        ok &= action.to != -1
        ok &= (state._board[action.to] == EMPTY) | (GOLD < state._board[action.to])
        ok &= _can_move(piece, action.from_, action.to)
        ok &= ~_is_checked(_step_move(state, action))
        return ok

    def is_legal_drop(action: Action):
        ok = state._board[action.to] == EMPTY
        ok &= state._hand[0, action.drop_piece] > 0
        ok &= ~_is_checked(_step_drop(state, action))
        return ok

    return jax.vmap(is_legal)(jnp.arange(132))


def _is_checked(state):
    king_pos = jnp.argmin(jnp.abs(state._board - KING))

    @jax.vmap
    def can_capture_king(from_):
        piece = state._board[from_]
        is_opp = piece >= 5
        return is_opp & _can_move(state._board[from_] % 5, king_pos, from_)

    return can_capture_king(jnp.arange(12)).any()  # TODO: King neighbours are enough


def _flip(state):
    empty_mask = state._board == EMPTY
    board = (state._board + 5) % 10
    board = jnp.where(empty_mask, EMPTY, board)
    board = board[::-1]
    empty_mask_hist = state._board_history == EMPTY
    board_history = (state._board_history + 5) % 10
    board_history = jnp.where(empty_mask_hist, EMPTY, board_history)
    board_history = board_history[:, ::-1]
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        _turn=(state._turn + 1) % 2,
        _board=board,
        _hand=state._hand[::-1],
        _zobrist_hash=state._zobrist_hash ^ ZOBRIST_SIDE,
        _board_history=board_history,
        _hand_history=jnp.roll(state._hand_history, 3, axis=1),
    )


def _can_move(piece, from_, to):
    def can_move(piece, from_, to):
        """Can <piece> move from <from_> to <to>?"""
        x0, y0 = from_ // 4, from_ % 4
        x1, y1 = to // 4, to % 4
        dx = x1 - x0
        dy = y1 - y0
        is_neighbour = ((dx != 0) | (dy != 0)) & (jnp.abs(dx) <= 1) & (jnp.abs(dy) <= 1)
        return jax.lax.switch(
            piece,
            [
                lambda: (dx == 0) & (dy == -1),  # PAWN
                lambda: is_neighbour & ((dx == dy) | (dx == -dy)),  # BISHOP
                lambda: is_neighbour & ((dx == 0) | (dy == 0)),  # ROOK
                lambda: is_neighbour,  # KING
                lambda: is_neighbour & ((dx == 0) | (dy != +1)),  # GOLD
            ],
        )

    # fmt: off
    # CAN_MOVE[piece, from_, to] = Can <piece> move from <from_> to <to>?
    CAN_MOVE = jax.vmap(jax.vmap(jax.vmap(
        can_move, (None, None, 0)), (None, 0, None)), (0, None, None)
    )(jnp.arange(5), jnp.arange(12), jnp.arange(12))
    # fmt: on

    return CAN_MOVE[piece, from_, to]


def _to(from_, dir):
    # 5 3 0
    # 6 x 1
    # 7 4 2
    x, y = from_ // 4, from_ % 4
    dx = jnp.int32([-1, -1, -1, 0, 0, 1, 1, 1])
    dy = jnp.int32([-1, 0, 1, -1, 1, -1, 0, 1])
    new_x = x + dx[dir]
    new_y = y + dy[dir]
    return jax.lax.select(
        (new_x < 0) | (new_x >= 3) | (new_y < 0) | (new_y >= 4),
        jnp.int32(-1),
        new_x * 4 + new_y,
    )
