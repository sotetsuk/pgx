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
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
BISHOP = jnp.int8(1)
ROOK = jnp.int8(2)
KING = jnp.int8(3)
GOLD = jnp.int8(4)
#  5: OPP_PAWN
#  6: OPP_ROOK
#  7: OPP_BISHOP
#  8: OPP_KING
#  9: OPP_GOLD
INIT_BOARD = jnp.int8([6, -1, -1, 2, 8, 5, 0, 3, 7, -1, -1, 1])  # (12,)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(132, dtype=jnp.bool_)  # (132,)
    observation: jnp.ndarray = jnp.zeros((4, 3, 22), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Animal Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD  # (12,)
    hand: jnp.ndarray = jnp.zeros((2, 3), dtype=jnp.int8)

    @property
    def env_id(self) -> core.EnvId:
        return "animal_shogi"


@dataclass
class Action:
    is_drop: jnp.ndarray = FALSE
    from_: jnp.ndarray = jnp.int8(-1)
    to: jnp.ndarray = jnp.int8(-1)
    drop_piece: jnp.ndarray = jnp.int8(-1)

    @staticmethod
    def _from_label(a: jnp.ndarray):
        # Implements AlphaZero like action label:
        # 132 labels =
        #   [Move] 8 (direction) * 12 (from_) +
        #   [Drop] 3 (piece_type) * 12 (to)
        x, sq = jnp.int8(a // 12), jnp.int8(a % 12)
        is_drop = 8 <= x
        return jax.lax.cond(
            is_drop,
            lambda: Action(is_drop=TRUE, to=sq, drop_piece=x - 8),  # type: ignore
            lambda: Action(is_drop=FALSE, from_=sq, to=_to(sq, x)),  # type: ignore
        )


class AnimalShogi(core.Env):
    def __init__(self, max_termination_steps: int = 200):
        super().__init__()
        self.max_termination_steps = max_termination_steps

    def _init(self, key: jax.random.KeyArray) -> State:
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        state = State(current_player=current_player)  # type: ignore
        state = state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore
        return state

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        state = _step(state, action)
        state = jax.lax.cond(
            (0 <= self.max_termination_steps)
            & (self.max_termination_steps <= state._step_count),
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "AnimalShogi"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: jnp.ndarray):
    a = Action._from_label(action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))
    is_try = (state.board[jnp.int8([0, 4, 8])] == KING).any()

    state = _flip(state)

    legal_action_mask = _legal_action_mask(state)  # TODO: fix me
    terminated = ~legal_action_mask.any() | is_try
    # fmt: off
    reward = jax.lax.select(
        terminated,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # fmt: on
    return state.replace(  # type: ignore
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        reward=reward,
    )


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    state, flip_state = jax.lax.cond(
        state.current_player == player_id,
        lambda: (state, _flip(state)),
        lambda: (_flip(state), state),
    )

    def is_piece(p, state):
        return state.board == p

    def num_hand(p, n, state):
        return state.hand.flatten()[p] >= n

    # fmt: off
    piece_feat = jax.vmap(partial(is_piece, state=state))(jnp.arange(10))  # (10, 12)
    hand_feat = jax.vmap(jax.vmap(
        partial(num_hand, state=state), (None, 0)), (0, None)
    )(jnp.arange(6), jnp.arange(1, 3)).flatten().reshape(12, 1)  # (12, 1)
    hand_feat = jnp.tile(hand_feat, reps=(1, 12))  # (12, 12)
    # fmt: on

    obs = jnp.vstack((piece_feat, hand_feat))
    obs = obs.reshape(-1, 3, 4).transpose((2, 1, 0))
    return jnp.flip(obs, axis=1)


def _step_move(state: State, action: Action) -> State:
    piece = state.board[action.from_]
    # remove piece from the original position
    board = state.board.at[action.from_].set(EMPTY)
    # capture the opponent if exists
    captured = board[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state.hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by % 5, and
        #   (2) filtering promoted piece by x % 4
        lambda: state.hand.at[0, (captured % 5) % 4].add(1),
    )
    # promote piece (PAWN to GOLD)
    is_promotion = (action.from_ % 4 == 1) & (piece == PAWN)
    piece = jax.lax.select(is_promotion, GOLD, piece)
    # set piece to the target position
    board = board.at[action.to].set(piece)
    # apply piece moves
    return state.replace(board=board, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    board = state.board.at[action.to].set(action.drop_piece)
    # remove piece from hand
    hand = state.hand.at[0, action.drop_piece].add(-1)
    return state.replace(board=board, hand=hand)  # type: ignore


def _legal_action_mask(state: State):
    def is_legal(label: jnp.ndarray):
        action = Action._from_label(label)
        return jax.lax.cond(
            action.is_drop, is_legal_drop, is_legal_move, action
        )

    def is_legal_move(action: Action):
        piece = state.board[action.from_]
        ok = (PAWN <= piece) & (piece <= GOLD)
        ok &= action.to != -1
        ok &= (state.board[action.to] == EMPTY) | (
            GOLD < state.board[action.to]
        )
        ok &= _can_move(piece, action.from_, action.to)
        ok &= ~_is_checked(_step_move(state, action))
        return ok

    def is_legal_drop(action: Action):
        ok = state.board[action.to] == EMPTY
        ok &= state.hand[0, action.drop_piece] > 0
        ok &= (action.drop_piece != PAWN) | (action.to % 4 != 0)
        ok &= ~_is_checked(_step_drop(state, action))
        return ok

    return jax.vmap(is_legal)(jnp.arange(132))


def _is_checked(state):
    king_pos = jnp.argmin(jnp.abs(state.board - KING))

    @jax.vmap
    def can_capture_king(from_):
        piece = state.board[from_]
        is_opp = piece >= 5
        return is_opp & _can_move(state.board[from_] % 5, king_pos, from_)

    return can_capture_king(
        jnp.arange(12)
    ).any()  # TODO: King neighbours are enough


def _flip(state):
    empty_mask = state.board == EMPTY
    board = (state.board + 5) % 10
    board = jnp.where(empty_mask, EMPTY, board)
    board = board[::-1]
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        turn=(state.turn + 1) % 2,
        board=board,
        hand=state.hand[::-1],
    )


def _can_move(piece, from_, to):
    def can_move(piece, from_, to):
        """Can <piece> move from <from_> to <to>?"""
        x0, y0 = from_ // 4, from_ % 4
        x1, y1 = to // 4, to % 4
        dx = x1 - x0
        dy = y1 - y0
        is_neighbour = (
            ((dx != 0) | (dy != 0)) & (jnp.abs(dx) <= 1) & (jnp.abs(dy) <= 1)
        )
        return jax.lax.switch(
            piece,
            [
                lambda: (dx == 0) & (dy == -1),  # PAWN
                lambda: is_neighbour & ((dx == dy) | (dx == -dy)),  # BISHOP
                lambda: is_neighbour & ((dx == 0) | (dy == 0)),  # ROOK
                lambda: is_neighbour,  # KING
                lambda: is_neighbour & ((dx != 0) | (dy != +1)),  # GOLD
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
    dx = jnp.int8([-1, -1, -1, 0, 0, 1, 1, 1])
    dy = jnp.int8([-1, 0, 1, -1, 1, -1, 0, 1])
    new_x = x + dx[dir]
    new_y = y + dy[dir]
    return jax.lax.select(
        (new_x < 0) | (new_x >= 3) | (new_y < 0) | (new_y >= 4),
        jnp.int8(-1),
        new_x * 4 + new_y,
    )
