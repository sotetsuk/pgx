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

"""Shogi environment (beta version)

Speeding up alpha version (_shogi.py)."""


from functools import partial

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass
from pgx._shogi_utils import (
    BETWEEN,
    CAN_MOVE,
    INIT_PIECE_BOARD,
    LEGAL_FROM_IDX,
    NEIGHBOURS,
    _from_sfen,
    _to_sfen,
)

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int8(-1)  # 空白
PAWN = jnp.int8(0)  # 歩
LANCE = jnp.int8(1)  # 香
KNIGHT = jnp.int8(2)  # 桂
SILVER = jnp.int8(3)  # 銀
BISHOP = jnp.int8(4)  # 角
ROOK = jnp.int8(5)  # 飛
GOLD = jnp.int8(6)  # 金
KING = jnp.int8(7)  # 玉
PRO_PAWN = jnp.int8(8)  # と
PRO_LANCE = jnp.int8(9)  # 成香
PRO_KNIGHT = jnp.int8(10)  # 成桂
PRO_SILVER = jnp.int8(11)  # 成銀
HORSE = jnp.int8(12)  # 馬
DRAGON = jnp.int8(13)  # 龍
OPP_PAWN = jnp.int8(14)  # 相手歩
OPP_LANCE = jnp.int8(15)  # 相手香
OPP_KNIGHT = jnp.int8(16)  # 相手桂
OPP_SILVER = jnp.int8(17)  # 相手銀
OPP_BISHOP = jnp.int8(18)  # 相手角
OPP_ROOK = jnp.int8(19)  # 相手飛
OPP_GOLD = jnp.int8(20)  # 相手金
OPP_KING = jnp.int8(21)  # 相手玉
OPP_PRO_PAWN = jnp.int8(22)  # 相手と
OPP_PRO_LANCE = jnp.int8(23)  # 相手成香
OPP_PRO_KNIGHT = jnp.int8(24)  # 相手成桂
OPP_PRO_SILVER = jnp.int8(25)  # 相手成銀
OPP_HORSE = jnp.int8(26)  # 相手馬
OPP_DRAGON = jnp.int8(27)  # 相手龍


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(27 * 81, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((119, 9, 9), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,) 後手のときにはflipする
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)  # 後手のときにはflipする
    # cache
    major_piece_board: jnp.ndarray = jnp.ones((2, 81), dtype=jnp.int8)  # 0 = LANCE, 1 = BISHOP,HORSE, 2 = ROOK,DRAGON
    legal_moves: jnp.ndarray = jnp.ones((2, 81, 81, 2), dtype=jnp.bool_)
    checked: jnp.ndarray = jnp.zeros(2, dtype=jnp.bool_)

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
        turn, pb, hand, step_count = _from_sfen(sfen)
        return jax.jit(State._from_board)(turn, pb, hand).replace(  # type: ignore
            _step_count=jnp.int32(step_count)
        )

    def _to_sfen(self):
        state = self if self.turn % 2 == 0 else _flip(self)
        return _to_sfen(state)


class Shogi(core.Env):
    def __init__(self, max_termination_steps: int = 1000):
        super().__init__()
        self.max_termination_steps = max_termination_steps

    def _init(self, key: jax.random.KeyArray) -> State:
        state = _init_board()
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        return state.replace(current_player=current_player)  # type: ignore

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        # Note: Assume that illegal action is already filtered by Env.step
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
        """Direction (from github.com/TadaoYamaoka/cshogi)

         0 Up
         1 Up left
         2 Up right
         3 Left
         4 Right
         5 Down
         6 Down left
         7 Down right
         8 Up2 left
         9 Up2 right
        10 Promote +  Up
        11 Promote +  Up left
        12 Promote +  Up right
        13 Promote +  Left
        14 Promote +  Right
        15 Promote +  Down
        16 Promote +  Down left
        17 Promote +  Down right
        18 Promote +  Up2 left
        19 Promote +  Up2 right
        20 Drop 歩
        21 Drop 香車
        22 Drop 桂馬
        23 Drop 銀
        24 Drop 角
        25 Drop 飛車
        26 Drop 金
        """
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
        from_ = jax.lax.select(is_drop, 0, legal_from_idx[i])
        piece = jax.lax.select(
            is_drop, direction - 20, state.piece_board[from_]
        )
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promotion=is_promotion)  # type: ignore


def _init_board():
    """Initialize Shogi State."""
    state = State()
    return state.replace(  # type: ignore
        legal_action_mask=...,
        major_piece_board=...,
        legal_movs=...,
    )


def _step(state: State, action: jnp.ndarray):
    a = Action._from_dlshogi_action(state, action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))

    # update cache
    state = _update_cache(state, action)

    # flip state
    state = _flip(state)
    state = state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        turn=(state.turn + 1) % 2,
    )

    legal_action_mask = _update_legal_action_mask(state)

    terminated = ~legal_action_mask.any()
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

def _update_cache(state: State, action: Action) -> State:
    return state

def _update_legal_action_mask(state: State) -> State:
    """Update legal action mask (only) from cache"""
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

def _rotate(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.rot90(board.reshape(9, 9), k=3)

def _flip_piece(piece):
    return jax.lax.select(piece >= 0, (piece + 14) % 28, piece)

def _flip(state):
    return state.replace(  # type: ignore
        piece_board=jax.vmap(_flip_piece)(state.piece_board)[::-1],
        hand=state.hand[jnp.int8((1, 0))],
    )

def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(state.observation)
