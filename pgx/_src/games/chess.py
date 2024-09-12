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

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array

from pgx._src.games.chess_utils import (  # type: ignore
    BETWEEN,
    CAN_MOVE,
    INIT_LEGAL_ACTION_MASK,
    LEGAL_DEST,
    LEGAL_DEST_ANY,
    PLANE_MAP,
    TO_MAP,
    ZOBRIST_BOARD,
    ZOBRIST_CASTLING_KING,
    ZOBRIST_CASTLING_QUEEN,
    ZOBRIST_EN_PASSANT,
    ZOBRIST_SIDE,
)

INIT_ZOBRIST_HASH = jnp.uint32([1172276016, 1112364556])
MAX_TERMINATION_STEPS = 512  # from AZ paper

EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))
# OPP_PAWN, OPP_KNIGHT, OPP_BISHOP, OPP_ROOK, OPP_QUEEN, OPP_KING = -1, -2, -3, -4, -5, -6

# board index
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
# fmt: off
INIT_BOARD = jnp.int32([
    4, 1, 0, 0, 0, 0, -1, -4,
    2, 1, 0, 0, 0, 0, -1, -2,
    3, 1, 0, 0, 0, 0, -1, -3,
    5, 1, 0, 0, 0, 0, -1, -5,
    6, 1, 0, 0, 0, 0, -1, -6,
    3, 1, 0, 0, 0, 0, -1, -3,
    2, 1, 0, 0, 0, 0, -1, -2,
    4, 1, 0, 0, 0, 0, -1, -4
])
# fmt: on

# Action
# 0 ... 8: underpromotions
#   plane // 3 == 0: rook, 1: bishop, 2: knight
#   plane  % 3 == 0: up  , 1: right,  2: left
# 51                   22                   50
#    52                21                49
#       53             20             48
#          54          19          47
#             55       18       46
#                56    17    45
#                   57 16 44
# 23 24 25 26 27 28 29  X 30 31 32 33 34 35 36
#                   43 15 58
#                42    14    59
#             41       13       60
#          40          12          61
#       39             11             62
#    38                10                64
# 37                    9                   64


class GameState(NamedTuple):
    turn: Array = jnp.int32(0)
    board: Array = INIT_BOARD  # From top left. like FEN
    # (curr, opp) Flips every turn
    can_castle_queen_side: Array = jnp.ones(2, dtype=jnp.bool_)
    can_castle_king_side: Array = jnp.ones(2, dtype=jnp.bool_)
    en_passant: Array = jnp.int32(-1)  # En passant target. Flips.
    # # of moves since the last piece capture or pawn move
    halfmove_count: Array = jnp.int32(0)
    fullmove_count: Array = jnp.int32(1)  # increase every black move
    zobrist_hash: Array = INIT_ZOBRIST_HASH
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(INIT_ZOBRIST_HASH)
    board_history: Array = jnp.zeros((8, 64), dtype=jnp.int32).at[0, :].set(INIT_BOARD)
    # index to possible piece positions for speeding up. Flips every turn.
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    step_count: Array = jnp.int32(0)


class Action(NamedTuple):
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: Array):
        """We use AlphaZero style label with channel-last representation: (8, 8, 73)

          73 = queen moves (56) + knight moves (8) + underpromotions (3 * 3)

        Note: this representation is reported as

        > We also tried using a flat distribution over moves for chess and shogi;
        > the final result was almost identical although training was slightly slower.

        Flat representation may have 1858 actions (= 1792 normal moves + (7 + 7 + 8) * 3 underpromotions)

        Also see
          - https://github.com/LeelaChessZero/lc0/issues/637
          - https://github.com/LeelaChessZero/lc0/pull/712
        """
        from_, plane = label // 73, label % 73
        underpromotion = jax.lax.select(plane >= 9, -1, plane // 3)
        return Action(from_=from_, to=TO_MAP[from_, plane], underpromotion=underpromotion)

    def _to_label(self):
        return self.from_ * 73 + PLANE_MAP[self.from_, self.to]


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        a = Action._from_label(action)
        state = _update_zobrist_hash(state, a)

        hash_ = state.zobrist_hash
        hash_ ^= _hash_castling_en_passant(state)

        state = _apply_move(state, a)
        state = _flip(state)

        hash_ ^= _hash_castling_en_passant(state)
        state = state._replace(zobrist_hash=hash_)

        state = _update_history(state)
        state = state._replace(legal_action_mask=_legal_action_mask(state))
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.turn
        ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

        def make(i):
            board = _rotate(state.board_history[i].reshape((8, 8)))

            def piece_feat(p):
                return (board == p).astype(jnp.float32)

            my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
            opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

            h = state.hash_history[i, :]
            rep = (state.hash_history == h).all(axis=1).sum() - 1
            rep = jax.lax.select((h == 0).all(), 0, rep)
            rep0 = ones * (rep == 0)
            rep1 = ones * (rep >= 1)
            return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

        board_feat = jax.vmap(make)(jnp.arange(8)).reshape(-1, 8, 8)
        color = color * ones
        total_move_cnt = (state.step_count / MAX_TERMINATION_STEPS) * ones
        my_queen_side_castling_right = ones * state.can_castle_queen_side[0]
        my_king_side_castling_right = ones * state.can_castle_king_side[0]
        opp_queen_side_castling_right = ones * state.can_castle_queen_side[1]
        opp_king_side_castling_right = ones * state.can_castle_king_side[1]
        no_prog_cnt = (state.halfmove_count.astype(jnp.float32) / 100.0) * ones
        return jnp.vstack(
            [
                board_feat,
                color,
                total_move_cnt,
                my_queen_side_castling_right,
                my_king_side_castling_right,
                opp_queen_side_castling_right,
                opp_king_side_castling_right,
                no_prog_cnt,
            ]
        ).transpose((1, 2, 0))

    def legal_action_mask(self, state: GameState) -> Array:
        return state.legal_action_mask

    def is_terminal(self, state: GameState) -> Array:
        terminated = ~state.legal_action_mask.any()
        terminated |= state.halfmove_count >= 100
        terminated |= has_insufficient_pieces(state)
        rep = (state.hash_history == state.zobrist_hash).all(axis=1).sum() - 1
        terminated |= rep >= 2
        terminated |= MAX_TERMINATION_STEPS <= state.step_count
        return terminated

    def rewards(self, state: GameState) -> Array:
        is_checkmate = (~state.legal_action_mask.any()) & _is_checked(state)
        return jax.lax.select(
            is_checkmate,
            jnp.ones(2, dtype=jnp.float32).at[state.turn].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )


def _update_history(state: GameState):
    # board history
    board_history = jnp.roll(state.board_history, 64)
    board_history = board_history.at[0].set(state.board)
    state = state._replace(board_history=board_history)
    # hash hist
    hash_hist = jnp.roll(state.hash_history, 2)
    hash_hist = hash_hist.at[0].set(state.zobrist_hash)
    state = state._replace(hash_history=hash_hist)
    return state


def has_insufficient_pieces(state: GameState):
    # Uses the same condition as OpenSpiel.
    # See https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/chess/chess_board.cc#L724
    num_pieces = (state.board != EMPTY).sum()
    num_pawn_rook_queen = ((jnp.abs(state.board) >= ROOK) | (jnp.abs(state.board) == PAWN)).sum() - 2  # two kings
    num_bishop = (jnp.abs(state.board) == 3).sum()
    coords = jnp.arange(64).reshape((8, 8))
    # [ 0  2  4  6 16 18 20 22 32 34 36 38 48 50 52 54 9 11 13 15 25 27 29 31 41 43 45 47 57 59 61 63]
    black_coords = jnp.hstack((coords[::2, ::2].ravel(), coords[1::2, 1::2].ravel()))
    num_bishop_on_black = (jnp.abs(state.board[black_coords]) == BISHOP).sum()
    is_insufficient = False
    # King vs King
    is_insufficient |= num_pieces <= 2
    # King + X vs King. X == KNIGHT or BISHOP
    is_insufficient |= (num_pieces == 3) & (num_pawn_rook_queen == 0)
    # King + Bishop* vs King + Bishop* (Bishops are on same color tile)
    is_bishop_all_on_black = num_bishop_on_black == num_bishop
    is_bishop_all_on_white = num_bishop_on_black == 0
    is_insufficient |= (num_pieces == num_bishop + 2) & (is_bishop_all_on_black | is_bishop_all_on_white)

    return is_insufficient


def _apply_move(state: GameState, a: Action) -> GameState:
    # apply move action
    piece = state.board[a.from_]
    # en passant
    is_en_passant = (state.en_passant >= 0) & (piece == PAWN) & (state.en_passant == a.to)
    removed_pawn_pos = a.to - 1
    state = state._replace(
        board=state.board.at[removed_pawn_pos].set(jax.lax.select(is_en_passant, EMPTY, state.board[removed_pawn_pos]))
    )
    is_en_passant = (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2)
    state = state._replace(en_passant=jax.lax.select(is_en_passant, (a.to + a.from_) // 2, -1))
    # update counters
    captured = (state.board[a.to] < 0) | is_en_passant
    state = state._replace(
        halfmove_count=jax.lax.select(captured | (piece == PAWN), 0, state.halfmove_count + 1),
        fullmove_count=state.fullmove_count + jnp.int32(state.turn == 1),
    )
    # castling
    board = state.board
    is_queen_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 16)
    board = jax.lax.select(is_queen_side_castling, board.at[0].set(EMPTY).at[24].set(ROOK), board)
    is_king_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 48)
    board = jax.lax.select(is_king_side_castling, board.at[56].set(EMPTY).at[40].set(ROOK), board)
    state = state._replace(board=board)
    # update castling rights
    state = state._replace(
        can_castle_queen_side=state.can_castle_queen_side.at[0].set(
            jax.lax.select((a.from_ == 32) | (a.from_ == 0), False, state.can_castle_queen_side[0])
        ),
        can_castle_king_side=state.can_castle_king_side.at[0].set(
            jax.lax.select((a.from_ == 32) | (a.from_ == 56), False, state.can_castle_king_side[0])
        ),
    )
    state = state._replace(
        can_castle_queen_side=state.can_castle_queen_side.at[1].set(
            jax.lax.select((a.to == 7), False, state.can_castle_queen_side[1])
        ),
        can_castle_king_side=state.can_castle_king_side.at[1].set(
            jax.lax.select((a.to == 63), False, state.can_castle_king_side[1])
        ),
    )
    # promotion to queen
    piece = jax.lax.select((piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0), QUEEN, piece)
    # underpromotion
    piece = jax.lax.select(a.underpromotion < 0, piece, jnp.int32([ROOK, BISHOP, KNIGHT])[a.underpromotion])
    # actually move
    state = state._replace(board=state.board.at[a.from_].set(EMPTY).at[a.to].set(piece))  # type: ignore
    return state


def _flip_pos(x):
    # e.g., 37 <-> 34, -1 <-> -1
    return jax.lax.select(x == -1, x, (x // 8) * 8 + (7 - (x % 8)))


def _rotate(board):
    return jnp.rot90(board, k=1)


def _flip(state: GameState) -> GameState:
    return state._replace(
        board=-jnp.flip(state.board.reshape(8, 8), axis=1).flatten(),
        turn=(state.turn + 1) % 2,
        en_passant=_flip_pos(state.en_passant),
        can_castle_queen_side=state.can_castle_queen_side[::-1],
        can_castle_king_side=state.can_castle_king_side[::-1],
        board_history=-jnp.flip(state.board_history.reshape(-1, 8, 8), axis=-1).reshape(-1, 64),
    )


def _legal_action_mask(state: GameState) -> Array:
    def legal_normal_moves(from_):
        piece = state.board[from_]

        def legal_label(to):
            ok = (from_ >= 0) & (piece > 0) & (to >= 0) & (state.board[to] <= 0)
            between_ixs = BETWEEN[from_, to]
            ok &= CAN_MOVE[piece, from_, to] & ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
            c0, c1 = from_ // 8, to // 8
            pawn_should = ((c1 == c0) & (state.board[to] == EMPTY)) | ((c1 != c0) & (state.board[to] < 0))
            ok &= (piece != PAWN) | pawn_should
            return jax.lax.select(ok, Action(from_=from_, to=to)._to_label(), -1)

        return jax.vmap(legal_label)(LEGAL_DEST[piece, from_])

    def legal_underpromotions(mask):
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state.board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return jax.lax.select(ok, label, -1)

        # from_ = 6 14 ... 62, plane = 0 1 ... 8
        labels = jnp.int32([from_ * 73 + i for i in range(9) for from_ in [6, 14, 22, 30, 38, 46, 54, 62]])
        return jax.vmap(legal_labels)(labels)

    def legal_en_passants():
        to = state.en_passant

        def legal_labels(from_):
            ok = (from_ >= 0) & (from_ < 64) & (to >= 0) & (state.board[from_] == PAWN) & (state.board[to - 1] == -PAWN)
            a = Action(from_=from_, to=to)
            return jax.lax.select(ok, a._to_label(), -1)

        return jax.vmap(legal_labels)(jnp.int32([to - 9, to + 7]))

    @jax.vmap
    def is_not_checked(label):
        a = Action._from_label(label)
        return ~_is_checked(_apply_move(state, a))

    # normal move and en passant
    possible_piece_positions = jnp.nonzero(state.board > 0, size=16, fill_value=-1)[0]
    a1 = jax.vmap(legal_normal_moves)(possible_piece_positions).flatten()
    a2 = legal_en_passants()
    actions = jnp.hstack((a1, a2))  # include -1
    actions = jnp.where(is_not_checked(actions), actions, -1)

    # +1 is to avoid setting True to the last element
    mask = jnp.zeros(64 * 73 + 1, dtype=jnp.bool_)
    mask = mask.at[actions].set(True)

    # castling
    b = state.board
    can_castle_queen_side = state.can_castle_queen_side[0]
    can_castle_queen_side &= (b[0] == ROOK) & (b[8] == EMPTY) & (b[16] == EMPTY) & (b[24] == EMPTY) & (b[32] == KING)
    can_castle_king_side = state.can_castle_king_side[0]
    can_castle_king_side &= (b[32] == KING) & (b[40] == EMPTY) & (b[48] == EMPTY) & (b[56] == ROOK)
    not_checked = ~jax.vmap(_is_attacked, in_axes=(None, 0))(state, jnp.int32([16, 24, 32, 40, 48]))
    mask = mask.at[2364].set(mask[2364] | (can_castle_queen_side & not_checked[:3].all()))
    mask = mask.at[2367].set(mask[2367] | (can_castle_king_side & not_checked[2:].all()))

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(True)

    return mask[:-1]


def _is_attacked(state: GameState, pos):
    def can_move(to):
        ok = (to >= 0) & (state.board[to] < 0)  # should be opponent's
        piece = jnp.abs(state.board[to])
        ok &= CAN_MOVE[piece, pos, to]
        between_ixs = BETWEEN[pos, to]
        ok &= ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
        ok &= ~((piece == PAWN) & (to // 8 == pos // 8))  # should move diagnally to capture the king
        return ok

    return jax.vmap(can_move)(LEGAL_DEST_ANY[pos, :]).any()


def _is_checked(state: GameState):
    """True if possible to capture the opponent king"""
    king_pos = jnp.argmin(jnp.abs(state.board - KING))
    return _is_attacked(state, king_pos)


def _zobrist_hash(state: GameState) -> Array:
    """
    >>> state = GameState()
    >>> _zobrist_hash(state)
    Array([1172276016, 1112364556], dtype=uint32)
    """
    hash_ = jnp.zeros(2, dtype=jnp.uint32)
    hash_ = jax.lax.select(state.turn == 0, hash_, hash_ ^ ZOBRIST_SIDE)
    board = jax.lax.select(state.turn == 0, state.board, _flip(state).board)

    def xor(i, h):
        # 0, ..., 12 (white pawn, ..., black king)
        piece = board[i] + 6
        return h ^ ZOBRIST_BOARD[i, piece]

    hash_ = jax.lax.fori_loop(0, 64, xor, hash_)
    hash_ ^= _hash_castling_en_passant(state)
    return hash_


def _hash_castling_en_passant(state: GameState):
    # we don't take care side (turn) as it's already taken into account in hash
    zero = jnp.uint32([0, 0])
    hash_ = zero
    hash_ ^= jax.lax.select(state.can_castle_queen_side[0], ZOBRIST_CASTLING_QUEEN[0], zero)
    hash_ ^= jax.lax.select(state.can_castle_queen_side[1], ZOBRIST_CASTLING_QUEEN[1], zero)
    hash_ ^= jax.lax.select(state.can_castle_king_side[0], ZOBRIST_CASTLING_KING[0], zero)
    hash_ ^= jax.lax.select(state.can_castle_king_side[1], ZOBRIST_CASTLING_KING[1], zero)
    hash_ ^= ZOBRIST_EN_PASSANT[state.en_passant]
    return hash_


def _update_zobrist_hash(state: GameState, action: Action) -> GameState:
    # do NOT take into account
    #  - en passant, and
    #  - castling
    hash_ = state.zobrist_hash
    source_piece = state.board[action.from_]
    source_piece = jax.lax.select(state.turn == 0, source_piece + 6, (source_piece * -1) + 6)
    destination_piece = state.board[action.to]
    destination_piece = jax.lax.select(state.turn == 0, destination_piece + 6, (destination_piece * -1) + 6)
    from_ = jax.lax.select(state.turn == 0, action.from_, _flip_pos(action.from_))
    to = jax.lax.select(state.turn == 0, action.to, _flip_pos(action.to))
    hash_ ^= ZOBRIST_BOARD[from_, source_piece]  # Remove the piece from source
    hash_ ^= ZOBRIST_BOARD[from_, 6]  # Make source empty
    hash_ ^= ZOBRIST_BOARD[to, destination_piece]  # Remove the piece at target pos (including empty)

    # promotion to queen
    piece = state.board[action.from_]
    source_piece = jax.lax.select(
        (piece == PAWN) & (action.from_ % 8 == 6) & (action.underpromotion < 0),
        jax.lax.select(state.turn == 0, QUEEN + 6, (QUEEN * -1) + 6),
        source_piece,
    )

    # underpromotion
    source_piece = jax.lax.select(
        action.underpromotion >= 0,
        jax.lax.select(
            state.turn == 0,
            source_piece + 3 - action.underpromotion,
            source_piece - (3 - action.underpromotion),
        ),
        source_piece,
    )

    hash_ ^= ZOBRIST_BOARD[to, source_piece]  # Put the piece to the target pos

    # en_passant
    is_en_passant = (state.en_passant >= 0) & (piece == PAWN) & (state.en_passant == action.to)
    removed_pawn_pos = action.to - 1
    removed_pawn_pos = jax.lax.select(state.turn == 0, removed_pawn_pos, _flip_pos(removed_pawn_pos))
    opp_pawn = jax.lax.select(state.turn == 0, (PAWN * -1) + 6, PAWN + 6)
    hash_ ^= jax.lax.select(
        is_en_passant,
        ZOBRIST_BOARD[removed_pawn_pos, opp_pawn],
        jnp.uint32([0, 0]),
    )  # Remove the pawn
    hash_ ^= jax.lax.select(is_en_passant, ZOBRIST_BOARD[removed_pawn_pos, 6], jnp.uint32([0, 0]))  # empty

    hash_ ^= ZOBRIST_SIDE
    return state._replace(  # type: ignore
        zobrist_hash=hash_,
    )
