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
import numpy as np
from jax import Array, lax

EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))  # opponent: -1 * piece
MAX_TERMINATION_STEPS = 512  # from AlphaZero paper

# prepare precomputed values here (e.g., available moves, map to label, etc.)

# index: a1: 0, a2: 1, ..., h8: 63
INIT_BOARD = jnp.int32([4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4])  # fmt: skip
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h

# Action: AlphaZero style label (4672 = 64 x 73)
# * [0:9]  underpromotions
#     plane // 3 == 0: rook, 1: bishop, 2: knight
#     plane  % 3 == 0: up  , 1: right,  2: left
# * [9:73] normal moves (queen:56 + knight:8)
#   51                   22                   50
#      52                21                49
#         53             20             48
#            54          19          47
#               55       18       46
#                  56    17    45
#                     57 16 44
#   23 24 25 26 27 28 29  X 30 31 32 33 34 35 36
#                     43 15 58
#                  42    14    59
#               41       13       60
#            40          12          61
#         39             11             62
#      38                10                64
#   37                    9                   64
FROM_PLANE = -np.ones((64, 73), dtype=np.int32)
TO_PLANE = -np.ones((64, 64), dtype=np.int32)  # ignores underpromotion
zeros, seq, rseq = [0] * 7, list(range(1, 8)), list(range(-7, 0))
# down, up, left, right, down-left, down-right, up-right, up-left, knight, and knight
dr = rseq[::] + seq[::] + zeros[::] + zeros[::] + rseq[::] + seq[::] + seq[::-1] + rseq[::-1]
dc = zeros[::] + zeros[::] + rseq[::] + seq[::] + rseq[::] + seq[::] + rseq[::] + seq[::]
dr += [-1, +1, -2, +2, -1, +1, -2, +2]
dc += [-2, -2, -1, -1, +2, +2, +1, +1]
for from_ in range(64):
    for plane in range(73):
        if plane < 9:  # underpromotion
            to = from_ + [+1, +9, -7][plane % 3] if from_ % 8 == 6 else -1
            if 0 <= to < 64:
                FROM_PLANE[from_, plane] = to
        else:  # normal moves
            r = from_ % 8 + dr[plane - 9]
            c = from_ // 8 + dc[plane - 9]
            if 0 <= r < 8 and 0 <= c < 8:
                to = c * 8 + r
                FROM_PLANE[from_, plane] = to
                TO_PLANE[from_, to] = plane

INIT_LEGAL_ACTION_MASK = np.zeros(64 * 73, dtype=np.bool_)
ixs = [89, 90, 652, 656, 673, 674, 1257, 1258, 1841, 1842, 2425, 2426, 3009, 3010, 3572, 3576, 3593, 3594, 4177, 4178]
INIT_LEGAL_ACTION_MASK[ixs] = True

LEGAL_DEST = -np.ones((7, 64, 27), np.int32)  # LEGAL_DEST[0, :, :] == -1
LEGAL_DEST_NEAR = -np.ones((64, 16), np.int32)
LEGAL_DEST_FAR = -np.ones((64, 19), np.int32)
CAN_MOVE = np.zeros((7, 64, 64), dtype=np.bool_)
for from_ in range(64):
    legal_dest = {p: [] for p in range(7)}
    for to in range(64):
        if from_ == to:
            continue
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if (r1 - r0 == 1 and abs(c1 - c0) <= 1) or ((r0, r1) == (1, 3) and abs(c1 - c0) == 0):
            legal_dest[PAWN].append(to)
        if (abs(r1 - r0) == 1 and abs(c1 - c0) == 2) or (abs(r1 - r0) == 2 and abs(c1 - c0) == 1):
            legal_dest[KNIGHT].append(to)
        if abs(r1 - r0) == abs(c1 - c0):
            legal_dest[BISHOP].append(to)
        if abs(r1 - r0) == 0 or abs(c1 - c0) == 0:
            legal_dest[ROOK].append(to)
        if (abs(r1 - r0) == 0 or abs(c1 - c0) == 0) or (abs(r1 - r0) == abs(c1 - c0)):
            legal_dest[QUEEN].append(to)
        if from_ != to and abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1:
            legal_dest[KING].append(to)
    for p in range(1, 7):
        LEGAL_DEST[p, from_, : len(legal_dest[p])] = legal_dest[p]
        CAN_MOVE[p, from_, legal_dest[p]] = True
    dests = list(set(legal_dest[KING]) | set(legal_dest[KNIGHT]))
    LEGAL_DEST_NEAR[from_, : len(dests)] = dests
    dests = list(set(legal_dest[QUEEN]).difference(set(legal_dest[KING])))
    LEGAL_DEST_FAR[from_, : len(dests)] = dests

BETWEEN = -np.ones((64, 64, 6), dtype=np.int32)
for from_ in range(64):
    for to in range(64):
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if not (abs(r1 - r0) == 0 or abs(c1 - c0) == 0 or abs(r1 - r0) == abs(c1 - c0)):
            continue
        dr, dc = max(min(r1 - r0, 1), -1), max(min(c1 - c0, 1), -1)
        for i in range(6):
            r, c = r0 + dr * (i + 1), c0 + dc * (i + 1)
            if r == r1 and c == c1:
                break
            BETWEEN[from_, to, i] = c * 8 + r

FROM_PLANE, TO_PLANE, INIT_LEGAL_ACTION_MASK, LEGAL_DEST, LEGAL_DEST_NEAR, LEGAL_DEST_FAR, CAN_MOVE, BETWEEN = (
    jnp.array(x) for x in (FROM_PLANE, TO_PLANE, INIT_LEGAL_ACTION_MASK, LEGAL_DEST, LEGAL_DEST_NEAR, LEGAL_DEST_FAR, CAN_MOVE, BETWEEN)
)

keys = jax.random.split(jax.random.PRNGKey(12345), 4)
ZOBRIST_BOARD = jax.random.randint(keys[0], shape=(64, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_SIDE = jax.random.randint(keys[1], shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_CASTLING = jax.random.randint(keys[2], shape=(4, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_EN_PASSANT = jax.random.randint(keys[3], shape=(65, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
INIT_ZOBRIST_HASH = jnp.uint32([1455170221, 1478960862])


class GameState(NamedTuple):
    color: Array = jnp.int32(0)  # w: 0, b: 1
    board: Array = INIT_BOARD  # (64,)
    castling_rights: Array = jnp.ones([2, 2], dtype=jnp.bool_)  # my queen, my king, opp queen, opp king
    en_passant: Array = jnp.int32(-1)
    halfmove_count: Array = jnp.int32(0)  # number of moves since the last piece capture or pawn move
    fullmove_count: Array = jnp.int32(1)  # increase every black move
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(INIT_ZOBRIST_HASH)
    board_history: Array = jnp.zeros((8, 64), dtype=jnp.int32).at[0, :].set(INIT_BOARD)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    step_count: Array = jnp.int32(0)


class Action(NamedTuple):
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: Array):
        from_, plane = label // 73, label % 73
        underpromotion = lax.select(plane >= 9, -1, plane // 3)
        return Action(from_=from_, to=FROM_PLANE[from_, plane], underpromotion=underpromotion)

    def _to_label(self):
        return self.from_ * 73 + TO_PLANE[self.from_, self.to]


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        state = _apply_move(state, Action._from_label(action))
        state = _flip(state)
        state = _update_history(state)
        state = state._replace(legal_action_mask=_legal_action_mask(state))
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

        def make(i):
            board = jnp.rot90(state.board_history[i].reshape((8, 8)), k=1)

            def piece_feat(p):
                return (board == p).astype(jnp.float32)

            my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
            opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

            h = state.hash_history[i, :]
            rep = (state.hash_history == h).all(axis=1).sum() - 1
            rep = lax.select((h == 0).all(), 0, rep)
            rep0 = ones * (rep == 0)
            rep1 = ones * (rep >= 1)
            return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

        return jnp.vstack(
            [
                jax.vmap(make)(jnp.arange(8)).reshape(-1, 8, 8),  # board feature
                color * ones,  # color
                (state.step_count / MAX_TERMINATION_STEPS) * ones,  # total move count
                state.castling_rights.flatten()[:, None, None] * ones,  # (my queen, my king, opp queen, opp king)
                (state.halfmove_count.astype(jnp.float32) / 100.0) * ones,  # no progress count
            ]
        ).transpose((1, 2, 0))

    def legal_action_mask(self, state: GameState) -> Array:
        return state.legal_action_mask

    def is_terminal(self, state: GameState) -> Array:
        terminated = ~state.legal_action_mask.any()
        terminated |= state.halfmove_count >= 100
        terminated |= has_insufficient_pieces(state)
        rep = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        terminated |= rep >= 2
        terminated |= MAX_TERMINATION_STEPS <= state.step_count
        return terminated

    def rewards(self, state: GameState) -> Array:
        is_checkmate = (~state.legal_action_mask.any()) & _is_checked(state)
        return lax.select(
            is_checkmate,
            jnp.ones(2, dtype=jnp.float32).at[state.color].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )


def _update_history(state: GameState):
    board_history = jnp.roll(state.board_history, 64)
    board_history = board_history.at[0].set(state.board)
    hash_hist = jnp.roll(state.hash_history, 2)
    hash_hist = hash_hist.at[0].set(_zobrist_hash(state))
    return state._replace(board_history=board_history, hash_history=hash_hist)


def has_insufficient_pieces(state: GameState):
    # uses the same condition as OpenSpiel
    num_pieces = (state.board != EMPTY).sum()
    num_pawn_rook_queen = ((jnp.abs(state.board) >= ROOK) | (jnp.abs(state.board) == PAWN)).sum() - 2  # two kings
    num_bishop = (jnp.abs(state.board) == BISHOP).sum()
    coords = jnp.arange(64).reshape((8, 8))
    black_coords = jnp.hstack((coords[::2, ::2].ravel(), coords[1::2, 1::2].ravel()))
    num_bishop_on_black = (jnp.abs(state.board[black_coords]) == BISHOP).sum()
    is_insufficient = False
    # king vs king
    is_insufficient |= num_pieces <= 2
    # king vs king + (knight or bishop)
    is_insufficient |= (num_pieces == 3) & (num_pawn_rook_queen == 0)
    # king + bishop* vs king + bishop* (bishops are on same color tile)
    is_bishop_all_on_black = num_bishop_on_black == num_bishop
    is_bishop_all_on_white = num_bishop_on_black == 0
    is_insufficient |= (num_pieces == num_bishop + 2) & (is_bishop_all_on_black | is_bishop_all_on_white)

    return is_insufficient


def _apply_move(state: GameState, a: Action) -> GameState:
    piece = state.board[a.from_]
    # en passant
    is_en_passant = (state.en_passant >= 0) & (piece == PAWN) & (state.en_passant == a.to)
    removed_pawn_pos = a.to - 1
    state = state._replace(
        board=state.board.at[removed_pawn_pos].set(lax.select(is_en_passant, EMPTY, state.board[removed_pawn_pos]))
    )
    is_en_passant = (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2)
    state = state._replace(en_passant=lax.select(is_en_passant, (a.to + a.from_) // 2, -1))
    # update counters
    captured = (state.board[a.to] < 0) | is_en_passant
    state = state._replace(
        halfmove_count=lax.select(captured | (piece == PAWN), 0, state.halfmove_count + 1),
        fullmove_count=state.fullmove_count + jnp.int32(state.color == 1),
    )
    # castling
    board = state.board
    is_queen_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 16)
    board = lax.select(is_queen_side_castling, board.at[0].set(EMPTY).at[24].set(ROOK), board)
    is_king_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 48)
    board = lax.select(is_king_side_castling, board.at[56].set(EMPTY).at[40].set(ROOK), board)
    state = state._replace(board=board)
    # update castling rights
    cond = jnp.bool_([[(a.from_ != 32) & (a.from_ != 0), (a.from_ != 32) & (a.from_ != 56)], [a.to != 7, a.to != 63]])
    state = state._replace(castling_rights=state.castling_rights & cond)
    # promotion to queen
    piece = lax.select((piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0), QUEEN, piece)
    # underpromotion
    piece = lax.select(a.underpromotion < 0, piece, jnp.int32([ROOK, BISHOP, KNIGHT])[a.underpromotion])
    # actually move
    state = state._replace(board=state.board.at[a.from_].set(EMPTY).at[a.to].set(piece))  # type: ignore
    return state


def _flip_pos(x: Array):  # e.g., 37 <-> 34, -1 <-> -1
    return lax.select(x == -1, x, (x // 8) * 8 + (7 - (x % 8)))


def _flip(state: GameState) -> GameState:
    return state._replace(
        board=-jnp.flip(state.board.reshape(8, 8), axis=1).flatten(),
        color=(state.color + 1) % 2,
        en_passant=_flip_pos(state.en_passant),
        castling_rights=state.castling_rights[::-1],
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
            return lax.select(ok, Action(from_=from_, to=to)._to_label(), -1)

        return jax.vmap(legal_label)(LEGAL_DEST[piece, from_])

    def legal_en_passants():
        to = state.en_passant

        def legal_labels(from_):
            ok = (from_ >= 0) & (from_ < 64) & (to >= 0) & (state.board[from_] == PAWN) & (state.board[to - 1] == -PAWN)
            a = Action(from_=from_, to=to)
            return lax.select(ok, a._to_label(), -1)

        return jax.vmap(legal_labels)(jnp.int32([to - 9, to + 7]))

    def is_not_checked(label):
        a = Action._from_label(label)
        return ~_is_checked(_apply_move(state, a))

    def legal_underpromotions(mask):
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state.board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return lax.select(ok, label, -1)

        labels = jnp.int32([from_ * 73 + i for i in range(9) for from_ in [6, 14, 22, 30, 38, 46, 54, 62]])
        return jax.vmap(legal_labels)(labels)

    # normal move and en passant
    possible_piece_positions = jnp.nonzero(state.board > 0, size=16, fill_value=-1)[0]
    a1 = jax.vmap(legal_normal_moves)(possible_piece_positions).flatten()
    a2 = legal_en_passants()
    actions = jnp.hstack((a1, a2))  # include -1
    actions = jnp.where(jax.vmap(is_not_checked)(actions), actions, -1)
    mask = jnp.zeros(64 * 73 + 1, dtype=jnp.bool_)  # +1 for sentinel
    mask = mask.at[actions].set(True)

    # castling
    b = state.board
    can_castle_queen_side = state.castling_rights[0, 0]
    can_castle_queen_side &= (b[0] == ROOK) & (b[8] == EMPTY) & (b[16] == EMPTY) & (b[24] == EMPTY) & (b[32] == KING)
    can_castle_king_side = state.castling_rights[0, 1]
    can_castle_king_side &= (b[32] == KING) & (b[40] == EMPTY) & (b[48] == EMPTY) & (b[56] == ROOK)
    not_checked = ~jax.vmap(_is_attacked, in_axes=(None, 0))(state, jnp.int32([16, 24, 32, 40, 48]))
    mask = mask.at[2364].set(mask[2364] | (can_castle_queen_side & not_checked[:3].all()))
    mask = mask.at[2367].set(mask[2367] | (can_castle_king_side & not_checked[2:].all()))

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(True)

    return mask[:-1]


def _is_attacked(state: GameState, pos: Array):
    def attacked_far(to):
        ok = (to >= 0) & (state.board[to] < 0)  # should be opponent's
        piece = jnp.abs(state.board[to])
        ok &= (piece == QUEEN) | (piece == ROOK) | (piece == BISHOP)
        between_ixs = BETWEEN[pos, to]
        ok &= CAN_MOVE[piece, pos, to] & ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
        return ok

    def attacked_near(to):
        ok = (to >= 0) & (state.board[to] < 0)  # should be opponent's
        piece = jnp.abs(state.board[to])
        ok &= CAN_MOVE[piece, pos, to]
        ok &= ~((piece == PAWN) & (to // 8 == pos // 8))  # should move diagonally to capture
        return ok

    by_minor = jax.vmap(attacked_near)(LEGAL_DEST_NEAR[pos, :]).any()
    by_major = jax.vmap(attacked_far)(LEGAL_DEST_FAR[pos, :]).any()
    return by_minor | by_major


def _is_checked(state: GameState):
    king_pos = jnp.argmin(jnp.abs(state.board - KING))
    return _is_attacked(state, king_pos)


def _zobrist_hash(state: GameState) -> Array:
    hash_ = lax.select(state.color == 0, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    to_reduce = ZOBRIST_BOARD[jnp.arange(64), state.board + 6]  # 0, ..., 12 (w:pawn, ..., b:king)
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))
    to_reduce = jnp.where(state.castling_rights.reshape(-1, 1), ZOBRIST_CASTLING, 0)
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))
    hash_ ^= ZOBRIST_EN_PASSANT[state.en_passant]
    return hash_
