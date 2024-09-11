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

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.chess_utils import (  # type: ignore
    BETWEEN,
    CAN_MOVE,
    CAN_MOVE_ANY,
    INIT_LEGAL_ACTION_MASK,
    INIT_POSSIBLE_PIECE_POSITIONS,
    PLANE_MAP,
    TO_MAP,
    ZOBRIST_BOARD,
    ZOBRIST_CASTLING_KING,
    ZOBRIST_CASTLING_QUEEN,
    ZOBRIST_EN_PASSANT,
    ZOBRIST_SIDE,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

INIT_ZOBRIST_HASH = jnp.uint32([1172276016, 1112364556])
MAX_TERMINATION_STEPS = 512  # from AZ paper

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int32(0)
PAWN = jnp.int32(1)
KNIGHT = jnp.int32(2)
BISHOP = jnp.int32(3)
ROOK = jnp.int32(4)
QUEEN = jnp.int32(5)
KING = jnp.int32(6)
# OPP_PAWN = -1
# OPP_KNIGHT = -2
# OPP_BISHOP = -3
# OPP_ROOK = -4
# OPP_QUEEN = -5
# OPP_KING = -6


# board index (white view)
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
# board index (flipped black view)
# 8  0  8 16 24 32 40 48 56
# 7  1  9 17 25 33 41 49 57
# 6  2 10 18 26 34 42 50 58
# 5  3 11 19 27 35 43 51 59
# 4  4 12 20 28 36 44 52 60
# 3  5 13 21 29 37 45 53 61
# 2  6 14 22 30 38 46 54 62
# 1  7 15 23 31 39 47 55 63
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
# 0 ... 9 = underpromotions
# plane // 3 == 0: rook
# plane // 3 == 1: bishop
# plane // 3 == 2: knight
# plane % 3 == 0: forward
# plane % 3 == 1: right
# plane % 3 == 2: left
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
    possible_piece_positions: Array = INIT_POSSIBLE_PIECE_POSITIONS


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # 64 * 73 = 4672
    observation: Array = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "chess"

    @staticmethod
    def _from_fen(fen: str):
        from pgx.experimental.chess import from_fen

        warnings.warn(
            "State._from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
            DeprecationWarning,
        )
        return from_fen(fen)

    def _to_fen(self) -> str:
        from pgx.experimental.chess import to_fen

        warnings.warn(
            "State._to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
            DeprecationWarning,
        )
        return to_fen(self)


@dataclass
class Action:
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
        return Action(  # type: ignore
            from_=from_,
            to=TO_MAP[from_, plane],  # -1 if impossible move
            underpromotion=jax.lax.select(plane >= 9, jnp.int32(-1), jnp.int32(plane // 3)),
        )

    def _to_label(self):
        plane = PLANE_MAP[self.from_, self.to]
        # plane = jax.lax.select(self.underpromotion >= 0, ..., plane)
        return jnp.int32(self.from_) * 73 + jnp.int32(plane)


class Chess(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        state = State(current_player=current_player)  # type: ignore
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        state = _step(state, action)
        state = jax.lax.cond(
            (MAX_TERMINATION_STEPS <= state._step_count),
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
        return "chess"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: Array):
    a = Action._from_label(action)
    state = state.replace(_x=_update_zobrist_hash(state._x, a))  # type: ignore

    hash_ = state._x.zobrist_hash
    hash_ ^= _hash_castling_en_passant(state._x)

    state = state.replace(_x=_apply_move(state._x, a))  # type: ignore
    state = state.replace(_x=_flip(state._x), current_player=(state.current_player + 1) % 2)  # type: ignore

    hash_ ^= _hash_castling_en_passant(state._x)
    state = state.replace(_x=state._x._replace(zobrist_hash=hash_))  # type: ignore

    state = state.replace(_x=_update_history(state._x))  # type: ignore
    state = state.replace(legal_action_mask=_legal_action_mask(state._x))  # type: ignore
    state = _check_termination(state)
    return state


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


def _check_termination(state: State):
    has_legal_action = state.legal_action_mask.any()
    terminated = ~has_legal_action
    terminated |= state._x.halfmove_count >= 100
    terminated |= has_insufficient_pieces(state._x)
    rep = (state._x.hash_history == state._x.zobrist_hash).all(axis=1).sum() - 1
    terminated |= rep >= 2

    is_checkmate = (~has_legal_action) & _is_checking(_flip(state._x))
    # fmt: off
    reward = jax.lax.select(
        is_checkmate,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # fmt: on
    return state.replace(  # type: ignore
        terminated=terminated,
        rewards=reward,
    )


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
    is_insufficient = FALSE
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
            board=state.board.at[removed_pawn_pos].set(
                jax.lax.select(is_en_passant, EMPTY, state.board[removed_pawn_pos])
            )
    )
    state = state._replace(
            en_passant=jax.lax.select(
                (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2),
                jnp.int32((a.to + a.from_) // 2),
                jnp.int32(-1),
            )
    )
    # update counters
    captured = (state.board[a.to] < 0) | (is_en_passant)
    state = state._replace(
            halfmove_count=jax.lax.select(captured | (piece == PAWN), 0, state.halfmove_count + 1),
            fullmove_count=state.fullmove_count + jnp.int32(state.turn == 1),
        )
    # castling
    # Whether castling is possible or not is not checked here.
    # We assume that if castling is not possible, it is filtered out.
    # left
    state = state._replace(
            board=jax.lax.cond(
                (piece == KING) & (a.from_ == 32) & (a.to == 16),
                lambda: state.board.at[0].set(EMPTY).at[24].set(ROOK),
                lambda: state.board,
            ),
            # update rook position
            possible_piece_positions=jax.lax.cond(
                (piece == KING) & (a.from_ == 32) & (a.to == 16),
                lambda: state.possible_piece_positions.at[0, 0].set(24),
                lambda: state.possible_piece_positions,
            ),
        )
    # right
    state = state._replace(
            board=jax.lax.cond(
                (piece == KING) & (a.from_ == 32) & (a.to == 48),
                lambda: state.board.at[56].set(EMPTY).at[40].set(ROOK),
                lambda: state.board,
            ),
            # update rook position
            possible_piece_positions=jax.lax.cond(
                (piece == KING) & (a.from_ == 32) & (a.to == 48),
                lambda: state.possible_piece_positions.at[0, 14].set(40),
                lambda: state.possible_piece_positions,
            ),
        )
    # update my can_castle_xxx_side
    state = state._replace(
            can_castle_queen_side=state.can_castle_queen_side.at[0].set(
                jax.lax.select(
                    (a.from_ == 32) | (a.from_ == 0),
                    FALSE,
                    state.can_castle_queen_side[0],
                )
            ),
            can_castle_king_side=state.can_castle_king_side.at[0].set(
                jax.lax.select(
                    (a.from_ == 32) | (a.from_ == 56),
                    FALSE,
                    state.can_castle_king_side[0],
                )
            ),
        )
    # update opp can_castle_xxx_side
    state = state._replace(
            can_castle_queen_side=state.can_castle_queen_side.at[1].set(
                jax.lax.select(
                    (a.to == 7),
                    FALSE,
                    state.can_castle_queen_side[1],
                )
            ),
            can_castle_king_side=state.can_castle_king_side.at[1].set(
                jax.lax.select(
                    (a.to == 63),
                    FALSE,
                    state.can_castle_king_side[1],
                )
            ),
        )
    # promotion to queen
    piece = jax.lax.select(
        piece == PAWN & (a.from_ % 8 == 6) & (a.underpromotion < 0),
        QUEEN,
        piece,
    )
    # underpromotion
    piece = jax.lax.select(
        a.underpromotion < 0,
        piece,
        jnp.int32([ROOK, BISHOP, KNIGHT])[a.underpromotion],
    )
    # actually move
    state = state._replace(board=state.board.at[a.from_].set(EMPTY).at[a.to].set(piece))  # type: ignore
    # update possible piece positions
    ix = jnp.argmin(jnp.abs(state.possible_piece_positions[0, :] - a.from_))
    state = state._replace(possible_piece_positions=state.possible_piece_positions.at[0, ix].set(a.to))
    return state


def _flip_pos(x):
    """
    >>> _flip_pos(jnp.int32(34))
    Array(37, dtype=int32)
    >>> _flip_pos(jnp.int32(37))
    Array(34, dtype=int32)
    >>> _flip_pos(jnp.int32(-1))
    Array(-1, dtype=int32)
    """
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
            possible_piece_positions=state.possible_piece_positions[::-1],
    )


def _legal_action_mask(state: GameState) -> Array:
    def is_legal(a: Action):
        ok = _is_pseudo_legal(state, a)
        ok &= ~_is_checking(_flip(_apply_move(state, a)))

        return ok

    @jax.vmap
    def legal_norml_moves(from_):
        piece = state.board[from_]

        @jax.vmap
        def legal_label(to):
            a = Action(from_=from_, to=to)
            return jax.lax.select(
                (from_ >= 0) & (piece > 0) & (to >= 0) & is_legal(a),
                a._to_label(),
                jnp.int32(-1),
            )

        return legal_label(CAN_MOVE[piece, from_])

    def legal_underpromotions(mask):
        # from_ = 6 14 22 30 38 46 54 62
        # plane = 0 ... 8
        @jax.vmap
        def make_labels(from_):
            return from_ * 73 + jnp.arange(9)

        labels = make_labels(jnp.int32([6, 14, 22, 30, 38, 46, 54, 62])).flatten()

        @jax.vmap
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state.board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return jax.lax.select(ok, label, -1)

        ok_labels = legal_labels(labels)
        return ok_labels.flatten()

    def legal_en_passants():
        to = state.en_passant

        @jax.vmap
        def legal_labels(from_):
            ok = (
                (from_ >= 0)
                & (from_ < 64)
                & (to >= 0)
                & (state.board[from_] == PAWN)
                & (state.board[to - 1] == -PAWN)
            )
            a = Action(from_=from_, to=to)
            ok &= ~_is_checking(_flip(_apply_move(state, a)))  # type: ignore
            return jax.lax.select(ok, a._to_label(), -1)

        return legal_labels(jnp.int32([to - 9, to + 7]))

    def can_castle_king_side():
        ok = state.board[32] == KING
        ok &= state.board[56] == ROOK
        ok &= state.can_castle_king_side[0]
        ok &= state.board[40] == EMPTY
        ok &= state.board[48] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label))))

        ok &= ~_is_checking(_flip(state))
        ok &= is_ok(jnp.int32([2366, 2367])).all()

        return ok

    def can_castle_queen_side():
        ok = state.board[32] == KING
        ok &= state.board[0] == ROOK
        ok &= state.can_castle_queen_side[0]
        ok &= state.board[8] == EMPTY
        ok &= state.board[16] == EMPTY
        ok &= state.board[24] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label))))

        ok &= ~_is_checking(_flip(state))
        ok &= is_ok(jnp.int32([2364, 2365])).all()

        return ok

    actions = legal_norml_moves(state.possible_piece_positions[0]).flatten()  # include -1
    # +1 is to avoid setting True to the last element
    mask = jnp.zeros(64 * 73 + 1, dtype=jnp.bool_)
    mask = mask.at[actions].set(TRUE)

    # castling
    mask = mask.at[2364].set(jax.lax.select(can_castle_queen_side(), TRUE, mask[2364]))
    mask = mask.at[2367].set(jax.lax.select(can_castle_king_side(), TRUE, mask[2367]))

    # set en passant
    actions = legal_en_passants()
    mask = mask.at[actions].set(TRUE)

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(TRUE)

    return mask[:-1]


def _is_attacking(state: GameState, pos):
    @jax.vmap
    def can_move(from_):
        a = Action(from_=from_, to=pos)
        return (from_ != -1) & _is_pseudo_legal(state, a)

    return can_move(CAN_MOVE_ANY[pos, :]).any()


def _is_checking(state: GameState):
    """True if possible to capture the opponent king"""
    opp_king_pos = jnp.argmin(jnp.abs(state.board - -KING))
    return _is_attacking(state, opp_king_pos)


def _is_pseudo_legal(state: GameState, a: Action):
    piece = state.board[a.from_]
    ok = (piece >= 0) & (state.board[a.to] <= 0)
    ok &= (CAN_MOVE[piece, a.from_] == a.to).any()
    between_ixs = BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
    # filter pawn move
    ok &= ~((piece == PAWN) & ((a.to % 8) < (a.from_ % 8)))
    ok &= ~((piece == PAWN) & (jnp.abs(a.to - a.from_) <= 2) & (state.board[a.to] < 0))
    ok &= ~((piece == PAWN) & (jnp.abs(a.to - a.from_) > 2) & (state.board[a.to] >= 0))
    return (a.to >= 0) & ok


def _possible_piece_positions(state):
    my_pos = jnp.nonzero(state._x.board > 0, size=16, fill_value=-1)[0].astype(jnp.int32)
    opp_pos = jnp.nonzero(_flip(state._x).board > 0, size=16, fill_value=-1)[0].astype(jnp.int32)
    return jnp.vstack((my_pos, opp_pos))


def _observe(state: State, player_id: Array):
    color = jax.lax.select(state.current_player == player_id, state._x.turn, 1 - state._x.turn)
    ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

    state = state.replace(  # type: ignore
        _x=jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x)),
    )

    def make(i):
        board = _rotate(state._x.board_history[i].reshape((8, 8)))

        def piece_feat(p):
            return (board == p).astype(jnp.float32)

        my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
        opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

        h = state._x.hash_history[i, :]
        rep = (state._x.hash_history == h).all(axis=1).sum() - 1
        rep = jax.lax.select((h == 0).all(), 0, rep)
        rep0 = ones * (rep == 0)
        rep1 = ones * (rep >= 1)
        return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

    board_feat = jax.vmap(make)(jnp.arange(8)).reshape(-1, 8, 8)
    color = color * ones
    total_move_cnt = (state._step_count / MAX_TERMINATION_STEPS) * ones
    my_queen_side_castling_right = ones * state._x.can_castle_queen_side[0]
    my_king_side_castling_right = ones * state._x.can_castle_king_side[0]
    opp_queen_side_castling_right = ones * state._x.can_castle_queen_side[1]
    opp_king_side_castling_right = ones * state._x.can_castle_king_side[1]
    no_prog_cnt = (state._x.halfmove_count.astype(jnp.float32) / 100.0) * ones

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


def _zobrist_hash(state):
    """
    >>> state = State()
    >>> _zobrist_hash(state)
    Array([1172276016, 1112364556], dtype=uint32)
    """
    hash_ = jnp.zeros(2, dtype=jnp.uint32)
    hash_ = jax.lax.select(state._x.turn == 0, hash_, hash_ ^ ZOBRIST_SIDE)
    board = jax.lax.select(state._x.turn == 0, state._x.board, _flip(state._x).board)

    def xor(i, h):
        # 0, ..., 12 (white pawn, ..., black king)
        piece = board[i] + 6
        return h ^ ZOBRIST_BOARD[i, piece]

    hash_ = jax.lax.fori_loop(0, 64, xor, hash_)
    hash_ ^= _hash_castling_en_passant(state._x)
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


def _from_fen(fen: str):
    from pgx.experimental.chess import from_fen

    warnings.warn(
        "_from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
        DeprecationWarning,
    )
    return from_fen(fen)


def _to_fen(state: State):
    from pgx.experimental.chess import to_fen

    warnings.warn(
        "_to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
        DeprecationWarning,
    )
    return to_fen(state)
