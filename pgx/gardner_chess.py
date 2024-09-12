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
from pgx._src.gardner_chess_utils import (  # type: ignore
    BETWEEN,
    CAN_MOVE,
    CAN_MOVE_ANY,
    INIT_LEGAL_ACTION_MASK,
    PLANE_MAP,
    TO_MAP,
    ZOBRIST_BOARD,
    ZOBRIST_SIDE,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

MAX_TERMINATION_STEPS = 256


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
# 5  4  9 14 19 24
# 4  3  8 13 18 23
# 3  2  7 12 17 22
# 2  1  6 11 16 21
# 1  0  5 10 15 20
#    a  b  c  d  e
# board index (flipped black view)
# 5  0  5 10 15 20
# 4  1  6 11 16 21
# 3  2  7 12 17 22
# 2  3  8 13 18 23
# 1  4  9 14 19 24
#    a  b  c  d  e
# fmt: off
INIT_BOARD = jnp.int32([
    4, 1, 0, -1, -4,
    2, 1, 0, -1, -2,
    3, 1, 0, -1, -3,
    5, 1, 0, -1, -5,
    6, 1, 0, -1, -6,
])

INIT_ZOBRIST_HASH = jnp.uint32([2025569903, 1172890342])
# fmt: on


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    observation: Array = jnp.zeros((5, 5, 115), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    # --- Chess specific ---
    _turn: Array = jnp.int32(0)
    _board: Array = INIT_BOARD  # From top left, like FEN
    # # of moves since the last piece capture or pawn move
    _halfmove_count: Array = jnp.int32(0)
    _fullmove_count: Array = jnp.int32(1)  # increase every black move
    _zobrist_hash: Array = jnp.uint32(INIT_ZOBRIST_HASH)
    _hash_history: Array = (
        jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(jnp.uint32(INIT_ZOBRIST_HASH))
    )
    _board_history: Array = jnp.zeros((8, 25), dtype=jnp.int32).at[0, :].set(INIT_BOARD)
    _possible_piece_positions: Array = jnp.int32(
        [
            [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
            [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
        ]
    )

    @staticmethod
    def _from_fen(fen: str):
        return _from_fen(fen)

    def _to_fen(self) -> str:
        return _to_fen(self)

    @property
    def env_id(self) -> core.EnvId:
        return "gardner_chess"


# Action
# 0 ... 9 = underpromotions
# plane // 3 == 0: rook
# plane // 3 == 1: bishop
# plane // 3 == 2: knight
# plane % 3 == 0: forward
# plane % 3 == 1: right
# plane % 3 == 2: left
# 33          16          32
#    34       15       31
#       35 44 14 48 30
#       42 36 13 29 46
# 17 18 19 20  X 21 22 23 24
#       41 28 12 37 45
#       27 43 11 47 38
#    26       10       39
# 25           9          40
@dataclass
class Action:
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: Array):
        """We use AlphaZero style label with channel-last representation: (5, 5, 49)

        49 = queen moves (32) + knight moves (8) + underpromotions (3 * 3)
        """
        from_, plane = label // 49, label % 49
        return Action(  # type: ignore
            from_=from_,
            to=TO_MAP[from_, plane],  # -1 if impossible move
            underpromotion=jax.lax.select(plane >= 9, jnp.int32(-1), jnp.int32(plane // 3)),
        )

    def _to_label(self):
        plane = PLANE_MAP[self.from_, self.to]
        # plane = jax.lax.select(self.underpromotion >= 0, ..., plane)
        return jnp.int32(self.from_) * 49 + jnp.int32(plane)


class GardnerChess(core.Env):
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
        return "gardner_chess"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: Array):
    a = Action._from_label(action)
    state = _update_zobrist_hash(state, a)
    state = _apply_move(state, a)
    state = _flip(state)
    state = _update_history(state)
    state = state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore
    state = _check_termination(state)
    return state


def _update_history(state: State):
    # board history
    board_history = jnp.roll(state._board_history, 25)
    board_history = board_history.at[0].set(state._board)
    state = state.replace(_board_history=board_history)  # type:ignore
    # hash hist
    hash_hist = jnp.roll(state._hash_history, 2)
    hash_hist = hash_hist.at[0].set(state._zobrist_hash)
    state = state.replace(_hash_history=hash_hist)  # type: ignore
    return state


def _apply_move(state: State, a: Action):
    # apply move action
    piece = state._board[a.from_]

    # update counters
    captured = state._board[a.to] < 0
    state = state.replace(  # type: ignore
        _halfmove_count=jax.lax.select(captured | (piece == PAWN), 0, state._halfmove_count + 1),
        _fullmove_count=state._fullmove_count + jnp.int32(state._turn == 1),
    )

    # promotion to queen
    piece = jax.lax.select(
        piece == PAWN & (a.from_ % 5 == 3) & (a.underpromotion < 0),
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
    state = state.replace(_board=state._board.at[a.from_].set(EMPTY).at[a.to].set(piece))  # type: ignore

    # update possible piece positions
    ix = jnp.argmin(jnp.abs(state._possible_piece_positions[0, :] - a.from_))
    state = state.replace(_possible_piece_positions=state._possible_piece_positions.at[0, ix].set(a.to))  # type: ignore
    return state


def _check_termination(state: State):
    has_legal_action = state.legal_action_mask.any()
    terminated = ~has_legal_action
    terminated |= state._halfmove_count >= 100
    terminated |= has_insufficient_pieces(state)
    rep = (state._hash_history == state._zobrist_hash).all(axis=1).sum() - 1
    terminated |= rep >= 2

    is_checkmate = (~has_legal_action) & _is_checking(_flip(state))
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


def has_insufficient_pieces(state: State):
    # Uses the same condition as OpenSpiel.
    # See https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/chess/chess_board.cc#L724
    num_pieces = (state._board != EMPTY).sum()
    num_pawn_rook_queen = ((jnp.abs(state._board) >= ROOK) | (jnp.abs(state._board) == PAWN)).sum() - 2  # two kings
    num_bishop = (jnp.abs(state._board) == 3).sum()
    # [ 0  2  4  6 16 18 20 22 32 34 36 38 48 50 52 54 9 11 13 15 25 27 29 31 41 43 45 47 57 59 61 63]
    black_coords = jnp.arange(0, 25, 2)
    num_bishop_on_black = (jnp.abs(state._board[black_coords]) == BISHOP).sum()
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


def _legal_action_mask(state):
    def is_legal(a: Action):
        ok = _is_pseudo_legal(state, a)
        next_s = _flip(_apply_move(state, a))
        ok &= ~_is_checking(next_s)

        return ok

    @jax.vmap
    def legal_normal_moves(from_):
        piece = state._board[from_]

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
        # from_ = 3, 8, 13, 18, 23
        # plane = 0 ... 8
        @jax.vmap
        def make_labels(from_):
            return from_ * 49 + jnp.arange(9)

        labels = make_labels(jnp.int32([3, 8, 13, 18, 23])).flatten()

        @jax.vmap
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state._board[a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return jax.lax.select(ok, label, -1)

        ok_labels = legal_labels(labels)
        return ok_labels.flatten()

    actions = legal_normal_moves(state._possible_piece_positions[0]).flatten()  # include -1
    # +1 is to avoid setting True to the last element
    mask = jnp.zeros(25 * 49 + 1, dtype=jnp.bool_)
    mask = mask.at[actions].set(TRUE)

    # set underpromotions
    actions = legal_underpromotions(mask)
    mask = mask.at[actions].set(TRUE)

    return mask[:-1]


def _is_attacking(state: State, pos):
    @jax.vmap
    def can_move(from_):
        a = Action(from_=from_, to=pos)
        return (from_ != -1) & _is_pseudo_legal(state, a)

    return can_move(CAN_MOVE_ANY[pos, :]).any()


def _is_checking(state: State):
    """True if possible to capture the opponent king"""
    opp_king_pos = jnp.argmin(jnp.abs(state._board - -KING))
    return _is_attacking(state, opp_king_pos)


def _is_pseudo_legal(state: State, a: Action):
    piece = state._board[a.from_]
    ok = (piece >= 0) & (state._board[a.to] <= 0)
    ok &= (CAN_MOVE[piece, a.from_] == a.to).any()
    between_ixs = BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state._board[between_ixs] == EMPTY)).all()
    # filter pawn move
    ok &= ~((piece == PAWN) & (a.to // 5 == a.from_ // 5) & (state._board[a.to] < 0))
    ok &= ~((piece == PAWN) & (a.to // 5 != a.from_ // 5) & (state._board[a.to] >= 0))
    return (a.to >= 0) & ok


def _zobrist_hash(state):
    """
    >>> state = State()
    >>> _zobrist_hash(state)
    Array([2025569903, 1172890342], dtype=uint32)
    """
    hash_ = jnp.zeros(2, dtype=jnp.uint32)
    hash_ = jax.lax.select(state._turn == 0, hash_, hash_ ^ ZOBRIST_SIDE)
    board = jax.lax.select(state._turn == 0, state._board, _flip(state)._board)

    def xor(i, h):
        # 0, ..., 12 (white pawn, ..., black king)
        piece = board[i] + 6
        return h ^ ZOBRIST_BOARD[i, piece]

    hash_ = jax.lax.fori_loop(0, 25, xor, hash_)
    return hash_


def _update_zobrist_hash(state: State, action: Action):
    hash_ = state._zobrist_hash
    source_piece = state._board[action.from_]
    source_piece = jax.lax.select(state._turn == 0, source_piece + 6, (source_piece * -1) + 6)
    destination_piece = state._board[action.to]
    destination_piece = jax.lax.select(state._turn == 0, destination_piece + 6, (destination_piece * -1) + 6)
    from_ = jax.lax.select(state._turn == 0, action.from_, _flip_pos(action.from_))
    to = jax.lax.select(state._turn == 0, action.to, _flip_pos(action.to))
    hash_ ^= ZOBRIST_BOARD[from_, source_piece]  # remove the piece from the source pos
    hash_ ^= ZOBRIST_BOARD[from_, 6]  # make the source pos empty
    hash_ ^= ZOBRIST_BOARD[to, destination_piece]  # remove the piece from the target pos (including empty)
    # underpromotion
    source_piece = jax.lax.select(
        action.underpromotion >= 0,
        jax.lax.select(
            state._turn == 0,
            source_piece + 3 - action.underpromotion,
            source_piece - (3 - action.underpromotion),
        ),
        source_piece,
    )
    hash_ ^= ZOBRIST_BOARD[to, source_piece]  # put the piece to the target pos
    hash_ ^= ZOBRIST_SIDE
    return state.replace(  # type: ignore
        _zobrist_hash=hash_,
    )


def _observe(state: State, player_id: Array):
    color = jax.lax.select(state.current_player == player_id, state._turn, 1 - state._turn)
    ones = jnp.ones((1, 5, 5), dtype=jnp.float32)

    state = jax.lax.cond(state.current_player == player_id, lambda: state, lambda: _flip(state))

    def make(i):
        board = _rotate(state._board_history[i].reshape((5, 5)))

        def piece_feat(p):
            return (board == p).astype(jnp.float32)

        my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
        opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

        h = state._hash_history[i, :]
        rep = (state._hash_history == h).all(axis=1).sum() - 1
        rep = jax.lax.select((h == 0).all(), 0, rep)
        rep0 = ones * (rep == 0)
        rep1 = ones * (rep >= 1)
        return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

    board_feat = jax.vmap(make)(jnp.arange(8)).reshape(-1, 5, 5)
    color = color * ones
    total_move_cnt = (state._step_count / MAX_TERMINATION_STEPS) * ones
    no_prog_cnt = (state._halfmove_count.astype(jnp.float32) / 100.0) * ones

    return jnp.vstack([board_feat, color, total_move_cnt, no_prog_cnt]).transpose((1, 2, 0))


def _possible_piece_positions(state: State):
    my_pos = jnp.nonzero(state._board > 0, size=10, fill_value=-1)[0].astype(jnp.int32)
    opp_pos = jnp.nonzero(_flip(state)._board > 0, size=10, fill_value=-1)[0].astype(jnp.int32)
    return jnp.vstack((my_pos, opp_pos))


def _flip_pos(x):
    """
    >>> _flip_pos(jnp.int32(0))
    Array(4, dtype=int32)
    >>> _flip_pos(jnp.int32(4))
    Array(0, dtype=int32)
    >>> _flip_pos(jnp.int32(-1))
    Array(-1, dtype=int32)
    """
    return jax.lax.select(x == -1, x, (x // 5) * 5 + (4 - (x % 5)))


def _flip(state: State) -> State:
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        _board=-jnp.flip(state._board.reshape(5, 5), axis=1).flatten(),
        _turn=(state._turn + 1) % 2,
        _board_history=-jnp.flip(state._board_history.reshape(8, 5, 5), axis=-1).reshape(-1, 25),
        _possible_piece_positions=state._possible_piece_positions[::-1],
    )


def _rotate(board):
    return jnp.rot90(board, k=1)


def _from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen("rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1")
    >>> _rotate(state._board.reshape(5, 5))
    Array([[-4, -2, -3, -5, -6],
           [-1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6]], dtype=int32)
    >>> state = _from_fen("bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1")
    >>> _rotate(state._board.reshape(5, 5))
    Array([[-4, -2, -3, -5, -6],
           [ 0, -1, -1, -1,  0],
           [ 0,  0,  0,  0,  0],
           [-1,  1,  1,  1,  1],
           [ 3,  3,  6,  5,  4]], dtype=int32)
    """
    board, turn, castling, en_passant, halfmove_cnt, fullmove_cnt = fen.split()
    arr = []
    for line in board.split("/"):
        for c in line:
            if str.isnumeric(c):
                for _ in range(int(c)):
                    arr.append(0)
            else:
                ix = "pnbrqk".index(str.lower(c)) + 1
                if str.islower(c):
                    ix *= -1
                arr.append(ix)
    mat = jnp.int32(arr).reshape(5, 5)
    if turn == "b":
        mat = -jnp.flip(mat, axis=0)
    state = State(  # type: ignore
        _board=jnp.rot90(mat, k=3).flatten(),
        _turn=jnp.int32(0) if turn == "w" else jnp.int32(1),
        _halfmove_count=jnp.int32(halfmove_cnt),
        _fullmove_count=jnp.int32(fullmove_cnt),
    )
    state = state.replace(_possible_piece_positions=jax.jit(_possible_piece_positions)(state))  # type: ignore
    state = state.replace(  # type: ignore
        legal_action_mask=jax.jit(_legal_action_mask)(state),
    )
    state = state.replace(_zobrist_hash=_zobrist_hash(state))  # type: ignore
    state = _update_history(state)
    state = jax.jit(_check_termination)(state)
    state = state.replace(observation=jax.jit(_observe)(state, state.current_player))  # type: ignore
    return state


def _to_fen(state: State):
    """Convert state into FEN expression.

    See chess.py for the explanation of FEN.

    >>> s = State()
    >>> _to_fen(s)
    'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1'
    >>> _to_fen(_from_fen("bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1"))
    'bbkqr/Ppppp/5/1PPP1/RNBQK b - - 0 1'
    """
    pb = jnp.rot90(state._board.reshape(5, 5), k=1)
    if state._turn == 1:
        pb = -jnp.flip(pb, axis=0)
    fen = ""
    # board
    for i in range(5):
        space_length = 0
        for j in range(5):
            piece = pb[i, j]
            if piece == 0:
                space_length += 1
            elif space_length != 0:
                fen += str(space_length)
                space_length = 0
            if piece != 0:
                if piece > 0:
                    fen += "PNBRQK"[piece - 1]
                else:
                    fen += "pnbrqk"[-piece - 1]
        if space_length != 0:
            fen += str(space_length)
        if i != 4:
            fen += "/"
        else:
            fen += " "
    # turn
    fen += "w " if state._turn == 0 else "b "
    # castling and em passant
    fen += "- - "
    fen += str(state._halfmove_count.item())
    fen += " "
    fen += str(state._fullmove_count.item())
    return fen
