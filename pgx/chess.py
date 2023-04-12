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
from pgx._chess_utils import (  # type: ignore
    BETWEEN,
    CAN_MOVE,
    INIT_LEGAL_ACTION_MASK,
    PLANE_MAP,
    TO_MAP,
)
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int8(0)
PAWN = jnp.int8(1)
KNIGHT = jnp.int8(2)
BISHOP = jnp.int8(3)
ROOK = jnp.int8(4)
QUEEN = jnp.int8(5)
KING = jnp.int8(6)
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
#    a  b  c  d  e  f  g  f
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
INIT_BOARD = jnp.int8([
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


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = INIT_LEGAL_ACTION_MASK  # 64 * 73 = 4672
    observation: jnp.ndarray = jnp.zeros((8, 8, 119), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Chess specific ---
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD  # 左上からFENと同じ形式で埋めていく
    # (curr, opp), flips every turn
    can_castle_queen_side: jnp.ndarray = jnp.ones(2, dtype=jnp.bool_)
    can_castle_king_side: jnp.ndarray = jnp.ones(2, dtype=jnp.bool_)
    en_passant: jnp.ndarray = jnp.int8(-1)  # En passant target. does not flip
    # # of moves since the last piece capture or pawn move
    halfmove_count: jnp.ndarray = jnp.int32(0)
    fullmove_count: jnp.ndarray = jnp.int32(1)  # increase every black move

    @staticmethod
    def _from_fen(fen: str):
        return _from_fen(fen)

    def _to_fen(self) -> str:
        return _to_fen(self)


@dataclass
class Action:
    from_: jnp.ndarray = jnp.int8(-1)
    to: jnp.ndarray = jnp.int8(-1)
    underpromotion: jnp.ndarray = jnp.int8(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: jnp.ndarray):
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
            underpromotion=jax.lax.select(
                plane >= 9, jnp.int8(-1), jnp.int8(plane // 3)
            ),
        )

    def _to_label(self):
        plane = PLANE_MAP[self.from_, self.to]
        # plane = jax.lax.select(self.underpromotion >= 0, ..., plane)
        return jnp.int32(self.from_) * 73 + jnp.int32(plane)


class Chess(core.Env):
    def __init__(self, max_termination_steps: int = 1000):
        super().__init__()
        self.max_termination_steps = max_termination_steps

    def _init(self, key: jax.random.KeyArray) -> State:
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        state = State(current_player=current_player)  # type: ignore
        state = state.replace(legal_action_mask=INIT_LEGAL_ACTION_MASK)  # type: ignore
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
        return jnp.zeros((8, 8, 119), dtype=jnp.bool_)

    @property
    def name(self) -> str:
        return "Chess"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: jnp.ndarray):
    a = Action._from_label(action)
    state = _apply_move(state, a)
    state = _flip(state)
    legal_action_mask = _legal_action_mask(state)

    terminated = ~legal_action_mask.any()
    # TODO: stailmate

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


def _apply_move(state: State, a: Action):
    # apply move action
    piece = state.board[a.from_]
    # en passant
    is_en_passant = (
        (state.en_passant >= 0)
        & (piece == PAWN)
        & (state.en_passant == _abs_pos(a.to, state.turn))
    )
    removed_pawn_pos = a.to - 1
    state = state.replace(  # type: ignore
        board=state.board.at[removed_pawn_pos].set(
            jax.lax.select(is_en_passant, EMPTY, state.board[removed_pawn_pos])
        ),
    )
    state = state.replace(  # type: ignore
        en_passant=jax.lax.select(
            (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2),
            _abs_pos(jnp.int8((a.to + a.from_) // 2), state.turn),
            jnp.int8(-1),
        )
    )
    # castling
    # 可能かどうかの判断はここでは行わない。castlingがlegalでない場合はフィルタされている前提
    # left
    state = state.replace(  # type: ignore
        board=jax.lax.cond(
            (piece == KING) & (a.from_ == 32) & (a.to == 16),
            lambda: state.board.at[0].set(EMPTY).at[24].set(ROOK),
            lambda: state.board,
        )
    )
    # right
    state = state.replace(  # type: ignore
        board=jax.lax.cond(
            (piece == KING) & (a.from_ == 32) & (a.to == 48),
            lambda: state.board.at[56].set(EMPTY).at[40].set(ROOK),
            lambda: state.board,
        )
    )
    state = state.replace(  # type: ignore
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
        jnp.int8([ROOK, BISHOP, KNIGHT])[a.underpromotion],
    )
    state = state.replace(  # type: ignore
        board=state.board.at[a.from_].set(EMPTY).at[a.to].set(piece)
    )
    return state


def _abs_pos(x, turn):
    return jax.lax.select(turn == 0, x, (x // 8) * 8 + (7 - x % 8))


def _flip_pos(x):
    """
    >>> _flip_pos(34)
    37
    >>> _flip_pos(37)
    34
    """
    return (x // 8) * 8 + (7 - (x % 8))


def _rotate(board):
    return jnp.rot90(board, k=1)


def _flip(state: State) -> State:
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        board=-jnp.flip(state.board.reshape(8, 8), axis=1).flatten(),
        turn=(state.turn + 1) % 2,
        can_castle_queen_side=state.can_castle_queen_side[::-1],
        can_castle_king_side=state.can_castle_king_side[::-1],
    )


def _legal_action_mask(state):
    def is_legal(a: Action):
        ok = _is_pseudo_legal(state, a)
        next_s = _flip(_apply_move(state, a))
        ok &= ~_is_checking(next_s)

        return ok

    @jax.vmap
    def legal_norml_moves(from_):
        piece = state.board[from_]

        @jax.vmap
        def legal_label(to):
            a = Action(from_=from_, to=to)
            return jax.lax.select(
                (piece >= 0) & (to >= 0) & is_legal(a),
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

        labels = make_labels(
            jnp.int32([6, 14, 22, 30, 38, 46, 54, 62])
        ).flatten()

        @jax.vmap
        def legal_labels(label):
            a = Action._from_label(label)
            ok = mask[Action(from_=a.from_, to=a.to)._to_label()]
            return jax.lax.select(ok, label, -1)

        ok_labels = legal_labels(labels)
        return ok_labels.flatten()

    def legal_en_passnts():
        to = state.en_passant
        # flip if black turn
        to = jax.lax.select((to >= 0) & (state.turn == 1), _flip_pos(to), to)

        @jax.vmap
        def legal_labels(from_):
            ok = (
                (to >= 0)
                & (state.board[from_] == PAWN)
                & (state.board[to - 1] == -PAWN)
            )
            return jax.lax.select(
                ok, Action(from_=from_, to=to)._to_label(), -1
            )

        return legal_labels(jnp.int32([to - 9, to + 7]))

    def can_castle_king_side():
        ok = state.board[32] == KING
        ok &= state.can_castle_king_side[0]
        ok &= state.board[40] == EMPTY
        ok &= state.board[48] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label))))

        ok &= is_ok(jnp.int32([2366, 2367])).all()

        return ok

    def can_castle_queen_side():
        ok = state.board[32] == KING
        ok &= state.can_castle_king_side[0]
        ok &= state.board[8] == EMPTY
        ok &= state.board[16] == EMPTY
        ok &= state.board[24] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label))))

        ok &= is_ok(jnp.int32([2364, 2365])).all()

        return ok

    actions = legal_norml_moves(jnp.arange(64)).flatten()  # include -1
    # +1 is to avoid setting True to the last element
    mask = jnp.zeros(64 * 73 + 1, dtype=jnp.bool_)
    mask = mask.at[actions].set(TRUE)

    # castling
    mask = mask.at[2364].set(can_castle_queen_side())
    mask = mask.at[2367].set(can_castle_king_side())

    # set en passant
    actions = legal_en_passnts()
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

    return can_move(CAN_MOVE[QUEEN, pos, :]).any()


def _is_checking(state: State):
    """True if possible to capture the opponent king"""
    opp_king_pos = jnp.argmin(jnp.abs(state.board - -KING))
    return _is_attacking(state, opp_king_pos)


def _is_pseudo_legal(state: State, a: Action):
    piece = state.board[a.from_]
    ok = (piece >= 0) & (state.board[a.to] <= 0)
    ok &= (CAN_MOVE[piece, a.from_] == a.to).any()
    between_ixs = BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state.board[between_ixs] == EMPTY)).all()
    # filter pawn move
    ok &= ~((piece == PAWN) & ((a.to % 8) < (a.from_ % 8)))
    ok &= ~(
        (piece == PAWN)
        & (jnp.abs(a.to - a.from_) <= 2)
        & (state.board[a.to] < 0)
    )
    ok &= ~(
        (piece == PAWN)
        & (jnp.abs(a.to - a.from_) > 2)
        & (state.board[a.to] >= 0)
    )
    return (a.to >= 0) & ok


def _from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen(
    ...     "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR w KQkq e3 0 1"
    ... )
    >>> _rotate(state.board.reshape(8, 8))
    Array([[-4, -2, -3, -5, -6, -3, -2, -4],
           [-1, -1, -1, -1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  1,  1,  1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6,  3,  2,  4]], dtype=int8)
    >>> state.en_passant
    Array(34, dtype=int8)
    >>> state = _from_fen(
    ...     "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1"
    ... )
    >>> _rotate(state.board.reshape(8, 8))
    Array([[-4, -2, -3, -5, -6, -3, -2, -4],
           [ 0, -1, -1, -1, -1, -1, -1, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1,  1,  1,  1],
           [ 4,  2,  3,  5,  6,  3,  2,  4]], dtype=int8)
    >>> state.en_passant
    Array(34, dtype=int8)
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
    can_castle_queen_side = jnp.zeros(2, dtype=jnp.bool_)
    can_castle_king_side = jnp.zeros(2, dtype=jnp.bool_)
    if "Q" in castling:
        can_castle_queen_side = can_castle_queen_side.at[0].set(TRUE)
    if "q" in castling:
        can_castle_queen_side = can_castle_queen_side.at[1].set(TRUE)
    if "K" in castling:
        can_castle_king_side = can_castle_king_side.at[0].set(TRUE)
    if "k" in castling:
        can_castle_king_side = can_castle_king_side.at[1].set(TRUE)
    if turn == "b":
        can_castle_queen_side = can_castle_queen_side[::-1]
        can_castle_king_side = can_castle_king_side[::-1]
    mat = jnp.int8(arr).reshape(8, 8)
    if turn == "b":
        mat = -jnp.flip(mat, axis=0)
    state = State(  # type: ignore
        board=jnp.rot90(mat, k=3).flatten(),
        turn=jnp.int8(0) if turn == "w" else jnp.int8(1),
        can_castle_queen_side=can_castle_queen_side,
        can_castle_king_side=can_castle_king_side,
        en_passant=jnp.int8(-1)
        if en_passant == "-"
        else jnp.int8(
            "abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1
        ),
        halfmove_count=jnp.int32(halfmove_cnt),
        fullmove_count=jnp.int32(fullmove_cnt),
    )
    return state.replace(  # type: ignore
        legal_action_mask=_legal_action_mask(state)
    )


def _to_fen(state: State):
    """Convert state into FEN expression.

    - ポーン:P ナイト:N ビショップ:B ルーク:R クイーン:Q キング:K
    - 先手の駒は大文字、後手の駒は小文字で表現
    - 空白の場合、連続する空白の数を入れて次の駒にシフトする。P空空空RならP3R
    - 左上から開始して右に見ていく
    - 段が変わるときは/を挿入
    - 盤面の記入が終わったら手番（w/b）
    - キャスリングの可否。キングサイドにできる場合はK, クイーンサイドにできる場合はQを先後それぞれ書く。全部不可なら-
    - アンパッサン可能な位置。ポーンが2マス動いた場合はそのポーンが通過した位置を記録
    - 最後にポーンの移動および駒取りが発生してからの手数と通常の手数（0, 1で固定にする）

    >>> s = State(en_passant=jnp.int8(34))
    >>> _to_fen(s)
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1'
    >>> _to_fen(
    ...     _from_fen(
    ...         "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1"
    ...     )
    ... )
    'rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1'
    """
    pb = jnp.rot90(state.board.reshape(8, 8), k=1)
    if state.turn == 1:
        pb = -jnp.flip(pb, axis=0)
    fen = ""
    # 盤面
    for i in range(8):
        space_length = 0
        for j in range(8):
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
        if i != 7:
            fen += "/"
        else:
            fen += " "
    # 手番
    fen += "w " if state.turn == 0 else "b "
    # キャスリング
    can_castle_queen_side = state.can_castle_queen_side
    can_castle_king_side = state.can_castle_king_side
    if state.turn == 1:
        can_castle_queen_side = can_castle_queen_side[::-1]
        can_castle_king_side = can_castle_king_side[::-1]
    if not (can_castle_queen_side.any() | can_castle_king_side.any()):
        fen += "-"
    else:
        if can_castle_king_side[0]:
            fen += "K"
        if can_castle_queen_side[0]:
            fen += "Q"
        if can_castle_king_side[1]:
            fen += "k"
        if can_castle_queen_side[1]:
            fen += "q"
    fen += " "
    # アンパッサン
    ep = int(state.en_passant.item())
    if ep == -1:
        fen += "-"
    else:
        fen += "abcdefgh"[ep // 8]
        fen += str(ep % 8 + 1)
    fen += " "
    fen += str(state.halfmove_count.item())
    fen += " "
    fen += str(state.fullmove_count.item())
    return fen
