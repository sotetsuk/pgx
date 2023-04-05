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


# board index
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  f
# fmt: off
INIT_BOARD = jnp.int8([
    -4, -1, 0, 0, 0, 0, 1, 4,
    -2, -1, 0, 0, 0, 0, 1, 2,
    -3, -1, 0, 0, 0, 0, 1, 3,
    -6, -1, 0, 0, 0, 0, 1, 6,
    -5, -1, 0, 0, 0, 0, 1, 5,
    -3, -1, 0, 0, 0, 0, 1, 3,
    -2, -1, 0, 0, 0, 0, 1, 2,
    -4, -1, 0, 0, 0, 0, 1, 4
])
TO_MAP = - jnp.ones((64, 73), dtype=jnp.int8)
# underpromotion
for from_ in range(8, 16):
    for plane in range(9):
        dir_ = plane % 3
        to = from_ + jnp.int8([-8, -7, -9])[dir_]
        if not(0 <= to < 8):
            continue
        TO_MAP = TO_MAP.at[from_, plane].set(to)
# normal move
seq = list(range(1, 8))
zeros = [0 for _ in range(7)]
# 下
dr = [-x for x in seq[::-1]]
dc = [0 for _ in range(7)]
# 上
dr += [x for x in seq]
dc += [0 for _ in range(7)]
# 左
dr += [0 for _ in range(7)]
dc += [-x for x in seq[::-1]]
# 右
dr += [0 for _ in range(7)]
dc += [x for x in seq]
# 左下
dr += [-x for x in seq[::-1]]
dc += [-x for x in seq[::-1]]
# 右上
dr += [x for x in seq]
dc += [x for x in seq]
# 左上
dr += [x for x in seq[::-1]]
dc += [-x for x in seq[::-1]]
# 右下
dr += [-x for x in seq]
dc += [x for x in seq]
dr = jnp.int8(dr)
dc = jnp.int8(dc)
for from_ in range(64):
    for plane in range(9, 73):
        r, c = jnp.int8(from_ % 8), jnp.int8(from_ // 8)
        r = r + dr[plane - 9]
        c = c + dc[plane - 9]
        if r < 0 or r >= 8 or c < 0 or c >= 8:
            continue
        TO_MAP = TO_MAP.at[from_, plane].set(c * 8 + r)
# fmt: on



@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(8 * 8 * 73, dtype=jnp.bool_)
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
        return Action(
            from_=from_,
            to=TO_MAP[from_, plane],  # -1 if impossible move
            underpromotion=jax.lax.select(plane >= 9, jnp.int8(-1), jnp.int8(plane // 3))
        )


class Chess(core.Env):
    def __init__(self, max_termination_steps: int = 1000):
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
    # apply move/drop action
    ...

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


def _flip(state):
    ...

def _legal_action_mask(state):
    return jnp.ones(8 * 8 * 73, dtype=jnp.bool_)


def _from_fen(fen: str):
    """Restore state from FEN

    >>> state = _from_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1')
    >>> state.board.reshape(8, 8)[:, 0]
    Array([-4, -2, -3, -6, -5, -3, -2, -4], dtype=int8)
    >>> state.en_passant
    Array(34, dtype=int8)
    """
    board, turn, castling, en_passant, halfmove_cnt, fullmove_cnt = fen.split()
    turn = jnp.int8(0) if turn == "w" else jnp.int8(1)
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
    if turn == 1:
        can_castle_queen_side = can_castle_queen_side[::-1]
        can_castle_king_side = can_castle_king_side[::-1]
    if en_passant == "-":
        en_passant = jnp.int8(-1)
    else:
        en_passant = jnp.int8(
            "abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1
        )
    state = State(
        board=jnp.rot90(jnp.int8(arr).reshape(8, 8), k=1).flatten(),
        turn=turn,
        can_castle_queen_side=can_castle_queen_side,
        can_castle_king_side=can_castle_king_side,
        en_passant=en_passant,
        halfmove_count=jnp.int32(halfmove_cnt),
        fullmove_count=jnp.int32(fullmove_cnt),
    )
    return state


def _to_fen(state: State):
    """Convert state into fen expression.
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
    """
    pb = jnp.rot90(state.board.reshape(8, 8), k=3)
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
