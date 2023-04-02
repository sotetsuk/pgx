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

EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
KNIGHT = jnp.int8(1)
BISHOP = jnp.int8(2)
ROOK = jnp.int8(3)
QUEEN = jnp.int8(4)
KING = jnp.int8(5)
# OPP_PAWN = 6
# OPP_KNIGHT = 7
# OPP_BISHOP = 8
# OPP_ROOK = 9
# OPP_QUEEN = 10
# OPP_KING = 11

INIT_BOARD = jnp.int8([
   [ 9,  7,  8, 11, 10,  8,  7,  9],
   [ 6,  6,  6,  6,  6,  6,  6,  6],
   [-1, -1, -1, -1, -1, -1, -1, -1],
   [-1, -1, -1, -1, -1, -1, -1, -1],
   [-1, -1, -1, -1, -1, -1, -1, -1],
   [-1, -1, -1, -1, -1, -1, -1, -1],
   [ 0,  0,  0,  0,  0,  0,  0,  0],
   [ 3,  1,  2,  5,  4,  2,  1,  3]
])


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(73 * 64, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((8, 8, 119), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Chess specific ---
    turn: jnp.ndarray = jnp.int8(0)
    board: jnp.ndarray = INIT_BOARD  # 左上からFENと同じ形式で埋めていく


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
    >>> s = State()
    >>> _to_fen(s)
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    """
    pb = state.board.reshape(8, 8)
    fen = ""
    # fmt: off
    board_char_dir = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    board_line_dir = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # fmt: on
    # 盤面
    for i in range(8):
        space_length = 0
        for j in range(8):
            piece = pb[i, j]
            if piece == -1:
                space_length += 1
            elif space_length != 0:
                fen += str(space_length)
                space_length = 0
            if piece != -1:
                fen += board_char_dir[piece]
        if space_length != 0:
            fen += str(space_length)
        if i != 7:
            fen += "/"
        else:
            fen += " "
    # 手番
    fen += "w " if state.turn == 0 else "b "
    # キャスリング
    # wk_cas = not state.wk_move_count and not state.wr2_move_count
    # wq_cas = not state.wk_move_count and not state.wr1_move_count
    # bk_cas = not state.bk_move_count and not state.br2_move_count
    # bq_cas = not state.bk_move_count and not state.br1_move_count
    # if not wk_cas and not wq_cas and not bk_cas and not bq_cas:
    #     fen += "- "
    # else:
    #     if wk_cas:
    #         fen += "K"
    #     if wq_cas:
    #         fen += "Q"
    #     if bk_cas:
    #         fen += "k"
    #     if bq_cas:
    #         fen += "q"
    #     fen += " "
    # アンパッサン
    # if state.en_passant == -1:
    #     fen += "- "
    # else:
    #     if state.turn == 0:
    #         en = state.en_passant + 1
    #     else:
    #         en = state.en_passant - 1
    #     fen += board_line_dir[en // 8]
    #     fen += str(en % 8 + 1)
    #     fen += " "
    fen += "0 1"
    return fen