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
from pgx._cache import load_shogi_is_on_the_way  # type: ignore
from pgx._cache import load_shogi_legal_from_idx  # type: ignore
from pgx._cache import load_shogi_raw_effect_boards  # type: ignore
from pgx._flax.struct import dataclass

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

# fmt: off
INIT_PIECE_BOARD = jnp.int8([[15, -1, 14, -1, -1, -1, 0, -1, 1],  # noqa: E241
                             [16, 18, 14, -1, -1, -1, 0,  5, 2],  # noqa: E241
                             [17, -1, 14, -1, -1, -1, 0, -1, 3],  # noqa: E241
                             [20, -1, 14, -1, -1, -1, 0, -1, 6],  # noqa: E241
                             [21, -1, 14, -1, -1, -1, 0, -1, 7],  # noqa: E241
                             [20, -1, 14, -1, -1, -1, 0, -1, 6],  # noqa: E241
                             [17, -1, 14, -1, -1, -1, 0, -1, 3],  # noqa: E241
                             [16, 19, 14, -1, -1, -1, 0,  4, 2],  # noqa: E241
                             [15, -1, 14, -1, -1, -1, 0, -1, 1]]).flatten()  # noqa: E241
# fmt: on

# Can <piece,14> reach from <from,81> to <to,81> ignoring pieces on board?
CAN_MOVE = load_shogi_raw_effect_boards()  # bool (14, 81, 81)
# When <lance/bishop/rook/horse/dragon,5> moves from <from,81> to <to,81>,
# is <point,81> on the way between two points?
BETWEEN = load_shogi_is_on_the_way()  # bool (5, 81, 81, 81)
# Give <dir,10> and <to,81>, return the legal from idx
# E.g. LEGAL_FROM_IDX[Up, to=19] = [20, 21, ..., -1]
# Used for computing dlshogi action
LEGAL_FROM_IDX = load_shogi_legal_from_idx()  # (10, 81, 8)


NEIGHBOURS = [[] for i in range(81)]  # include knight moves

dx = [ 0, -1, -1, -1,  0, +1, +1, +1, +1,-1]
dy = [-1, -1,  0, +1, +1, +1,  0, -1, -2,-2]
for i in range(81):
    for j in range(10):
        x, y = i // 9, i % 9
        x += dx[j]
        y += dy[j]
        if x < 0 or x >= 9 or y < 0 or y >= 9:
            NEIGHBOURS[i].append(-1)
        else:
            NEIGHBOURS[i].append(x * 9 + y)

NEIGHBOURS = jnp.int8(NEIGHBOURS)




def _rotate(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.rot90(board.reshape(9, 9), k=3)


def _flip(state):
    empty_mask = state.piece_board == EMPTY
    pb = (state.piece_board + 14) % 28
    pb = jnp.where(empty_mask, EMPTY, pb)
    pb = pb[::-1]
    return state.replace(  # type: ignore
        piece_board=pb,
        hand=state.hand[jnp.int8((1, 0))],
    )


def _to_sfen(state):
    """Convert state into sfen expression.

    - 歩:P 香車:L 桂馬:N 銀:S 角:B 飛車:R 金:G 王:K
    - 成駒なら駒の前に+をつける（と金なら+P）
    - 先手の駒は大文字、後手の駒は小文字で表現
    - 空白の場合、連続する空白の数を入れて次の駒にシフトする。歩空空空飛ならP3R
    - 左上から開始して右に見ていく
    - 段が変わるときは/を挿入
    - 盤面の記入が終わったら手番（b/w）
    - 持ち駒は先手の物から順番はRBGSNLPの順
    - 最後に手数（1で固定）

    """
    if state.turn % 2 == 1:
        state = _flip(state)

    pb = jnp.rot90(state.piece_board.reshape((9, 9)), k=3)
    sfen = ""
    # fmt: off
    board_char_dir = ["", "P", "L", "N", "S", "B", "R", "G", "K", "+P", "+L", "+N", "+S", "+B", "+R", "p", "l", "n", "s", "b", "r", "g", "k", "+p", "+l", "+n", "+s", "+b", "+r"]
    hand_char_dir = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
    hand_dir = [5, 4, 6, 3, 2, 1, 0, 12, 11, 13, 10, 9, 8, 7]
    # fmt: on
    # 盤面
    for i in range(9):
        space_length = 0
        for j in range(9):
            piece = pb[i, j] + 1
            if piece == 0:
                space_length += 1
            elif space_length != 0:
                sfen += str(space_length)
                space_length = 0
            if piece != 0:
                sfen += board_char_dir[piece]
        if space_length != 0:
            sfen += str(space_length)
        if i != 8:
            sfen += "/"
        else:
            sfen += " "
    # 手番
    if state.turn == 0:
        sfen += "b "
    else:
        sfen += "w "
    # 持ち駒
    if jnp.all(state.hand == 0):
        sfen += "-"
    else:
        for i in range(2):
            for j in range(7):
                piece_type = hand_dir[i * 7 + j]
                num_piece = state.hand.flatten()[piece_type]
                if num_piece == 0:
                    continue
                if num_piece >= 2:
                    sfen += str(num_piece)
                sfen += hand_char_dir[piece_type]
    sfen += f" {state._step_count + 1}"
    return sfen


def _from_sfen(sfen):
    # fmt: off
    board_char_dir = ["P", "L", "N", "S", "B", "R", "G", "K", "", "", "", "", "", "", "p", "l", "n", "s", "b", "r", "g", "k"]
    hand_char_dir = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
    # fmt: on
    board, turn, hand, step_count = sfen.split()
    board_ranks = board.split("/")
    piece_board = jnp.zeros(81, dtype=jnp.int8)
    for i in range(9):
        file = board_ranks[i]
        rank = []
        piece = 0
        for char in file:
            if char.isdigit():
                num_space = int(char)
                for j in range(num_space):
                    rank.append(-1)
            elif char == "+":
                piece += 8
            else:
                piece += board_char_dir.index(char)
                rank.append(piece)
                piece = 0
        for j in range(9):
            piece_board = piece_board.at[9 * i + j].set(rank[j])
    s_hand = jnp.zeros(14, dtype=jnp.int8)
    if hand != "-":
        num_piece = 1
        for char in hand:
            if char.isdigit():
                num_piece = int(char)
            else:
                s_hand = s_hand.at[hand_char_dir.index(char)].set(num_piece)
                num_piece = 1
    piece_board = jnp.rot90(piece_board.reshape((9, 9)), k=1).flatten()
    hand = jnp.reshape(s_hand, (2, 7))
    turn = jnp.int8(0) if turn == "b" else jnp.int8(1)
    return turn ,piece_board, hand, int(step_count) - 1
