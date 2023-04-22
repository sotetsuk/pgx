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

import os

import jax
import jax.numpy as jnp
import numpy as np

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
file_path = "assets/can_move.npy"
with open(os.path.join(os.path.dirname(__file__), file_path), "rb") as f:
    CAN_MOVE = jnp.load(f)
assert CAN_MOVE.sum() == 8228


# When <lance/bishop/rook/horse/dragon,5> moves from <from,81> to <to,81>,
# is <point,81> on the way between two points?
file_path = "assets/between.npy"
with open(os.path.join(os.path.dirname(__file__), file_path), "rb") as f:
    BETWEEN = jnp.load(f)
assert BETWEEN.sum() == 10564

# Give <dir,10> and <to,81>, return the legal <from> idx
# E.g. LEGAL_FROM_IDX[Up, to=19] = [20, 21, ..., -1] (filled by -1)
# Used for computing dlshogi action
#
#  dir, to, from
#  (10, 81, 81)
#
#  0 Up
#  1 Up left
#  2 Up right
#  3 Left
#  4 Right
#  5 Down
#  6 Down left
#  7 Down right
#  8 Up2 left
#  9 Up2 right

LEGAL_FROM_IDX = -np.ones((10, 81, 8), dtype=jnp.int32)  # type: ignore

for dir_ in range(10):
    for to in range(81):
        x, y = to // 9, to % 9
        if dir_ == 0:  # Up
            dx, dy = 0, +1
        elif dir_ == 1:  # Up left
            dx, dy = -1, +1
        elif dir_ == 2:  # Up right
            dx, dy = +1, +1
        elif dir_ == 3:  # Left
            dx, dy = -1, 0
        elif dir_ == 4:  # Right
            dx, dy = +1, 0
        elif dir_ == 5:  # Down
            dx, dy = 0, -1
        elif dir_ == 6:  # Down left
            dx, dy = -1, -1
        elif dir_ == 7:  # Down right
            dx, dy = +1, -1
        elif dir_ == 8:  # Up2 left
            dx, dy = -1, +2
        elif dir_ == 9:  # Up2 right
            dx, dy = +1, +2
        for i in range(8):
            x += dx
            y += dy
            if x < 0 or 8 < x or y < 0 or 8 < y:
                break
            LEGAL_FROM_IDX[dir_, to, i] = x * 9 + y
            if dir_ == 8 or dir_ == 9:
                break

LEGAL_FROM_IDX = jnp.array(LEGAL_FROM_IDX)  # type: ignore


@jax.jit
@jax.vmap
def can_move_any_ix(from_):
    return jnp.nonzero(
        (CAN_MOVE[:, from_, :] | CAN_MOVE[:, :, from_]).any(axis=0),
        size=36,
        fill_value=-1,
    )[0]


@jax.jit
@jax.vmap
def neighbour_ix(from_):
    return jnp.nonzero(
        (CAN_MOVE[7, from_, :] | CAN_MOVE[2, :, from_]),
        size=10,
        fill_value=-1,
    )[0]


NEIGHBOUR_IX = neighbour_ix(jnp.arange(81))


def between_ix(p, from_, to):
    return jnp.nonzero(BETWEEN[p, from_, to], size=8, fill_value=-1)[0]


BETWEEN_IX = jax.jit(
    jax.vmap(
        jax.vmap(jax.vmap(between_ix, (None, None, 0)), (None, 0, None)),
        (0, None, None),
    )
)(jnp.arange(5), jnp.arange(81), jnp.arange(81))


CAN_MOVE_ANY = can_move_any_ix(jnp.arange(81))  # (81, 36)

INIT_LEGAL_ACTION_MASK = jnp.zeros(81 * 27, dtype=jnp.bool_)
# fmt: off
ixs = [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331]
# fmt: on
for ix in ixs:
    INIT_LEGAL_ACTION_MASK = INIT_LEGAL_ACTION_MASK.at[ix].set(True)
assert INIT_LEGAL_ACTION_MASK.shape == (81 * 27,)
assert INIT_LEGAL_ACTION_MASK.sum() == 30


def _around(c):
    x, y = c // 9, c % 9
    dx = jnp.int8([-1, -1, 0, +1, +1, +1, 0, -1])
    dy = jnp.int8([0, -1, -1, -1, 0, +1, +1, +1])

    def f(i):
        new_x, new_y = x + dx[i], y + dy[i]
        return jax.lax.select(
            (new_x < 0) | (new_x >= 9) | (new_y < 0) | (new_y >= 9),
            -1,
            new_x * 9 + new_y,
        )

    return jax.vmap(f)(jnp.arange(8))


AROUND_IX = jax.vmap(_around)(jnp.arange(81))


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
    # NOTE: input must be flipped if white turn

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
    return turn, piece_board, hand, int(step_count) - 1
