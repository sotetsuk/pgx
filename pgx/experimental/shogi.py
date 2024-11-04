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

import numpy as np
import jax
from pgx._src.games.shogi import _flip, Game, GameState
from pgx.shogi import State


def to_sfen(state):
    """Convert state into sfen expression.

    - Board
        - 歩:P 香車:L 桂馬:N 銀:S 角:B 飛車:R 金:G 王:K
        - + before promoted piece（と金 = +P）
        - The first player's pieces are capitalized
        - 空白の場合、連続する空白の数を入れて次の駒にシフトする。歩空空空飛ならP3R
        - If the square is empty, the number of consecutive spaces is entered and shifted to the next piece (e.g., P3R for P _ _ _ R)
        - From the upper left corner to the right
        - When the row changes, insert /
    - Turn (b/w) after board
    - Hand piece (prisoners) are in order of RBGSNLP
    - Step count (fixed to 1 here)

    """
    # NOTE: input must be flipped if white turn
    state = state if state._x.color % 2 == 0 else state.replace(_x=_flip(state._x))  # type: ignore

    pb = np.rot90(state._x.board.reshape((9, 9)), k=3)
    sfen = ""
    # fmt: off
    board_char_dir = ["", "P", "L", "N", "S", "B", "R", "G", "K", "+P", "+L", "+N", "+S", "+B", "+R", "p", "l", "n", "s", "b", "r", "g", "k", "+p", "+l", "+n", "+s", "+b", "+r"]
    hand_char_dir = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
    hand_dir = [5, 4, 6, 3, 2, 1, 0, 12, 11, 13, 10, 9, 8, 7]
    # fmt: on
    # Board
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
    # Turn
    if state._x.color == 0:
        sfen += "b "
    else:
        sfen += "w "
    # Hand (prisoners)
    if np.all(state._x.hand == 0):
        sfen += "-"
    else:
        for i in range(2):
            for j in range(7):
                piece_type = hand_dir[i * 7 + j]
                num_piece = state._x.hand.flatten()[piece_type]
                if num_piece == 0:
                    continue
                if num_piece >= 2:
                    sfen += str(num_piece)
                sfen += hand_char_dir[piece_type]
    sfen += f" {state._step_count + 1}"
    return sfen


@jax.jit
def _from_board(color, piece_board, hand):
    """Mainly for debugging purpose.
    terminated, reward, and current_player are not changed"""
    state = State(_x=GameState(color=color, board=piece_board, hand=hand))  # type: ignore
    # fmt: off
    state = jax.lax.cond(color % 2 == 1, lambda: state.replace(_x=_flip(state._x)), lambda: state)  # type: ignore
    # fmt: on
    return state.replace(legal_action_mask=Game().legal_action_mask(state._x))  # type: ignore


def from_sfen(sfen):
    # fmt: off
    board_char_dir = ["P", "L", "N", "S", "B", "R", "G", "K", "", "", "", "", "", "", "p", "l", "n", "s", "b", "r", "g", "k"]
    hand_char_dir = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
    # fmt: on
    board, color, hand, step_count = sfen.split()
    board_ranks = board.split("/")
    piece_board = np.zeros(81, dtype=np.int32)
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
            piece_board[9 * i + j] = rank[j]
    s_hand = np.zeros(14, dtype=np.int32)
    if hand != "-":
        num_piece = 1
        for char in hand:
            if char.isdigit():
                num_piece = int(char)
            else:
                s_hand[hand_char_dir.index(char)] = num_piece
                num_piece = 1
    piece_board = np.rot90(piece_board.reshape((9, 9)), k=1).flatten()
    hand = np.reshape(s_hand, (2, 7))
    color = 0 if color == "b" else 1
    color, piece_board, hand, step_count = color, piece_board, hand, int(step_count) - 1
    state = _from_board(color, piece_board, hand)
    state = state.replace(  # type: ignore
        _step_count=np.int32(step_count),
        _x=state._x._replace(step_count=np.int32(step_count)),
    )
    return state
