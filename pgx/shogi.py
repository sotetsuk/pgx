import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

#   0 空白
#   1 先手歩
#   2 先手香車
#   3 先手桂馬
#   4 先手銀
#   5 先手角
#   6 先手飛車
#   7 先手金
#   8 先手玉
#   9 先手と
#  10 先手成香
#  11 先手成桂
#  12 先手成銀
#  13 先手馬
#  14 先手龍
#  15 後手歩
#  16 後手香車
#  17 後手桂馬
#  18 後手銀
#  19 後手角
#  20 後手飛車
#  21 後手金
#  22 後手玉
#  23 後手と
#  24 後手成香
#  25 後手成桂
#  26 後手成銀
#  27 後手馬
#  28 後手龍


# fmt: off
INIT_PIECE_BOARD = jnp.int8([16, 0, 15, 0, 0, 0, 1, 0, 2, 17, 19, 15, 0, 0, 0, 1, 6, 3, 18, 0, 15, 0, 0, 0, 1, 0, 4, 21, 0, 15, 0, 0, 0, 1, 0, 7, 22, 0, 15, 0, 0, 0, 1, 0, 8, 21, 0, 15, 0, 0, 0, 1, 0,  7, 18, 0, 15, 0, 0, 0, 1, 0, 4, 17, 20, 15, 0, 0, 0, 1, 5, 3, 16, 0, 15, 0, 0, 0, 1, 0, 2])
# fmt: on


@dataclass
class State:
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,)
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)


def init():
    """Initialize Shogi State.
    >>> s = init()
    >>> s.piece_board.reshape((9, 9))
    Array([[16,  0, 15,  0,  0,  0,  1,  0,  2],
           [17, 19, 15,  0,  0,  0,  1,  6,  3],
           [18,  0, 15,  0,  0,  0,  1,  0,  4],
           [22,  0, 15,  0,  0,  0,  1,  0,  7],
           [23,  0, 15,  0,  0,  0,  1,  0,  8],
           [22,  0, 15,  0,  0,  0,  1,  0,  7],
           [18,  0, 15,  0,  0,  0,  1,  0,  4],
           [17, 20, 15,  0,  0,  0,  1,  5,  3],
           [16,  0, 15,  0,  0,  0,  1,  0,  2]], dtype=int8)
    >>> jnp.rot90(s.piece_board.reshape((9, 9)), k=3)
    Array([[16, 17, 18, 22, 23, 22, 18, 17, 16],
           [ 0, 20,  0,  0,  0,  0,  0, 19,  0],
           [15, 15, 15, 15, 15, 15, 15, 15, 15],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
           [ 0,  5,  0,  0,  0,  0,  0,  6,  0],
           [ 2,  3,  4,  7,  8,  7,  4,  3,  2]], dtype=int8)
    """
    return State()


def _to_sfen(state: State):
    # 歩:P 香車:L 桂馬:N 銀:S 角:B 飛車:R 金:G 王:K
    # 成駒なら駒の前に+をつける（と金なら+P）
    # 先手の駒は大文字、後手の駒は小文字で表現
    # 空白の場合、連続する空白の数を入れて次の駒にシフトする。歩空空空飛ならP3R
    # 左上から開始して右に見ていく。段が変わるときは/を挿入
    # 盤面の記入が終わったら手番の記入。b または w
    # 持ち駒は先手の物から記入。順番はRBGSNLPの順
    # 最後に手数。ただここは1でいいと思う
    pb = jnp.rot90(state.piece_board.reshape((9, 9)), k=3)
    sfen = ""
    board_char_dir = np.array(
        [
            "",
            "P",
            "L",
            "N",
            "S",
            "B",
            "R",
            "G",
            "K",
            "+P",
            "+L",
            "+N",
            "+S",
            "+B",
            "+R",
            "p",
            "l",
            "n",
            "s",
            "b",
            "r",
            "g",
            "k",
            "+p",
            "+l",
            "+n",
            "+s",
            "+b",
            "+r",
        ],
        dtype=str,
    )
    hand_char_dir = np.array(
        ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"],
        dtype=str,
    )
    hand_dir = np.array([5, 4, 6, 3, 2, 1, 0, 12, 11, 13, 10, 9, 8, 7])
    # 盤面
    for i in range(9):
        space_length = 0
        for j in range(9):
            piece = pb[i][j]
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
        sfen += "- 1"
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
        sfen += " 1"
    return sfen
