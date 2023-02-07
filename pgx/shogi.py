import jax
import jax.numpy as jnp
from flax.struct import dataclass

#   0 空白
#   1 歩
#   2 香車
#   3 桂馬
#   4 銀
#   5 角
#   6 飛車
#   7 金
#   8 玉
#   9 と
#  10 成香
#  11 成桂
#  12 成銀
#  13 馬
#  14 龍
#  15 相手歩
#  16 相手香車
#  17 相手桂馬
#  18 相手銀
#  19 相手角
#  20 相手飛車
#  21 相手金
#  22 相手玉
#  23 相手と
#  24 相手成香
#  25 相手成桂
#  26 相手成銀
#  27 相手馬
#  28 相手龍


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# fmt: off
INIT_PIECE_BOARD = jnp.int8([[15, -1, 14, -1, -1, -1,  0, -1,  1],
                             [16, 18, 14, -1, -1, -1,  0,  5,  2],
                             [17, -1, 14, -1, -1, -1,  0, -1,  3],
                             [20, -1, 14, -1, -1, -1,  0, -1,  6],
                             [21, -1, 14, -1, -1, -1,  0, -1,  7],
                             [20, -1, 14, -1, -1, -1,  0, -1,  6],
                             [17, -1, 14, -1, -1, -1,  0, -1,  3],
                             [16, 19, 14, -1, -1, -1,  0,  4,  2],
                             [15, -1, 14, -1, -1, -1,  0, -1,  1]]).flatten()
# fmt: on


@dataclass
class State:
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,) 後手のときにはflipする
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)  # 後手のときにはflipする


def init():
    """Initialize Shogi State.
    >>> s = init()
    >>> s.piece_board.reshape((9, 9))
    Array([[15, -1, 14, -1, -1, -1,  0, -1,  1],
           [16, 18, 14, -1, -1, -1,  0,  5,  2],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [21, -1, 14, -1, -1, -1,  0, -1,  7],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [16, 19, 14, -1, -1, -1,  0,  4,  2],
           [15, -1, 14, -1, -1, -1,  0, -1,  1]], dtype=int8)
    >>> jnp.rot90(s.piece_board.reshape((9, 9)), k=3)
    Array([[15, 16, 17, 20, 21, 20, 17, 16, 15],
           [-1, 19, -1, -1, -1, -1, -1, 18, -1],
           [14, 14, 14, 14, 14, 14, 14, 14, 14],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [-1,  4, -1, -1, -1, -1, -1,  5, -1],
           [ 1,  2,  3,  6,  7,  6,  3,  2,  1]], dtype=int8)
    """
    return State()


# 指し手のdataclass
@dataclass
class Action:
    """
    direction (from github.com/TadaoYamaoka/cshogi)

     0 Up
     1 Up left
     2 Up right
     3 Left
     4 Right
     5 Down
     6 Down left
     7 Down right
     8 Up2 left
     9 Up2 right
    10 Promote +  Up
    11 Promote +  Up left
    12 Promote +  Up right
    13 Promote +  Left
    14 Promote +  Right
    15 Promote +  Down
    16 Promote +  Down left
    17 Promote +  Down right
    18 Promote +  Up2 left
    19 Promote +  Up2 right
    20 Drop 歩
    21 Drop 香車
    22 Drop 桂馬
    23 Drop 銀
    24 Drop 角
    25 Drop 飛車
    26 Drop 金

    piece:
     0 歩
     1 香車
     2 桂馬
     3 銀
     4 角
     5 飛車
     6 金
     7 玉
     8 と
     9 成香
    10 成桂
    11 成銀
    12 馬
    13 龍
    """

    # 駒打ちかどうか
    is_drop: jnp.ndarray
    # piece: 動かした(打った)駒の種類
    piece: jnp.ndarray
    # 移動後の座標
    to: jnp.ndarray
    # 移動前の座標 (zero if drop action)
    from_: jnp.ndarray
    # captured: 取られた駒の種類 (false if drop action)
    is_capture: jnp.ndarray
    # is_promote: 駒を成るかどうかの判定 (false if drop action)
    is_promotion: jnp.ndarray

    @classmethod
    def from_dlshogi_action(cls, state: State, action: jnp.ndarray):
        direction, to = action // 81, action % 81
        is_drop = direction >= 20
        from_ = ...  # TODO: write me
        piece = jax.lax.cond(
            is_drop,
            lambda: direction - 20,
            lambda: state.piece_board[from_],
        )
        is_capture = state.piece_board[to] != 0
        is_promotion = (10 <= direction) & (direction < 20)
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_capture=is_capture, is_promtotion=is_promotion)  # type: ignore

    def to_dlshogi_action(self) -> jnp.ndarray:
        direction = jax.lax.cond(self.is_drop, lambda: ...)
        return 81 * direction + self.to


def to_sfen(state: State):
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

    >>> s = init()
    >>> to_sfen(s)
    'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'
    """

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
