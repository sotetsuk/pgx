import jax.numpy as jnp
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


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# fmt: off
INIT_PIECE_BOARD = jnp.int8([16, 0, 15, 0, 0, 0, 1, 0, 2, 17, 19, 15, 0, 0, 0, 1, 6, 3, 18, 0, 15, 0, 0, 0, 1, 0, 4, 22, 0, 15, 0, 0, 0, 1, 0, 7, 23, 0, 15, 0, 0, 0, 1, 0, 8, 22, 0, 15, 0, 0, 0, 1, 0, 7, 18, 0, 15, 0, 0, 0, 1, 0, 4, 17, 20, 15, 0, 0, 0, 1, 5, 3, 16, 0, 15, 0, 0, 0, 1, 0, 2])
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


# 指し手のdataclass
@dataclass
class Action:
    """
    direction (from github.com/TadaoYamaoka/cshogi)

     0 UP
     1 UP_LEFT
     2 UP_RIGHT
     3 LEFT
     4 RIGHT
     5 DOWN
     6 DOWN_LEFT
     7 DOWN_RIGHT
     8 UP2_LEFT
     9 UP2_RIGHT
    10 PROMOTE + UP
    11 PROMOTE + UP_LEFT
    12 PROMOTE + UP_RIGHT
    13 PROMOTE + LEFT
    14 PROMOTE + RIGHT
    15 PROMOTE + DOWN
    16 PROMOTE + DOWN_LEFT
    17 PROMOTE + DOWN_RIGHT
    18 PROMOTE + UP2_LEFT
    19 PROMOTE + UP2_RIGHT
    20 PROMOTE + UP_PROMOTE
    ---
    drop:
    21 歩
    22 香車
    23 桂馬
    24 銀
    25 角
    26 飛車
    27 金

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
    # ---- Optional (only for moves) ---
    # 移動前の座標
    from_: jnp.ndarray = jnp.int8(0)
    # captured: 取られた駒の種類。駒が取られていない場合は0
    is_capture: jnp.ndarray = FALSE
    # is_promote: 駒を成るかどうかの判定
    is_promotion: jnp.ndarray = FALSE

    @classmethod
    def from_dlaction(cls, state: State, action: jnp.ndarray):
        direction, to = action // 81, action % 81
        from_ = ...  # TODO: write me
        piece = ...  # TODO: write me
        is_drop = direction > 20
        is_capture = state.piece_board[to] != 0
        is_promtotion = (10 <= direction) & (direction < 21)
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_capture=is_capture, is_promtotion=is_promtotion)  # type: ignore

    def to_dlaction(self) -> jnp.ndarray:
        ...
