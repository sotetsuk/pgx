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
INIT_PIECE_BOARD = jnp.int8([16, 0, 15, 0, 0, 0, 1, 0, 2, 17, 19, 15, 0, 0, 0, 1, 6, 3, 18, 0, 15, 0, 0, 0, 1, 0, 4, 22, 0, 15, 0, 0, 0, 1, 0, 7, 23, 0, 15, 0, 0, 0, 1, 0, 8, 22, 0, 15, 0, 0, 0, 1, 0, 7, 18, 0, 15, 0, 0, 0, 1, 0, 4, 17, 20, 15, 0, 0, 0, 1, 5, 3, 16, 0, 15, 0, 0, 0, 1, 0, 2])
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
        direction = jax.lax.cond(
            self.is_drop,
            lambda: self.piece + 20
        )
        return 81 * self.
