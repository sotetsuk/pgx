"""Shogi.

piece_board (81,):
  -1 空白
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
  14 相手歩
  15 相手香車
  16 相手桂馬
  17 相手銀
  18 相手角
  19 相手飛車
  20 相手金
  21 相手玉
  22 相手と
  23 相手成香
  24 相手成桂
  25 相手成銀
  26 相手馬
  27 相手龍
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from pgx.cache import load_shogi_is_on_the_way, load_shogi_raw_effect_boards

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pieces
EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
LANCE = jnp.int8(1)
KNIGHT = jnp.int8(2)
SILVER = jnp.int8(3)
BISHOP = jnp.int8(4)
ROOK = jnp.int8(5)
GOLD = jnp.int8(6)
KING = jnp.int8(7)
PRO_PAWN = jnp.int8(8)
PRO_LANCE = jnp.int8(9)
PRO_KNIGHT = jnp.int8(10)
PRO_SILVER = jnp.int8(11)
HORSE = jnp.int8(12)
DRAGON = jnp.int8(13)
OPP_PAWN = jnp.int8(14)
OPP_LANCE = jnp.int8(15)
OPP_KNIGHT = jnp.int8(16)
OPP_SILVER = jnp.int8(17)
OPP_BISHOP = jnp.int8(18)
OPP_ROOK = jnp.int8(19)
OPP_GOLD = jnp.int8(20)
OPP_KING = jnp.int8(21)
OPP_PRO_PAWN = jnp.int8(22)
OPP_PRO_LANCE = jnp.int8(23)
OPP_PRO_KNIGHT = jnp.int8(24)
OPP_PRO_SILVER = jnp.int8(25)
OPP_HORSE = jnp.int8(26)
OPP_DRAGON = jnp.int8(27)


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
RAW_EFFECT_BOARDS = load_shogi_raw_effect_boards()  # bool (14, 81, 81)
# When <lance/bishop/rook/horse/dragon,5> moves from <from,81> to <to,81>,
# is <point,81> on the way between two points?
IS_ON_THE_WAY = load_shogi_is_on_the_way()  # bool (5, 81, 81, 81)


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
    """

    # 駒打ちかどうか
    is_drop: jnp.ndarray
    # 動かした(打った)駒の種類/打つ前が成っていなければ成ってない
    piece: jnp.ndarray
    # 移動後の座標
    to: jnp.ndarray
    # --- Optional (only for move action) ---
    # 移動前の座標
    from_: jnp.ndarray = jnp.int8(0)
    # 駒を成るかどうかの判定
    is_promotion: jnp.ndarray = FALSE

    @classmethod
    def make_move(cls, piece, from_, to, is_promotion=FALSE):
        return Action(
            is_drop=False,
            piece=piece,
            from_=from_,
            to=to,
            is_promotion=is_promotion,
        )

    @classmethod
    def make_drop(cls, piece, to):
        return Action(is_drop=True, piece=piece, to=to)

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
        is_promotion = (10 <= direction) & (direction < 20)
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promtotion=is_promotion)  # type: ignore

    def to_dlshogi_action(self) -> jnp.ndarray:
        direction = jax.lax.cond(self.is_drop, lambda: ...)
        return 81 * direction + self.to


def _step(state: State, action: Action) -> State:
    # apply move/drop action
    state = jax.lax.cond(
        action.is_drop, _step_drop, _step_move, *(state, action)
    )
    # flip state
    state = _flip(state)
    state = state.replace(turn=(state.turn + 1) % 2)  # type: ignore
    return state


def _step_move(state: State, action: Action) -> State:
    pb = state.piece_board
    # remove piece from the original position
    pb = pb.at[action.from_].set(EMPTY)
    # capture the opponent if exists
    captured = pb[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state.hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by (x + 14) % 28, and
        #   (2) filtering promoted piece by x % 8
        lambda: state.hand.at[0, ((captured + 14) % 28) % 8].add(1),
    )
    # promote piece
    piece = jax.lax.cond(
        action.is_promotion,
        lambda: _promote(action.piece),
        lambda: action.piece,
    )
    # set piece to the target position
    pb = pb.at[action.to].set(piece)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    pb = state.piece_board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _flip(state: State):
    empty_mask = state.piece_board == EMPTY
    pb = (state.piece_board + 14) % 28
    pb = jnp.where(empty_mask, EMPTY, pb)
    pb = pb[::-1]
    return state.replace(  # type: ignore
        piece_board=pb, hand=state.hand[jnp.int8((1, 0))]
    )


def _promote(piece: jnp.ndarray) -> jnp.ndarray:
    return piece + 8


def _apply_raw_effects(state: State) -> jnp.ndarray:
    """Obtain raw effect boards from piece board by batch.

    >>> s = init()
    >>> jnp.rot90(_apply_raw_effects(s).any(axis=0).reshape(9, 9), k=3)
    Array([[ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False,  True,  True,  True],
           [ True, False, False, False, False,  True, False,  True,  True],
           [ True, False, False, False,  True, False, False,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True,  True],
           [ True, False,  True, False, False, False,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True,  True],
           [ True, False,  True,  True,  True,  True,  True,  True, False]],      dtype=bool)
    """
    from_ = jnp.arange(81)
    pieces = state.piece_board  # include -1

    # fix to and apply (pieces, from_) by batch
    @jax.vmap
    def _raw_effect_boards(p, f):
        return RAW_EFFECT_BOARDS[p, f, :]  # (81,)

    mask = ((0 <= pieces) & (pieces < 14)).reshape(81, 1)
    raw_effect_boards = _raw_effect_boards(pieces, from_)
    raw_effect_boards = jnp.where(mask, raw_effect_boards, FALSE)
    return raw_effect_boards  # (81, 81)


def _obtain_effect_filter(state: State) -> jnp.ndarray:
    """
    >>> s = init()
    >>> jnp.rot90(_obtain_effect_filter(s).any(axis=0).reshape(9, 9), k=3)
    Array([[ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False,  True,  True,  True],
           [ True, False, False, False, False,  True, False,  True,  True],
           [ True, False, False, False,  True, False, False,  True,  True],
           [ True, False, False,  True, False, False, False,  True,  True],
           [False, False, False, False, False, False, False, False, False],
           [ True, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)

    """

    def _is_brocked(p, f, t):
        # (piece, from, to) を固定したとき、pieceがfromからtoへ妨害されずに到達できるか否か
        # True = 途中でbrockされ、到達できない
        # TODO: filtering by jnp.where at last might be faster?
        return jax.lax.cond(
            p >= 0,
            lambda: (
                IS_ON_THE_WAY[p, f, t, :] & (state.piece_board >= 0)
            ).any(),
            lambda: FALSE,
        )

    def _is_brocked2(p, f):
        to = jnp.arange(81)
        return jax.vmap(partial(_is_brocked, p=p, f=f))(t=to)

    def _reduce_ix(piece):
        # Filtering only Lance, Bishop, Rook, Horse, and Dragon
        # NOTE: last 14th -1 is sentinel for avoid accessing via -1
        return jnp.int8(
            [-1, 0, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, 3, 4, -1]
        )[piece]

    pieces = state.piece_board
    reduced_pieces = jax.vmap(_reduce_ix)(pieces)
    from_ = jnp.arange(81)

    filter_boards = jax.vmap(_is_brocked2)(reduced_pieces, from_)

    return filter_boards  # (81=from, 81=to)


def _apply_effects(state: State):
    """
    >>> s = init()
    >>> jnp.rot90(_apply_effects(s)[8].reshape(9, 9), k=3)  # 香
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False,  True],
           [False, False, False, False, False, False, False, False,  True],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)
    >>> jnp.rot90(_apply_effects(s)[70].reshape(9, 9), k=3)  # 角
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [ True, False,  True, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [ True, False,  True, False, False, False, False, False, False]],      dtype=bool)
    >>> jnp.rot90(_apply_effects(s)[16].reshape(9, 9), k=3)  # 飛
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False,  True, False],
           [False,  True,  True,  True,  True,  True,  True, False,  True],
           [False, False, False, False, False, False, False,  True, False]],      dtype=bool)
    """
    raw_effect_boards = _apply_raw_effects(state)
    effect_filter_boards = _obtain_effect_filter(state)
    return raw_effect_boards & ~effect_filter_boards


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
