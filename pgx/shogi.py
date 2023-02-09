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
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from pgx.cache import load_shogi_is_on_the_way, load_shogi_raw_effect_boards

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)
ZERO = jnp.int8(0)
ONE = jnp.int8(1)
TWO = jnp.int8(2)

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


def _to_large_piece_ix(piece):
    # Filtering only Lance(0), Bishop(1), Rook(2), Horse(3), and Dragon(4)
    # NOTE: last 14th -1 is sentinel for avoid accessing via -1
    return jnp.int8([-1, 0, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, 3, 4, -1])[
        piece
    ]


def _apply_effect_filter(state: State) -> jnp.ndarray:
    """
    >>> s = init()
    >>> jnp.rot90(_apply_effect_filter(s).any(axis=0).reshape(9, 9), k=3)
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
    pieces = state.piece_board
    large_pieces = jax.vmap(_to_large_piece_ix)(pieces)
    from_ = jnp.arange(81)
    to = jnp.arange(81)

    def func1(p, f, t):
        # (piece, from, to) を固定したとき、pieceがfromからtoへ妨害されずに到達できるか否か
        # True = 途中でbrockされ、到達できない
        return (IS_ON_THE_WAY[p, f, t, :] & (state.piece_board >= 0)).any()

    def func2(p, f):
        # (piece, from) を固定してtoにバッチで適用
        return jax.vmap(partial(func1, p=p, f=f))(t=to)

    # (piece, from) にバッチで適用
    filter_boards = jax.vmap(func2)(large_pieces, from_)  # (81,81)

    mask = (large_pieces >= 0).reshape(81, 1)
    filter_boards = jnp.where(mask, filter_boards, FALSE)

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
    effect_filter_boards = _apply_effect_filter(state)
    return raw_effect_boards & ~effect_filter_boards


def _legal_moves(
    state: State, effect_boards: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Filter (84, 84) effects and return legal moves (84, 84) and promotion (84, 84)

    >>> s = init()
    >>> effect_boards = _apply_effects(s)
    >>> legal_moves, _ = _legal_moves(s, effect_boards)
    >>> jnp.rot90(legal_moves[8].reshape(9, 9), k=3)  # 香
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False,  True],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)
    """
    pb = state.piece_board

    # filter the destinations where my piece exists
    is_my_piece = (PAWN <= pb) & (pb < OPP_PAWN)
    effect_boards = jnp.where(is_my_piece, FALSE, effect_boards)

    # Filter suicide action
    #   - King moves into the effected area
    #   - Pinned piece moves
    #  A piece is pinned when
    #   - it exists between king and (Lance/Bishop/Rook/Horse/Dragon)
    #   - no other pieces exist on the way to king
    opp_effect_board = jnp.flip(_apply_effects(_flip(state))).any(axis=0)  # (81,)
    king_mask = pb == KING
    mask = king_mask.reshape(81, 1) * opp_effect_board.reshape(1, 81)
    effect_boards = jnp.where(mask, FALSE, effect_boards)

    # TODO: 王手放置

    # promotion (80, 80)
    #   0 = cannot promote
    #   1 = can promote (from or to opp area)
    #   2 = have to promote (get stuck)
    promotion = effect_boards.astype(jnp.int8)
    # mask where piece cannot promote
    in_opp_area = jnp.arange(81) % 9 < 3
    tgt_in_opp_area = jnp.tile(in_opp_area, reps=(81, 1))
    src_in_opp_area = tgt_in_opp_area.transpose()
    mask = src_in_opp_area | tgt_in_opp_area
    promotion = jnp.where(mask, promotion, ZERO)
    # mask where piece have to promote
    is_line1 = jnp.tile(jnp.arange(81) % 9 == 0, reps=(81, 1))
    is_line2 = jnp.tile(jnp.arange(81) % 9 == 1, reps=(81, 1))
    where_pawn_or_lance = (pb == PAWN) | (pb == LANCE)
    where_knight = pb == KNIGHT
    is_stuck = jnp.tile(where_pawn_or_lance, (81, 1)).transpose() & is_line1
    is_stuck |= jnp.tile(where_knight, (81, 1)).transpose() & (
        is_line1 | is_line2
    )
    promotion = jnp.where((promotion != 0) & is_stuck, TWO, promotion)

    return effect_boards, promotion


def _legal_drops(state: State, effect_boards: jnp.ndarray) -> jnp.ndarray:
    """Return (7, 81) boolean array

    >>> s = init()
    >>> s = s.replace(piece_board=s.piece_board.at[15].set(EMPTY))
    >>> s = s.replace(hand=s.hand.at[0].set(1))
    >>> effect_boards = _apply_effects(s)
    >>> _rotate(_legal_drops(s, effect_boards)[PAWN])
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)
    """
    legal_drops = jnp.zeros((7, 81), dtype=jnp.bool_)

    # is the piece in my hand?
    is_in_my_hand = (state.hand[0] > 0).reshape(7, 1)
    legal_drops = jnp.where(is_in_my_hand, TRUE, legal_drops)

    # piece exists
    is_not_empty = state.piece_board != EMPTY
    legal_drops = jnp.where(is_not_empty, FALSE, legal_drops)

    # get stuck
    is_line1 = jnp.arange(81) % 9 == 0
    is_line2 = jnp.arange(81) % 9 == 1
    is_pawn_or_lance = jnp.arange(7) <= LANCE
    is_knight = jnp.arange(7) == KNIGHT
    mask_pawn_lance = is_pawn_or_lance.reshape(7, 1) * is_line1.reshape(1, 81)
    mask_knight = is_knight.reshape(7, 1) * (is_line1 | is_line2).reshape(
        1, 81
    )
    legal_drops = jnp.where(mask_pawn_lance, FALSE, legal_drops)
    legal_drops = jnp.where(mask_knight, FALSE, legal_drops)

    # double pawn
    has_pawn = (state.piece_board == PAWN).reshape(9, 9).any(axis=1)
    has_pawn = jnp.tile(has_pawn, reps=(9, 1)).transpose().flatten()
    legal_drops = jnp.where(has_pawn, FALSE, legal_drops)

    # 打ち歩詰
    #  避け方は次の3通り
    #  - (1) 頭の歩を王で取る
    #  - (2) 王が逃げる
    #  - (3) 頭の歩を王以外の駒で取る
    #  (1)と(2)はようするに、今王が逃げられるところがあるか（利きがないか）ということでまとめて処理できる
    pb = state.piece_board
    opp_king_pos = jnp.nonzero(pb == OPP_KING, size=1)[0].item()
    opp_king_head_pos = (
        opp_king_pos + 1
    )  # NOTE: 王が一番下の段にいるとき間違っているが、その場合は使われないので問題ない
    can_check_by_pawn_drop = opp_king_pos % 9 != 8

    # 王が利きも味方の駒もないところへ逃げられるか
    king_escape_mask = RAW_EFFECT_BOARDS[KING, opp_king_pos, :]  # (81,)
    king_escape_mask &= ~(
        (OPP_PAWN <= pb) & (pb <= OPP_DRAGON)
    )  # 味方駒があり、逃げられない
    king_escape_mask &= ~effect_boards.any(axis=0)  # 利きがあり逃げられない
    can_king_escape = king_escape_mask.any()

    # 反転したボードで処理していることに注意
    flipped_opp_effects = _apply_effects(_flip(state))
    flipped_opp_king_head_pos = 80 - opp_king_head_pos
    can_capture_pawn = (
        flipped_opp_effects[:, flipped_opp_king_head_pos].sum() > 1
    )  # 自分以外の利きがないといけない

    legal_drops = jax.lax.cond(
        (can_check_by_pawn_drop & (~can_king_escape) & (~can_capture_pawn)),
        lambda: legal_drops.at[PAWN, opp_king_head_pos].set(FALSE),
        lambda: legal_drops,
    )

    # TODO: 王手放置

    return legal_drops


def _rotate(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.rot90(board.reshape(9, 9), k=3)


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
