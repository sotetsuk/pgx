import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# 指し手のdataclass
@dataclass
class ChessAction:
    # piece: 動かした駒の種類
    piece: int
    # 移動前の座標
    from_: int
    # to: 移動後の座標
    to: int
    # promotion: プロモーションで成る駒を選ぶ。デフォルトはクイーン
    # 0: Queen, 1: Rook, 2: Knight, 3:Bishop
    promotion: int = 0


# 盤面のdataclass
@dataclass
class ChessState:
    # turn 先手番なら0 後手番なら1
    turn: int = 0
    # board 盤面の駒。
    # 空白,WPawn,WKnight,WBishop,WRook,WQueen,WKing,BPawn,BKnight,BBishop,BRook,BQueen,BKing
    # の順で駒がどの位置にあるかをone_hotで記録
    board: np.ndarray = np.zeros((13, 64), dtype=np.int32)
    # en_passant 直前にポーンが2マス動いていた場合、通過した地点でもそのポーンは取れる
    # 直前の動きがポーンを2マス進めるものだった場合は、位置を記録
    # 普段は-1
    en_passant: int = -1
    # move_count ルークやキングがすでに動いたかどうか
    # キャスリングの判定に使用
    # 先手番左ルーク
    wr1_move_count: bool = False
    # 先手番右ルーク
    wr2_move_count: bool = False
    # 先手番キング
    wk_move_count: bool = False
    # 後手番左ルーク
    br1_move_count: bool = False
    # 後手番右ルーク
    br2_move_count: bool = False
    # 後手番キング
    bk_move_count: bool = False


def init():
    bs = np.zeros(64, dtype=np.int32)
    for i in range(8):
        bs[1 + 8 * i] = 1
    bs[8] = 2
    bs[48] = 2
    bs[16] = 3
    bs[40] = 3
    bs[0] = 4
    bs[56] = 4
    bs[24] = 5
    bs[32] = 6
    for i in range(8):
        bs[6 + 8 * i] = 7
    bs[15] = 8
    bs[55] = 8
    bs[23] = 9
    bs[47] = 9
    bs[7] = 10
    bs[63] = 10
    bs[31] = 11
    bs[39] = 12
    return ChessState(board=_make_board(bs))


def _make_board(bs: np.ndarray) -> np.ndarray:
    board = np.zeros((13, 64), dtype=np.int32)
    for i in range(64):
        board[0][i] = 0
        board[bs[i]][i] = 1
    return board


# intのactionを向きと距離に変換
def _separate_int_action(action: int) -> Tuple[int, int]:
    # 後ろ
    if action <= 6:
        return 0, 7 - action
    # 前
    elif action <= 13:
        return 1, action - 6
    # 左
    elif action <= 20:
        return 2, 21 - action
    # 右
    elif action <= 27:
        return 3, action - 20
    # 左下
    elif action <= 34:
        return 4, 35 - action
    # 右上
    elif action <= 41:
        return 5, action - 34
    # 左上
    elif action <= 48:
        return 6, 49 - action
    # 右下
    elif action <= 55:
        return 7, action - 48
    # Knightのaction
    # 8~15に割り振る
    elif action <= 63:
        return action - 48, 1
    # promotionのaction
    # 二つ目の返り値をpromotionの駒種にする
    # 直進
    elif action <= 66:
        return 16, action - 63
    # 左
    elif action <= 69:
        return 17, action - 66
    # 右
    elif action <= 72:
        return 18, action - 69
    return -1, 0


# AlphaZeroのdirectionは後ろ、前、左、右、左下、右上、左上、右下の順で記録
def _direction_to_dif(direction: int) -> int:
    # 後ろ
    if direction == 0:
        return -1
    # 前
    if direction == 1:
        return 1
    # 左
    if direction == 2:
        return -8
    # 右
    if direction == 3:
        return 8
    # 左下
    if direction == 4:
        return -9
    # 右上
    if direction == 5:
        return 9
    # 左上
    if direction == 6:
        return -7
    # 右下
    if direction == 7:
        return 7
    # Knightのdif
    # 上左
    if direction == 8:
        return -6
    # 上右
    if direction == 9:
        return 10
    # 左上
    if direction == 10:
        return -15
    # 左下
    if direction == 11:
        return -17
    # 下右
    if direction == 12:
        return 6
    # 下左
    if direction == 13:
        return -10
    # 右下
    if direction == 14:
        return 15
    # 右上
    if direction == 15:
        return 17
    return 0


def convert_to(from_: int, dir: int, dis: int, turn: int):
    if dir <= 15:
        return from_ + _direction_to_dif(dir) * dis
    # 直進のプロモーション
    if dir == 16:
        if turn == 0:
            return from_ + 1
        else:
            return from_ - 1
    # 左のプロモーション
    if dir == 17:
        if turn == 0:
            return from_ - 7
        else:
            return from_ - 9
    # 右のプロモーション
    if dir == 18:
        if turn == 0:
            return from_ + 9
        else:
            return from_ + 7


def int_to_action(state: ChessState, action: int):
    from_ = action % 64
    dir, dis = _separate_int_action(action // 64)
    to = convert_to(from_, dir, dis, state.turn)
    if dir <= 15:
        return ChessAction(_piece_type(state, from_), from_, to)
    else:
        return ChessAction(_piece_type(state, from_), from_, to, dis)


def _piece_type(state: ChessState, position: int):
    return state.board[:, position].argmax()


# promotionの場合にはpromotion後の駒、そうでない場合は元の駒を返す
def _promoted_piece(action: ChessAction):
    if (action.piece == 1 and action.to % 8 == 7) or (
        action.piece == 7 and action.to % 8 == 0
    ):
        # Queenの場合
        if action.promotion == 0:
            return action.piece + 4
        # Rook
        elif action.promotion == 1:
            return action.piece + 3
        # Knight
        elif action.promotion == 2:
            return action.piece + 1
        # Bishop
        else:
            return action.piece + 2
    return action.piece


# castling判定
def _is_castling(action: ChessAction) -> int:
    # King以外の動きは無視
    if action.piece % 6 != 0:
        return 0
    # 左キャスリング
    if action.from_ - action.to == 16:
        return 1
    # 右キャスリング
    if action.from_ - action.to == -16:
        return 2
    return 0


def _move(state: ChessState, action: ChessAction) -> ChessState:
    s = copy.deepcopy(state)
    s.board[action.piece][action.from_] = 0
    s.board[0][action.from_] = 1
    s.board[:, action.to] *= 0
    p = _promoted_piece(action)
    s.board[p][action.to] = 1
    # 先手側のアンパッサン
    if action.piece == 1 and action.to == s.en_passant + 1:
        s.board[:, s.en_passant] *= 0
    # 後手側のアンパッサン
    if action.piece == 7 and action.to == s.en_passant - 1:
        s.board[:, s.en_passant] *= 0
    # 左キャスリング
    if _is_castling(action) == 1:
        if s.turn == 0:
            s.board[4][0] = 0
            s.board[0][0] = 1
            s.board[0][24] = 0
            s.board[4][24] = 1
            s.wr1_move_count = True
        else:
            s.board[10][7] = 0
            s.board[0][7] = 1
            s.board[0][31] = 0
            s.board[10][31] = 1
            s.br1_move_count = True
    # 右キャスリング
    if _is_castling(action) == 2:
        if s.turn == 0:
            s.board[4][56] = 0
            s.board[0][56] = 1
            s.board[0][40] = 0
            s.board[4][40] = 1
            s.wr2_move_count = True
        else:
            s.board[10][63] = 0
            s.board[0][63] = 1
            s.board[0][47] = 0
            s.board[10][47] = 1
            s.br2_move_count = True
    # 各種フラグの更新
    if not s.wr1_move_count and action.from_ == 0:
        s.wr1_move_count = True
    if not s.wr2_move_count and action.from_ == 56:
        s.wr2_move_count = True
    if not s.br1_move_count and action.from_ == 7:
        s.br1_move_count = True
    if not s.br2_move_count and action.from_ == 63:
        s.br2_move_count = True
    if not s.wk_move_count and action.from_ == 32:
        s.wk_move_count = True
    if not s.bk_move_count and action.from_ == 39:
        s.bk_move_count = True
    if action.piece % 6 == 1 and abs(action.from_ - action.to) == 2:
        s.en_passant = action.to
    else:
        s.en_passant = -1
    return s


def point_to_coordinate(point: int):
    return point // 8, point % 8


def _is_same_column(_from: int, to: int) -> bool:
    x1, y1 = point_to_coordinate(_from)
    x2, y2 = point_to_coordinate(to)
    return x1 == x2


def _is_same_row(_from: int, to: int) -> bool:
    x1, y1 = point_to_coordinate(_from)
    x2, y2 = point_to_coordinate(to)
    return y1 == y2


def _is_same_rising(_from: int, to: int) -> bool:
    x1, y1 = point_to_coordinate(_from)
    x2, y2 = point_to_coordinate(to)
    return x1 - x2 == y1 - y2


# from, toが右肩下がりの斜め方向で同じ筋にあるか
def _is_same_declining(_from: int, to: int) -> bool:
    x1, y1 = point_to_coordinate(_from)
    x2, y2 = point_to_coordinate(to)
    return x1 - x2 == y2 - y1


def _board_status(state: ChessState):
    bs = np.zeros(64, dtype=np.int32)
    for i in range(64):
        bs[i] = _piece_type(state, i)
    return bs


def _owner(piece: int):
    if piece == 0:
        return 2
    else:
        return (piece - 1) // 6


def _is_in_board(point: int):
    return 0 <= point <= 63


def _is_side(point: int):
    x, y = point_to_coordinate(point)
    u = False
    d = False
    l_ = False
    r = False
    if y == 7:
        u = True
    if y == 0:
        d = True
    if x == 0:
        l_ = True
    if x == 7:
        r = True
    return u, d, l_, r


def _is_second_line(point: int):
    x, y = point_to_coordinate(point)
    su = False
    sd = False
    sl = False
    sr = False
    if y >= 6:
        su = True
    if y <= 1:
        sd = True
    if x <= 1:
        sl = True
    if x >= 6:
        sr = True
    return su, sd, sl, sr


def _white_pawn_moves(bs: np.ndarray, from_: int, en_passant: int):
    to = np.zeros(64, dtype=np.int32)
    if bs[from_ + 1] == 0:
        to[from_ + 1] = 1
        # 初期位置の場合はニマス進める
        if from_ % 8 == 1 and bs[from_ + 2] == 0:
            to[from_ + 2] = 1
    # 斜めには相手の駒があるときかアンパッサンの時のみ進める
    # 左斜め前
    if _owner(bs[from_ - 7]) == 1 or en_passant == from_ - 8:
        to[from_ - 7] = 1
    # 右斜め前
    if _owner(bs[from_ + 9]) == 1 or en_passant == from_ + 8:
        to[from_ + 9] = 1
    return to


def _black_pawn_moves(bs: np.ndarray, from_: int, en_passant: int):
    to = np.zeros(64, dtype=np.int32)
    if bs[from_ - 1] == 0:
        to[from_ - 1] = 1
        # 初期位置の場合はニマス進める
        if from_ % 8 == 6 and bs[from_ - 2] == 0:
            to[from_ - 2] = 1
    # 斜めには相手の駒があるときかアンパッサンの時のみ進める
    # 左斜め前
    if _owner(bs[from_ - 9]) == 0 or en_passant == from_ - 8:
        to[from_ - 9] = 1
    # 右斜め前
    if _owner(bs[from_ + 7]) == 0 or en_passant == from_ + 8:
        to[from_ + 7] = 1
    return to


def _pawn_moves(bs: np.ndarray, from_: int, en_passant: int, turn: int):
    if turn == 0:
        return _white_pawn_moves(bs, from_, en_passant)
    else:
        return _black_pawn_moves(bs, from_, en_passant)


def _knight_moves(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    su, sd, sl, sr = _is_second_line(from_)
    # 上方向
    if not su:
        if not l_ and _owner(bs[from_ - 6]) != turn:
            to[from_ - 6] = 1
        if not r and _owner(bs[from_ + 10]) != turn:
            to[from_ + 10] = 1
    # 左方向
    if not sl:
        if not u and _owner(bs[from_ - 15]) != turn:
            to[from_ - 15] = 1
        if not d and _owner(bs[from_ - 17]) != turn:
            to[from_ - 17] = 1
    # 下方向
    if not sd:
        if not l_ and _owner(bs[from_ - 10]) != turn:
            to[from_ - 10] = 1
        if not r and _owner(bs[from_ + 6]) != turn:
            to[from_ + 6] = 1
    # 右方向
    if not sr:
        if not u and _owner(bs[from_ + 17]) != turn:
            to[from_ + 17] = 1
        if not d and _owner(bs[from_ + 15]) != turn:
            to[from_ + 15] = 1
    return to


def _bishop_move(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    ur_flag = True
    ul_flag = True
    dr_flag = True
    dl_flag = True
    for i in range(8):
        ur = from_ + 9 * (1 + i)
        ul = from_ - 7 * (1 + i)
        dr = from_ + 7 * (1 + i)
        dl = from_ - 9 * (1 + i)
        if (
            ur_flag
            and _is_in_board(ur)
            and _is_same_rising(from_, ur)
            and _owner(bs[ur]) != turn
        ):
            to[ur] = 1
        if (
            ul_flag
            and _is_in_board(ul)
            and _is_same_declining(from_, ul)
            and _owner(bs[ul]) != turn
        ):
            to[ul] = 1
        if (
            dr_flag
            and _is_in_board(dr)
            and _is_same_declining(from_, dr)
            and _owner(bs[dr]) != turn
        ):
            to[dr] = 1
        if (
            dl_flag
            and _is_in_board(dl)
            and _is_same_rising(from_, dl)
            and _owner(bs[dl]) != turn
        ):
            to[dl] = 1
        if not _is_in_board(ur) or bs[ur] != 0:
            ur_flag = False
        if not _is_in_board(ul) or bs[ul] != 0:
            ul_flag = False
        if not _is_in_board(dr) or bs[dr] != 0:
            dr_flag = False
        if not _is_in_board(dl) or bs[dl] != 0:
            dl_flag = False
    return to


def _rook_move(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u_flag = True
    d_flag = True
    r_flag = True
    l_flag = True
    for i in range(8):
        u = from_ + 1 * (1 + i)
        d = from_ - 1 * (1 + i)
        l_ = from_ - 8 * (1 + i)
        r = from_ + 8 * (1 + i)
        if (
            u_flag
            and _is_in_board(u)
            and _is_same_column(from_, u)
            and _owner(bs[u]) == turn
        ):
            to[u] = 1
        if (
            d_flag
            and _is_in_board(d)
            and _is_same_column(from_, d)
            and _owner(bs[d]) == turn
        ):
            to[d] = 1
        if (
            l_flag
            and _is_in_board(l_)
            and _is_same_row(from_, l_)
            and _owner(bs[l_]) == turn
        ):
            to[l_] = 1
        if (
            r_flag
            and _is_in_board(r)
            and _is_same_row(from_, r)
            and _owner(bs[r]) == turn
        ):
            to[r] = 1
        if not _is_in_board(u) or bs[u] != 0:
            u_flag = False
        if not _is_in_board(d) or bs[d] != 0:
            d_flag = False
        if not _is_in_board(l_) or bs[l_] != 0:
            l_flag = False
        if not _is_in_board(r) or bs[r] != 0:
            r_flag = False
    return to


def _queen_move(bs: np.ndarray, from_: int, turn: int):
    r_move = _rook_move(bs, from_, turn)
    b_move = _bishop_move(bs, from_, turn)
    # r_moveとb_moveは共通項がないので足してよい
    return r_move + b_move


def _king_moves(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    if not u:
        if _owner(bs[from_ + 1]) != turn:
            to[from_ + 1] = 1
        if not l_ and _owner(bs[from_ - 7]) != turn:
            to[from_ - 7] = 1
        if not r and _owner(bs[from_ + 9]) != turn:
            to[from_ + 9] = 1
    if not l_ and _owner(bs[from_ - 8]) != turn:
        to[from_ - 8] = 1
    if not r and _owner(bs[from_ + 8]) != turn:
        to[from_ + 8] = 1
    if not d:
        if _owner(bs[from_ - 1]) != turn:
            to[from_ - 1] = 1
        if not l_ and _owner(bs[from_ - 9]) != turn:
            to[from_ - 9] = 1
        if not r and _owner(bs[from_ + 7]) != turn:
            to[from_ + 7] = 1
