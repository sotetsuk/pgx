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
    np.put(bs, np.arange(1, 65, 8), 1)
    # for i in range(8):
    #    bs[1 + 8 * i] = 1
    bs[8] = 2
    bs[48] = 2
    bs[16] = 3
    bs[40] = 3
    bs[0] = 4
    bs[56] = 4
    bs[24] = 5
    bs[32] = 6
    np.put(bs, np.arange(6, 70, 8), 7)
    # for i in range(8):
    #    bs[6 + 8 * i] = 7
    bs[15] = 8
    bs[55] = 8
    bs[23] = 9
    bs[47] = 9
    bs[7] = 10
    bs[63] = 10
    bs[31] = 11
    bs[39] = 12
    return ChessState(board=_make_board(bs))


def step(state: ChessState, i_action: int) -> Tuple[ChessState, int, int]:
    # 指定値が合法手でない
    action = int_to_action(state, i_action)
    is_castling = _is_castling(action)
    kp = int(state.board[6 + state.turn * 6, :].argmax())
    bs = _board_status(state)
    if (
        not _is_legal_action(state, i_action, _pin(state, kp))
        and not (is_castling == 1 and _can_left_castling(state, bs))
        and not (is_castling == 2 and _can_right_castling(state, bs))
    ):
        print("not legal action")
        return state, _turn_to_reward(_another_color(state)), True
    s = _move(state, action, is_castling)
    # move後にcheckがかかっている
    kp = int(s.board[6 + s.turn * 6, :].argmax())
    bs = _board_status(s)
    if _is_check(bs, s.turn, kp):
        print("leave check")
        return s, _turn_to_reward(_another_color(state)), True
    s.turn = _another_color(s)
    kp = int(s.board[6 + s.turn * 6, :].argmax())
    pins = _pin(s, kp)
    if _is_check(bs, s.turn, kp) and _is_mate_check(s, kp, pins):
        print("mate")
        return s, _turn_to_reward(state.turn), True
    if not _is_check(bs, s.turn, kp) and _is_mate_non_check(s, kp, pins):
        print("stale mate")
        return s, 0, True
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
    return s, 0, False


def _turn_to_reward(turn: int) -> int:
    if turn == 0:
        return 1
    else:
        return -1


def _another_color(state: ChessState) -> int:
    return (state.turn + 1) % 2


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
    # 後ろ 0~6
    if direction == 0:
        return -1
    # 前 7~13
    if direction == 1:
        return 1
    # 左 14~20
    if direction == 2:
        return -8
    # 右 21~27
    if direction == 3:
        return 8
    # 左下 28~34
    if direction == 4:
        return -9
    # 右上 35~41
    if direction == 5:
        return 9
    # 左上 42~48
    if direction == 6:
        return -7
    # 右下 49~55
    if direction == 7:
        return 7
    # Knightのdif
    # 上左 56
    if direction == 8:
        return -6
    # 上右 57
    if direction == 9:
        return 10
    # 左上 58
    if direction == 10:
        return -15
    # 左下 59
    if direction == 11:
        return -17
    # 下右 60
    if direction == 12:
        return 6
    # 下左 61
    if direction == 13:
        return -10
    # 右下 62
    if direction == 14:
        return 15
    # 右上 63
    if direction == 15:
        return 17
    return 0


def convert_to(from_: int, dir: int, dis: int, turn: int) -> int:
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
    return from_


def int_to_action(state: ChessState, action: int) -> ChessAction:
    from_ = action % 64
    dir, dis = _separate_int_action(action // 64)
    to = convert_to(from_, dir, dis, state.turn)
    if dir <= 15:
        return ChessAction(_piece_type(state, from_), from_, to)
    else:
        return ChessAction(_piece_type(state, from_), from_, to, dis)


# 二地点の位置関係と距離
def dif_to_direction(from_: int, to: int) -> Tuple[int, int]:
    dif = to - from_
    if _is_same_column(from_, to):
        if dif < 0:
            return 0, 7 + dif
        else:
            return 1, dif - 1
    elif _is_same_row(from_, to):
        if dif < 0:
            return 2, 7 + dif // 8
        else:
            return 3, dif // 8 - 1
    elif _is_same_rising(from_, to):
        if dif < 0:
            return 4, 7 + dif // 9
        else:
            return 5, dif // 9 - 1
    elif _is_same_declining(from_, to):
        if dif < 0:
            return 6, 7 + dif // 7
        else:
            return 7, dif // 7 - 1
    elif dif == -6:
        return 8, 1
    elif dif == 10:
        return 9, 1
    elif dif == -15:
        return 10, 1
    elif dif == -17:
        return 11, 1
    elif dif == 6:
        return 12, 1
    elif dif == -10:
        return 13, 1
    elif dif == 15:
        return 14, 1
    elif dif == 17:
        return 15, 1
    return -1, 0


def _is_same_line(from_: int, to: int, direction: int):
    if direction <= 1:
        return _is_same_column(from_, to)
    elif direction <= 3:
        return _is_same_row(from_, to)
    elif direction <= 5:
        return _is_same_rising(from_, to)
    elif direction <= 7:
        return _is_same_declining(from_, to)
    else:
        return False


def _dis_direction_array(from_: int, direction: int):
    array = np.zeros(64, dtype=np.int32)
    dif = _direction_to_dif(direction)
    to = from_ + dif
    # for i in range(7):
    #    to_ = from_ + dif * (1 + i)
    #    if _is_in_board(to_) and _is_same_line(from_, to_, direction):
    #        array[to_] = i + 1
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 1
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 2
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 3
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 4
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 5
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 6
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 7
    return array


def _piece_type(state: ChessState, position: int) -> int:
    return int(state.board[:, position].argmax())


# promotionの場合にはpromotion後の駒、そうでない場合は元の駒を返す
def _promoted_piece(action: ChessAction) -> int:
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


def _move(
    state: ChessState, action: ChessAction, is_castling: int
) -> ChessState:
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
    if is_castling == 1:
        if s.turn == 0:
            s.board[4][0] = 0
            s.board[0][0] = 1
            s.board[0][24] = 0
            s.board[4][24] = 1
        else:
            s.board[10][7] = 0
            s.board[0][7] = 1
            s.board[0][31] = 0
            s.board[10][31] = 1
    # 右キャスリング
    if is_castling == 2:
        if s.turn == 0:
            s.board[4][56] = 0
            s.board[0][56] = 1
            s.board[0][40] = 0
            s.board[4][40] = 1
        else:
            s.board[10][63] = 0
            s.board[0][63] = 1
            s.board[0][47] = 0
            s.board[10][47] = 1
    return s


def point_to_coordinate(point: int) -> Tuple[int, int]:
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


def _board_status(state: ChessState) -> np.ndarray:
    return state.board.argmax(axis=0)


def _owner(piece: int) -> int:
    if piece == 0:
        return 2
    else:
        return (piece - 1) // 6


def _is_in_board(point: int) -> bool:
    return 0 <= point <= 63


def _is_side(point: int) -> Tuple[bool, bool, bool, bool]:
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


def _is_second_line(point: int) -> Tuple[bool, bool, bool, bool]:
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


# fromの座標とtoの座標からdirを生成
# 8方向のみ生成
def _point_to_direction(_from: int, to: int) -> int:
    direction = -1
    dis = to - _from
    if _is_same_column(_from, to) and dis < 0:
        direction = 0
    if _is_same_column(_from, to) and dis > 0:
        direction = 1
    if _is_same_row(_from, to) and dis < 0:
        direction = 2
    if _is_same_row(_from, to) and dis > 0:
        direction = 3
    if _is_same_rising(_from, to) and dis < 0:
        direction = 4
    if _is_same_rising(_from, to) and dis > 0:
        direction = 5
    if _is_same_declining(_from, to) and dis < 0:
        direction = 6
    if _is_same_declining(_from, to) and dis > 0:
        direction = 7
    return direction


def _white_pawn_moves(bs: np.ndarray, from_: int, pin: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    if bs[from_ + 1] == 0 and (pin == 0 or pin == 1):
        to[from_ + 1] = 1
        # 初期位置の場合はニマス進める
        if from_ % 8 == 1 and bs[from_ + 2] == 0:
            to[from_ + 2] = 1
    # 斜めには相手の駒があるとき進める
    # アンパッサンはリーガルアクションで追加
    # 左斜め前
    if _is_in_board(from_ - 7) and (pin == 0 or pin == 4):
        if _owner(bs[from_ - 7]) == 1:
            to[from_ - 7] = 1
    # 右斜め前
    if _is_in_board(from_ + 9) and (pin == 0 or pin == 3):
        if _owner(bs[from_ + 9]) == 1:
            to[from_ + 9] = 1
    return to


def _black_pawn_moves(bs: np.ndarray, from_: int, pin: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    if bs[from_ - 1] == 0 and (pin == 0 or pin == 1):
        to[from_ - 1] = 1
        # 初期位置の場合はニマス進める
        if from_ % 8 == 6 and bs[from_ - 2] == 0:
            to[from_ - 2] = 1
    # 斜めには相手の駒があるとき進める
    # 左斜め前
    if _is_in_board(from_ - 9) and (pin == 0 or pin == 3):
        if _owner(bs[from_ - 9]) == 0:
            to[from_ - 9] = 1
        elif _owner(bs[from_ - 9]) == 1:
            to[from_ - 9] = 2
    # 右斜め前
    if _is_in_board(from_ + 7) and (pin == 0 or pin == 4):
        if _owner(bs[from_ + 7]) == 0:
            to[from_ + 7] = 1
        elif _owner(bs[from_ + 7]) == 1:
            to[from_ + 7] = 2
    return to


def _pawn_moves(bs: np.ndarray, from_: int, turn: int, pin: int) -> np.ndarray:
    if turn == 0:
        return _white_pawn_moves(bs, from_, pin)
    else:
        return _black_pawn_moves(bs, from_, pin)


def _white_pawn_effects(from_: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    # 左斜め前
    if _is_in_board(from_ - 7):
        to[from_ - 7] = 1
    # 右斜め前
    if _is_in_board(from_ + 9):
        to[from_ + 9] = 1
    return to


def _black_pawn_effects(from_: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    # 左斜め前
    if _is_in_board(from_ - 9):
        to[from_ - 9] = 1
    # 右斜め前
    if _is_in_board(from_ + 7):
        to[from_ + 7] = 1
    return to


def _pawn_effects(from_: int, turn: int) -> np.ndarray:
    if turn == 0:
        return _white_pawn_effects(from_)
    else:
        return _black_pawn_effects(from_)


def _piece_turn_to_color(piece: int, turn: int) -> int:
    if _owner(piece) == turn:
        return 2
    else:
        return 1


def _knight_moves(
    bs: np.ndarray, from_: int, turn: int, pin: int
) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    # pinされている場合は動けない
    if pin != 0:
        return to
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


def _knight_effect(from_: int):
    to = np.zeros(64, dtype=np.int32)
    # pinされている場合は動けない
    u, d, l_, r = _is_side(from_)
    su, sd, sl, sr = _is_second_line(from_)
    # 上方向
    if not su:
        if not l_:
            to[from_ - 6] = 1
        if not r:
            to[from_ + 10] = 1
    if not sl:
        if not u:
            to[from_ - 15] = 1
        if not d:
            to[from_ - 17] = 1
    if not sd:
        if not l_:
            to[from_ - 10] = 1
        if not r:
            to[from_ + 6] = 1
    if not sr:
        if not u:
            to[from_ + 17] = 1
        if not d:
            to[from_ + 15] = 1
    return to


def _bishop_moves(
    bs: np.ndarray, from_: int, turn: int, pin: int
) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    # pinされている方向以外のフラグはあらかじめ折っておく
    ur_flag = pin == 0 or pin == 3
    ul_flag = pin == 0 or pin == 4
    dr_flag = pin == 0 or pin == 4
    dl_flag = pin == 0 or pin == 3
    ur_array = _dis_direction_array(from_, 5)
    ul_array = _dis_direction_array(from_, 6)
    dr_array = _dis_direction_array(from_, 7)
    dl_array = _dis_direction_array(from_, 4)
    bs_one = np.where(bs == 0, 0, 1)
    if ur_flag:
        if np.all(bs_one * ur_array == 0):
            if np.all(ur_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(ur_array)
        else:
            max_dis = np.min(ur_array[np.nonzero(ur_array * bs_one)])
            if _owner(bs[from_ + 9 * max_dis]) == turn:
                max_dis -= 1
        ur_point = from_ + 9 * max_dis
        np.put(to, np.arange(ur_point, from_, -9), 1)
    if ul_flag:
        if np.all(bs_one * ul_array == 0):
            if np.all(ul_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(ul_array)
        else:
            max_dis = np.min(ul_array[np.nonzero(ul_array * bs_one)])
            if _owner(bs[from_ - 7 * max_dis]) == turn:
                max_dis -= 1
        ul_point = from_ - 7 * max_dis
        np.put(to, np.arange(ul_point, from_, 7), 1)
    if dr_flag:
        if np.all(bs_one * dr_array == 0):
            if np.all(dr_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(dr_array)
        else:
            max_dis = np.min(dr_array[np.nonzero(dr_array * bs_one)])
            if _owner(bs[from_ + 7 * max_dis]) == turn:
                max_dis -= 1
        dr_point = from_ + 7 * max_dis
        np.put(to, np.arange(dr_point, from_, -7), 1)
    if dl_flag:
        if np.all(bs_one * dl_array == 0):
            if np.all(dl_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(dl_array)
        else:
            max_dis = np.min(dl_array[np.nonzero(dl_array * bs_one)])
            if _owner(bs[from_ - 9 * max_dis]) == turn:
                max_dis -= 1
        dl_point = from_ - 9 * max_dis
        np.put(to, np.arange(dl_point, from_, 9), 1)
    return to


def _bishop_effect(bs: np.ndarray, from_: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    ur_array = _dis_direction_array(from_, 5)
    ul_array = _dis_direction_array(from_, 6)
    dr_array = _dis_direction_array(from_, 7)
    dl_array = _dis_direction_array(from_, 4)
    bs_one = np.where(bs == 0, 0, 1)
    if np.all(bs_one * ur_array == 0):
        # ur_arrayに被る駒が存在しない
        if np.all(ur_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(ur_array)
    else:
        max_dis = np.min(ur_array[np.nonzero(ur_array * bs_one)])
    ur_point = from_ + 9 * max_dis
    np.put(to, np.arange(ur_point, from_, -9), 1)
    if np.all(bs_one * ul_array == 0):
        if np.all(ul_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(ul_array)
    else:
        max_dis = np.min(ul_array[np.nonzero(ul_array * bs_one)])
    ul_point = from_ - 7 * max_dis
    np.put(to, np.arange(ul_point, from_, 7), 1)
    if np.all(bs_one * dr_array == 0):
        if np.all(dr_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(dr_array)
    else:
        max_dis = np.min(dr_array[np.nonzero(dr_array * bs_one)])
    dr_point = from_ + 7 * max_dis
    np.put(to, np.arange(dr_point, from_, -7), 1)
    if np.all(bs_one * dl_array == 0):
        if np.all(dl_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(dl_array)
    else:
        max_dis = np.min(dl_array[np.nonzero(dl_array * bs_one)])
    dl_point = from_ - 9 * max_dis
    np.put(to, np.arange(dl_point, from_, 9), 1)
    return to


def _rook_moves(bs: np.ndarray, from_: int, turn: int, pin: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    u_flag = pin == 0 or pin == 1
    d_flag = pin == 0 or pin == 1
    r_flag = pin == 0 or pin == 2
    l_flag = pin == 0 or pin == 2
    u_array = _dis_direction_array(from_, 1)
    d_array = _dis_direction_array(from_, 0)
    r_array = _dis_direction_array(from_, 3)
    l_array = _dis_direction_array(from_, 2)
    bs_one = np.where(bs == 0, 0, 1)
    if u_flag:
        if np.all(bs_one * u_array == 0):
            if np.all(u_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(u_array)
        else:
            max_dis = np.min(u_array[np.nonzero(u_array * bs_one)])
            if _owner(bs[from_ + max_dis]) == turn:
                max_dis -= 1
        u_point = from_ + max_dis
        np.put(to, np.arange(u_point, from_, -1), 1)
    if d_flag:
        if np.all(bs_one * d_array == 0):
            if np.all(d_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(d_array)
        else:
            max_dis = np.min(d_array[np.nonzero(d_array * bs_one)])
            if _owner(bs[from_ - max_dis]) == turn:
                max_dis -= 1
        d_point = from_ - max_dis
        np.put(to, np.arange(d_point, from_, 1), 1)
    if r_flag:
        if np.all(bs_one * r_array == 0):
            if np.all(r_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(r_array)
        else:
            max_dis = np.min(r_array[np.nonzero(r_array * bs_one)])
            if _owner(bs[from_ + 8 * max_dis]) == turn:
                max_dis -= 1
        r_point = from_ + 8 * max_dis
        np.put(to, np.arange(r_point, from_, -8), 1)
    if l_flag:
        if np.all(bs_one * l_array == 0):
            if np.all(l_array == 0):
                max_dis = 0
            else:
                max_dis = np.max(l_array)
        else:
            max_dis = np.min(l_array[np.nonzero(l_array * bs_one)])
            if _owner(bs[from_ - 8 * max_dis]) == turn:
                max_dis -= 1
        l_point = from_ - 8 * max_dis
        np.put(to, np.arange(l_point, from_, 8), 1)
    return to


def _rook_effect(bs: np.ndarray, from_: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    u_array = _dis_direction_array(from_, 1)
    d_array = _dis_direction_array(from_, 0)
    r_array = _dis_direction_array(from_, 3)
    l_array = _dis_direction_array(from_, 2)
    bs_one = np.where(bs == 0, 0, 1)
    if np.all(bs_one * u_array == 0):
        if np.all(u_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(u_array)
    else:
        max_dis = np.min(u_array[np.nonzero(u_array * bs_one)])
    u_point = from_ + max_dis
    np.put(to, np.arange(u_point, from_, -1), 1)
    if np.all(bs_one * d_array == 0):
        if np.all(d_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(d_array)
    else:
        max_dis = np.min(d_array[np.nonzero(d_array * bs_one)])
    d_point = from_ - max_dis
    np.put(to, np.arange(d_point, from_, 1), 1)
    if np.all(bs_one * r_array == 0):
        if np.all(r_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(r_array)
    else:
        max_dis = np.min(r_array[np.nonzero(r_array * bs_one)])
    r_point = from_ + 8 * max_dis
    np.put(to, np.arange(r_point, from_, -8), 1)
    if np.all(bs_one * l_array == 0):
        if np.all(l_array == 0):
            max_dis = 0
        else:
            max_dis = np.max(l_array)
    else:
        max_dis = np.min(l_array[np.nonzero(l_array * bs_one)])
    l_point = from_ - 8 * max_dis
    np.put(to, np.arange(l_point, from_, 8), 1)
    return to


def _queen_moves(
    bs: np.ndarray, from_: int, turn: int, pin: int
) -> np.ndarray:
    r_move = _rook_moves(bs, from_, turn, pin)
    b_move = _bishop_moves(bs, from_, turn, pin)
    # r_moveとb_moveは共通項がないので足してよい
    return r_move + b_move


def _queen_effect(bs: np.ndarray, from_: int) -> np.ndarray:
    r_ef = _rook_effect(bs, from_)
    b_ef = _bishop_effect(bs, from_)
    # r_moveとb_moveは共通項がないので足してよい
    return r_ef + b_ef


def _king_moves(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    if not u:
        if _owner(bs[from_ + 1]) != turn:
            to[from_ + 1] = 1
        if not l_:
            if _owner(bs[from_ - 7]) != turn:
                to[from_ - 7] = 1
        if not r:
            if _owner(bs[from_ + 9]) != turn:
                to[from_ + 9] = 1
    if not l_:
        if _owner(bs[from_ - 8]) != turn:
            to[from_ - 8] = 1
    if not r:
        if _owner(bs[from_ + 8]) != turn:
            to[from_ + 8] = 1
    if not d:
        if _owner(bs[from_ - 1]) != turn:
            to[from_ - 1] = 1
        if not l_:
            if _owner(bs[from_ - 9]) != turn:
                to[from_ - 9] = 1
        if not r:
            if _owner(bs[from_ + 7]) != turn:
                to[from_ + 7] = 1
    return to


def _king_effect(from_: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    if not u:
        to[from_ + 1] = 1
        if not l_:
            to[from_ - 7] = 1
        if not r:
            to[from_ + 9] = 1
    if not l_:
        to[from_ - 8] = 1
    if not r:
        to[from_ + 8] = 1
    if not d:
        to[from_ - 1] = 1
        if not l_:
            to[from_ - 9] = 1
        if not r:
            to[from_ + 7] = 1
    return to


def _piece_moves(
    bs: np.ndarray, from_: int, piece: int, pins: np.ndarray
) -> np.ndarray:
    pin = pins[from_]
    if piece == 0:
        return np.zeros(64, dtype=np.int32)
    turn = (piece - 1) // 6
    p = piece % 6
    if p == 1:
        return _pawn_moves(bs, from_, turn, pin)
    elif p == 2:
        return _knight_moves(bs, from_, turn, pin)
    elif p == 3:
        return _bishop_moves(bs, from_, turn, pin)
    elif p == 4:
        return _rook_moves(bs, from_, turn, pin)
    elif p == 5:
        return _queen_moves(bs, from_, turn, pin)
    else:
        return _king_moves(bs, from_, turn)


def _piece_effect(bs: np.ndarray, from_: int, piece: int) -> np.ndarray:
    if piece == 0:
        return np.zeros(64, dtype=np.int32)
    turn = (piece - 1) // 6
    p = piece % 6
    if p == 1:
        return _pawn_effects(from_, turn)
    elif p == 2:
        return _knight_effect(from_)
    elif p == 3:
        return _bishop_effect(bs, from_)
    elif p == 4:
        return _rook_effect(bs, from_)
    elif p == 5:
        return _queen_effect(bs, from_)
    else:
        return _king_effect(from_)


def _create_actions(from_: int, to: np.ndarray) -> np.ndarray:
    actions = np.zeros(4608, dtype=np.int32)
    for i in range(64):
        if to[i] != 1:
            continue
        dir, dis = dif_to_direction(from_, i)
        if dir <= 7:
            actions[64 * (7 * dir + dis) + from_] = 1
        else:
            actions[64 * (48 + dir) + from_] = 1
    return actions


def _is_check(bs: np.ndarray, turn: int, kp: int) -> bool:
    return _is_check2(bs, turn, kp)[0] != 0


def _is_check2(bs: np.ndarray, turn: int, kp: int) -> Tuple[int, np.ndarray]:
    checking_piece = np.zeros(64, dtype=np.int32)
    e_color = (turn + 1) % 2
    pe = _pawn_effects(kp, turn)
    np.put(checking_piece, np.where(bs * pe == 1 + 6 * e_color), 1)
    ke = _knight_effect(kp)
    np.put(checking_piece, np.where(bs * ke == 2 + 6 * e_color), 1)
    be = _bishop_effect(bs, kp)
    np.put(checking_piece, np.where(bs * be == 3 + 6 * e_color), 1)
    re = _rook_effect(bs, kp)
    np.put(checking_piece, np.where(bs * re == 4 + 6 * e_color), 1)
    qe = _queen_effect(bs, kp)
    np.put(checking_piece, np.where(bs * qe == 5 + 6 * e_color), 1)
    return np.count_nonzero(checking_piece), checking_piece


def _effected_positions(bs: np.ndarray, turn: int) -> np.ndarray:
    effects = np.zeros(64, dtype=np.int32)
    for i in range(64):
        piece = bs[i]
        effects += _piece_effect(bs, i, piece) if _owner(piece) == turn else 0
    return effects


def _can_left_castling(state: ChessState, bs: np.ndarray) -> bool:
    if state.turn == 0:
        if state.wk_move_count or state.wr1_move_count:
            return False
        # 間に他の駒がある場合不可
        if bs[8] + bs[16] + bs[24] != 0:
            return False
        effects = _effected_positions(bs, 1)
        # 相手の駒の利きが通り道にある場合不可
        return effects[16] + effects[24] + effects[32] == 0
    if state.turn == 1:
        if state.bk_move_count or state.br1_move_count:
            return False
        # 間に他の駒がある場合不可
        if bs[15] + bs[23] + bs[31] != 0:
            return False
        effects = _effected_positions(bs, 0)
        # 相手の駒の利きが通り道にある場合不可
        return effects[23] + effects[31] + effects[39] == 0
    return False


def _can_right_castling(state: ChessState, bs: np.ndarray) -> bool:
    if state.turn == 0:
        if state.wk_move_count or state.wr2_move_count:
            return False
        # 間に他の駒がある場合不可
        if bs[48] + bs[40] != 0:
            return False
        effects = _effected_positions(bs, 1)
        # 相手の駒の利きが通り道にある場合不可
        return effects[48] + effects[40] + effects[32] == 0
    if state.turn == 1:
        if state.bk_move_count or state.br2_move_count:
            return False
        # 間に他の駒がある場合不可
        if bs[55] + bs[47] != 0:
            return False
        effects = _effected_positions(bs, 0)
        # 相手の駒の利きが通り道にある場合不可
        return effects[55] + effects[47] + effects[39] == 0
    return False


def _legal_actions(state: ChessState) -> np.ndarray:
    actions = np.zeros(4608, dtype=np.int32)
    bs = _board_status(state)
    for i in range(64):
        piece = bs[i]
        if _owner(piece) != state.turn:
            continue
        p_moves = _piece_moves(bs, i, piece, np.zeros(64, dtype=np.int32))
        p_actions = _create_actions(i, p_moves)
        actions += p_actions
        # promotionの場合
        if piece == 1 and i % 8 == 6:
            if _is_in_board(i - 7) and p_moves[i - 7] == 1:
                actions[i + 64 * 67] = 1
                actions[i + 64 * 68] = 1
                actions[i + 64 * 69] = 1
            if _is_in_board(i + 1) and p_moves[i + 1] == 1:
                actions[i + 64 * 64] = 1
                actions[i + 64 * 65] = 1
                actions[i + 64 * 66] = 1
            if _is_in_board(i + 9) and p_moves[i + 9] == 1:
                actions[i + 64 * 70] = 1
                actions[i + 64 * 71] = 1
                actions[i + 64 * 72] = 1
        if piece == 7 and i % 8 == 1:
            if _is_in_board(i - 9) and p_moves[i - 9] == 1:
                actions[i + 64 * 67] = 1
                actions[i + 64 * 68] = 1
                actions[i + 64 * 69] = 1
            if _is_in_board(i - 1) and p_moves[i - 1] == 1:
                actions[i + 64 * 64] = 1
                actions[i + 64 * 65] = 1
                actions[i + 64 * 66] = 1
            if _is_in_board(i + 7) and p_moves[i + 7] == 1:
                actions[i + 64 * 70] = 1
                actions[i + 64 * 71] = 1
                actions[i + 64 * 72] = 1
        # アンパッサンの場合
        if piece == 1 and i - 8 == state.en_passant:
            actions[i + 64 * 48] = 1
        if piece == 1 and i + 8 == state.en_passant:
            actions[i + 64 * 35] = 1
        if piece == 7 and i - 8 == state.en_passant:
            actions[i + 64 * 34] = 1
        if piece == 7 and i + 8 == state.en_passant:
            actions[i + 64 * 49] = 1
    # castling
    if _can_left_castling(state, bs):
        actions[32 + state.turn * 7 + 64 * 19] = 1
    if _can_right_castling(state, bs):
        actions[32 + state.turn * 7 + 64 * 22] = 1
    return actions


# メイトおよびスティルメイトの判定関数
def _is_mate(state: ChessState, actions: np.ndarray) -> bool:
    f = True
    for i in range(4608):
        if actions[i] == 0:
            continue
        action = int_to_action(state, i)
        # is_castlingは呼び出す必要がない（castling後にcheckがかかる場合は弾いている）
        s = _move(state, action, 0)
        king_point = int(s.board[6 + 6 * state.turn, :].argmax())
        # move後にcheckがかかっていない手が存在するならFalse
        if not _is_check(_board_status(s), s.turn, king_point):
            f = False
    return f


# point1とpoint2の間の位置を返す
# point2も含める
def _between(point1: int, point2: int) -> np.ndarray:
    between = np.zeros(64, dtype=np.int32)
    direction = _point_to_direction(point1, point2)
    if direction == -1:
        return between
    dif = _direction_to_dif(direction)
    np.put(between, np.arange(point1 + dif, point2 + dif, dif), 1)
    return between


# checkを受けているときの詰み判定
def _is_mate_check(state: ChessState, kp: int, pins: np.ndarray) -> bool:
    # Kingが逃げる手に合法手が存在するかを考える
    bs = _board_status(state)
    king_move = _king_moves(bs, kp, state.turn)
    num_check, checking_piece = _is_check2(bs, state.turn, kp)
    kingless_bs = _board_status(state)
    kingless_bs[kp] = 0
    kingless_effects = _effected_positions(kingless_bs, _another_color(state))
    # king_moveが1かつkingless_effectsが0の地点があれば詰みではない
    if np.any(king_move * (kingless_effects + 1) == 1):
        return False
    # Kingを動かさない回避手の存在判定
    # 2枚以上からCheckされている場合→詰み
    if num_check == 2:
        return True
    check_point = int(checking_piece.argmax())
    bet = _between(kp, check_point)
    for i in range(64):
        piece = kingless_bs[i]
        if _owner(piece) != state.turn:
            continue
        pm = _piece_moves(bs, i, piece, pins)
        # checkしている駒やその駒との間に動ける駒がある場合、詰みでない
        if np.any(bet * pm == 1):
            return False
    return True


def _is_mate_non_check(state: ChessState, kp: int, pins: np.ndarray) -> bool:
    # Kingが移動する手に合法手が存在するかを考える
    normal_bs = _board_status(state)
    kingless_bs = _board_status(state)
    king_move = _king_moves(normal_bs, kp, state.turn)
    kingless_bs[kp] = 0
    kingless_effects = _effected_positions(kingless_bs, _another_color(state))
    # king_moveが1かつkingless_effectsが0の地点があれば詰みではない
    if np.any(king_move * (kingless_effects + 1) == 1):
        return False
    # Kingが逃げる手以外の合法手を調べる
    for i in range(64):
        piece = kingless_bs[i]
        if _owner(piece) != state.turn:
            continue
        pm = _piece_moves(normal_bs, i, piece, pins)
        # ひとつでも行ける地点があるならスティルメイトではない
        if np.any(pm == 1):
            return False
    return True


def _is_legal_action(state: ChessState, action: int, pins: np.ndarray):
    bs = _board_status(state)
    from_ = action % 64
    pin = pins[from_]
    direction = action // 64
    piece = bs[from_]
    p_moves = _piece_moves(bs, from_, piece, pins)
    p_actions = _create_actions(from_, p_moves)
    # en_passantを追加
    if piece == 1 and from_ - 8 == state.en_passant and (pin == 0 or pin == 4):
        p_actions[from_ + 64 * 48] = 1
    if piece == 1 and from_ + 8 == state.en_passant and (pin == 0 or pin == 3):
        p_actions[from_ + 64 * 35] = 1
    if piece == 7 and from_ - 8 == state.en_passant and (pin == 0 or pin == 3):
        p_actions[from_ + 64 * 34] = 1
    if piece == 7 and from_ + 8 == state.en_passant and (pin == 0 or pin == 4):
        p_actions[from_ + 64 * 49] = 1
    # Queen以外へのpromotionの場合、Queenへのpromotionが合法ならOK
    format_action = action
    if direction >= 64:
        if piece == 1:
            if direction <= 66:
                format_action = from_ + 64 * 7
            elif direction <= 69:
                format_action = from_ + 64 * 48
            else:
                format_action = from_ + 64 * 35
        elif piece == 7:
            if direction <= 66:
                format_action = from_ + 64 * 6
            elif direction <= 69:
                format_action = from_ + 64 * 34
            else:
                format_action = from_ + 64 * 49
    return p_actions[format_action] == 1


def _direction_to_pin(direction: int) -> int:
    if direction == 0 or direction == 1:
        return 1
    if direction == 2 or direction == 3:
        return 2
    if direction == 4 or direction == 5:
        return 3
    if direction == 6 or direction == 7:
        return 4
    return 0


def _direction_pin(
    bs: np.ndarray,
    turn: int,
    king_point: int,
    direction: int,
    array: np.ndarray,
) -> np.ndarray:
    new_array = array
    dir_array = _dis_direction_array(king_point, direction)
    e_turn = (turn + 1) % 2
    bs_one = np.where(bs == 0, 0, 1)
    dir_one_array = dir_array * bs_one
    if np.count_nonzero(dir_one_array) <= 1:
        return new_array
    dif = _direction_to_dif(direction)
    dis1 = np.partition(dir_one_array[dir_one_array.nonzero()].flatten(), 0)[0]
    dis2 = np.partition(dir_one_array[dir_one_array.nonzero()].flatten(), 1)[1]
    point1 = king_point + dis1 * dif
    point2 = king_point + dis2 * dif
    piece1 = bs[point1]
    piece2 = bs[point2]
    if _owner(piece1) != turn:
        return new_array
    if direction <= 3:
        if piece2 == 4 + 6 * e_turn or piece2 == 5 + 6 * e_turn:
            new_array[point1] = _direction_to_pin(direction)
    else:
        if piece2 == 3 + 6 * e_turn or piece2 == 5 + 6 * e_turn:
            new_array[point1] = _direction_to_pin(direction)
    return new_array


def _pin(state: ChessState, kp: int):
    bs = _board_status(state)
    turn = state.turn
    pins = np.zeros(64, dtype=np.int32)
    for i in range(8):
        pins = _direction_pin(bs, turn, kp, i, pins)
    return pins
