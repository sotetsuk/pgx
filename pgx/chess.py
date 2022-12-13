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
    #for i in range(8):
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
    #for i in range(8):
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
    #for i in range(7):
    #    to_ = from_ + dif * (1 + i)
    #    if _is_in_board(to_) and _is_same_line(from_, to_, direction):
    #        array[to_] = i + 1
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 7
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 6
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 5
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 4
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 3
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 2
    to += dif
    if _is_in_board(to) and _is_same_line(from_, to, direction):
        array[to] = 1
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
        elif _owner(bs[from_ - 7]) == 0:
            to[from_ - 7] = 2
    # 右斜め前
    if _is_in_board(from_ + 9) and (pin == 0 or pin == 3):
        if _owner(bs[from_ + 9]) == 1:
            to[from_ + 9] = 1
        elif _owner(bs[from_ + 9]) == 0:
            to[from_ + 9] = 2
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


def _knight_effect(from_: int, turn: int):
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
            # ur_arrayに被る駒が存在しない
            if np.all(ur_array == 0):
                max_dis = 0
            else:
                max_dis = 8 - np.min(ur_array[np.nonzero(ur_array)])
        else:
            # ur_arrayの途中に駒が存在する
            max_dis = 7 - np.max(bs_one * ur_array)
        print(max_dis)
        ur_point = from_ + 9 * max_dis
        if _is_in_board(ur_point + 9) and _is_same_rising(from_, ur_point + 9) and _owner(bs[ur_point + 9]) != turn:
            ur_point += 9
        print(ur_point)
        np.put(to, np.arange(ur_point, from_, -9), 1)
    if ul_flag:
        if np.all(bs_one * ul_array == 0):
            if np.all(ul_array == 0):
                max_dis = 0
            else:
                max_dis = 8 - np.min(ul_array[np.nonzero(ul_array)])
        else:
            max_dis = 7 - np.max(bs_one * ul_array)
        print(max_dis)
        ul_point = from_ - 7 * max_dis
        if _is_in_board(ul_point - 7) and _is_same_declining(from_, ul_point - 7) and _owner(bs[ul_point - 7]) != turn:
            ul_point -= 7
        print(ul_point)
        np.put(to, np.arange(ul_point, from_, 7), 1)
    if dr_flag:
        if np.all(bs_one * dr_array == 0):
            if np.all(dr_array == 0):
                max_dis = 0
            else:
                max_dis = 8 - np.min(dr_array[np.nonzero(dr_array)])
        else:
            # ur_arrayの途中に駒が存在する
            max_dis = 7 - np.max(bs_one * dr_array)
        print(max_dis)
        dr_point = from_ + 7 * max_dis
        if _is_in_board(dr_point + 7) and _is_same_declining(from_, dr_point + 7) and _owner(bs[dr_point + 7]) != turn:
            dr_point += 7
        print(dr_point)
        np.put(to, np.arange(dr_point, from_, -7), 1)
    if dl_flag:
        if np.all(bs_one * dl_array == 0):
            if np.all(dl_array == 0):
                max_dis = 0
            else:
                max_dis = 8 - np.min(dl_array[np.nonzero(dl_array)])
        else:
            # ur_arrayの途中に駒が存在する
            max_dis = 7 - np.max(bs_one * dl_array)
        print(max_dis)
        dl_point = from_ - 9 * max_dis
        if _is_in_board(dl_point - 9) and _is_same_declining(from_, dl_point - 9) and _owner(bs[dl_point - 9]) != turn:
            dl_point -= 9
        print(dl_point)
        np.put(to, np.arange(dl_point, from_, 9), 1)
    #for i in range(8):
    #    ur = from_ + 9 * (1 + i)
    #    ul = from_ - 7 * (1 + i)
    #    dr = from_ + 7 * (1 + i)
    #    dl = from_ - 9 * (1 + i)
    #    if ur_flag and _is_in_board(ur) and _is_same_rising(from_, ur):
    #        if _owner(bs[ur]) == turn:
    #            to[ur] = 2
    #        else:
    #            to[ur] = 1
    #    if ul_flag and _is_in_board(ul) and _is_same_declining(from_, ul):
    #        if _owner(bs[ul]) == turn:
    #            to[ul] = 2
    #        else:
    #            to[ul] = 1
    #    if dr_flag and _is_in_board(dr) and _is_same_declining(from_, dr):
    #        if _owner(bs[dr]) == turn:
    #            to[dr] = 2
    #        else:
    #            to[dr] = 1
    #    if dl_flag and _is_in_board(dl) and _is_same_rising(from_, dl):
    #        if _owner(bs[dl]) == turn:
    #            to[dl] = 2
    #        else:
    #            to[dl] = 1
    #    if not _is_in_board(ur) or bs[ur] != 0:
    #        ur_flag = False
    #    if not _is_in_board(ul) or bs[ul] != 0:
    #        ul_flag = False
    #    if not _is_in_board(dr) or bs[dr] != 0:
    #        dr_flag = False
    #    if not _is_in_board(dl) or bs[dl] != 0:
    #        dl_flag = False
    return to


def _bishop_effect(
    bs: np.ndarray, from_: int, turn: int
) -> np.ndarray:
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
            max_dis = 8 - np.min(ur_array[np.nonzero(ur_array)])
    else:
        max_dis = 7 - np.max(bs_one * ur_array)
    ur_point = from_ + 9 * max_dis
    if _is_in_board(ur_point + 9) and _is_same_rising(from_, ur_point + 9):
        ur_point += 9
    np.put(to, np.arange(ur_point, from_, -9), 1)
    if np.all(bs_one * ul_array == 0):
        if np.all(ul_array == 0):
            max_dis = 0
        else:
            max_dis = 8 - np.min(ul_array[np.nonzero(ul_array)])
    else:
        max_dis = 7 - np.max(bs_one * ul_array)
    ul_point = from_ - 7 * max_dis
    if _is_in_board(ul_point - 7) and _is_same_declining(from_, ul_point - 7):
        ul_point -= 7
    np.put(to, np.arange(ul_point, from_, 7), 1)
    if np.all(bs_one * dr_array == 0):
        if np.all(dr_array == 0):
            max_dis = 0
        else:
            max_dis = 8 - np.min(dr_array[np.nonzero(dr_array)])
    else:
            # ur_arrayの途中に駒が存在する
        max_dis = 7 - np.max(bs_one * dr_array)
    dr_point = from_ + 7 * max_dis
    if _is_in_board(dr_point + 7) and _is_same_declining(from_, dr_point + 7):
        dr_point += 7
    np.put(to, np.arange(dr_point, from_, -7), 1)
    if np.all(bs_one * dl_array == 0):
        if np.all(dl_array == 0):
            max_dis = 0
        else:
            max_dis = 8 - np.min(dl_array[np.nonzero(dl_array)])
    else:
            # ur_arrayの途中に駒が存在する
        max_dis = 7 - np.max(bs_one * dl_array)
    dl_point = from_ - 9 * max_dis
    if _is_in_board(dl_point - 9) and _is_same_declining(from_, dl_point - 9):
        dl_point -= 9
    np.put(to, np.arange(dl_point, from_, 9), 1)
    return to


def _rook_moves(bs: np.ndarray, from_: int, turn: int, pin: int) -> np.ndarray:
    to = np.zeros(64, dtype=np.int32)
    u_flag = pin == 0 or pin == 1
    d_flag = pin == 0 or pin == 1
    r_flag = pin == 0 or pin == 2
    l_flag = pin == 0 or pin == 2
    for i in range(8):
        u = from_ + 1 * (1 + i)
        d = from_ - 1 * (1 + i)
        l_ = from_ - 8 * (1 + i)
        r = from_ + 8 * (1 + i)
        if u_flag and _is_in_board(u) and _is_same_column(from_, u):
            if _owner(bs[u]) == turn:
                to[u] = 2
            else:
                to[u] = 1
        if d_flag and _is_in_board(d) and _is_same_column(from_, d):
            if _owner(bs[d]) == turn:
                to[d] = 2
            else:
                to[d] = 1
        if l_flag and _is_in_board(l_) and _is_same_row(from_, l_):
            if _owner(bs[l_]) == turn:
                to[l_] = 2
            else:
                to[l_] = 1
        if r_flag and _is_in_board(r) and _is_same_row(from_, r):
            if _owner(bs[r]) == turn:
                to[r] = 2
            else:
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


def _rook_effect(bs: np.ndarray, from_: int, turn: int) -> np.ndarray:
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
        if u_flag and _is_in_board(u) and _is_same_column(from_, u):
            if _owner(bs[u]) == turn:
                to[u] = 2
            else:
                to[u] = 1
        if d_flag and _is_in_board(d) and _is_same_column(from_, d):
            if _owner(bs[d]) == turn:
                to[d] = 2
            else:
                to[d] = 1
        if l_flag and _is_in_board(l_) and _is_same_row(from_, l_):
            if _owner(bs[l_]) == turn:
                to[l_] = 2
            else:
                to[l_] = 1
        if r_flag and _is_in_board(r) and _is_same_row(from_, r):
            if _owner(bs[r]) == turn:
                to[r] = 2
            else:
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


def _queen_moves(
    bs: np.ndarray, from_: int, turn: int, pin: int
) -> np.ndarray:
    r_move = _rook_moves(bs, from_, turn, pin)
    b_move = _bishop_moves(bs, from_, turn, pin)
    # r_moveとb_moveは共通項がないので足してよい
    return r_move + b_move


def _queen_effect(
    bs: np.ndarray, from_: int, turn: int
) -> np.ndarray:
    r_ef = _rook_effect(bs, from_, turn)
    b_ef = _bishop_effect(bs, from_, turn)
    # r_moveとb_moveは共通項がないので足してよい
    return r_ef + b_ef


def _king_moves(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    if not u:
        if _owner(bs[from_ + 1]) == turn:
            to[from_ + 1] = 2
        else:
            to[from_ + 1] = 1
        if not l_:
            if _owner(bs[from_ - 7]) == turn:
                to[from_ - 7] = 2
            else:
                to[from_ - 7] = 1
        if not r:
            if _owner(bs[from_ + 9]) == turn:
                to[from_ + 9] = 2
            else:
                to[from_ + 9] = 1
    if not l_:
        if _owner(bs[from_ - 8]) == turn:
            to[from_ - 8] = 2
        else:
            to[from_ - 8] = 1
    if not r:
        if _owner(bs[from_ + 8]) == turn:
            to[from_ + 8] = 2
        else:
            to[from_ + 8] = 1
    if not d:
        if _owner(bs[from_ - 1]) == turn:
            to[from_ - 1] = 2
        else:
            to[from_ - 1] = 1
        if not l_:
            if _owner(bs[from_ - 9]) == turn:
                to[from_ - 9] = 2
            else:
                to[from_ - 9] = 1
        if not r:
            if _owner(bs[from_ + 7]) == turn:
                to[from_ + 7] = 2
            else:
                to[from_ + 7] = 1
    return to


def _king_effect(bs: np.ndarray, from_: int, turn: int):
    to = np.zeros(64, dtype=np.int32)
    u, d, l_, r = _is_side(from_)
    if not u:
        if _owner(bs[from_ + 1]) == turn:
            to[from_ + 1] = 2
        else:
            to[from_ + 1] = 1
        if not l_:
            if _owner(bs[from_ - 7]) == turn:
                to[from_ - 7] = 2
            else:
                to[from_ - 7] = 1
        if not r:
            if _owner(bs[from_ + 9]) == turn:
                to[from_ + 9] = 2
            else:
                to[from_ + 9] = 1
    if not l_:
        if _owner(bs[from_ - 8]) == turn:
            to[from_ - 8] = 2
        else:
            to[from_ - 8] = 1
    if not r:
        if _owner(bs[from_ + 8]) == turn:
            to[from_ + 8] = 2
        else:
            to[from_ + 8] = 1
    if not d:
        if _owner(bs[from_ - 1]) == turn:
            to[from_ - 1] = 2
        else:
            to[from_ - 1] = 1
        if not l_:
            if _owner(bs[from_ - 9]) == turn:
                to[from_ - 9] = 2
            else:
                to[from_ - 9] = 1
        if not r:
            if _owner(bs[from_ + 7]) == turn:
                to[from_ + 7] = 2
            else:
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


def _piece_effect(
    bs: np.ndarray, from_: int, piece: int
) -> np.ndarray:
    if piece == 0:
        return np.zeros(64, dtype=np.int32)
    turn = (piece - 1) // 6
    p = piece % 6
    if p == 1:
        return _pawn_effects(from_, turn)
    elif p == 2:
        return _knight_effect(from_, turn)
    elif p == 3:
        return _bishop_effect(bs, from_, turn)
    elif p == 4:
        return _rook_effect(bs, from_, turn)
    elif p == 5:
        return _queen_effect(bs, from_, turn)
    else:
        return _king_effect(bs, from_, turn)


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


def _effected_positions(bs: np.ndarray, turn: int) -> np.ndarray:
    effects = np.zeros(64, dtype=np.int32)
    for i in range(64):
        piece = bs[i]
        if _owner(piece) != turn:
            continue
        effects += _piece_effect(bs, i, piece)
    return effects


def _is_check(bs: np.ndarray, turn: int, kp: int) -> bool:
    effects = _effected_positions(bs, (turn + 1) % 2)
    return effects[kp] != 0


# 王手している駒の位置も返すis_check
def _is_check2(bs: np.ndarray, turn: int, kp: int) -> Tuple[int, np.ndarray]:
    num_check = 0
    checking_piece = np.zeros(64, dtype=np.int32)
    for i in range(64):
        piece = bs[i]
        if _owner(piece) != turn and _owner(piece) != 0:
            continue
        moves = _piece_moves(bs, i, piece, np.zeros(64, dtype=np.int32))
        if moves[kp] == 1:
            num_check += 1
            checking_piece[i] = 1
    return num_check, checking_piece


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


def _up_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    # 上方向のピン
    u = king_point
    u_num = 0
    u_piece = -1
    u_flag = False
    # 自分より上は最大7マスしかない
    for i in range(7):
        # 探索位置の更新
        u += 1
        # 探索が終わっている場合はスルー
        if u_flag:
            continue
        # 盤外や同じ列にない場合は弾く
        if not _is_in_board(u) or not _is_same_column(king_point, u):
            continue
        piece = bs[u]
        # 駒がない場合は処理をしない
        if piece == 0:
            continue
        # 駒がある場合
        u_num += 1
        # 1枚目
        if u_num == 1:
            # 自分の駒なら位置を記録
            if _owner(piece) == turn:
                u_piece = u
            # 相手の駒なら関係ないので探索終了
            else:
                u_flag = True
        # 2枚目
        if u_num == 2:
            # Rook, Queenによってピン
            if turn == 0 and (piece == 10 or piece == 11):
                new_array[u_piece] = 1
            elif turn == 1 and (piece == 4 or piece == 5):
                new_array[u_piece] = 1
            # 一回通ったら探索する必要はない
            u_flag = True
    return new_array


def _down_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    d = king_point
    d_num = 0
    d_piece = -1
    d_flag = False
    for i in range(7):
        d -= 1
        if d_flag:
            continue
        if not _is_in_board(d) or not _is_same_column(king_point, d):
            continue
        piece = bs[d]
        if piece == 0:
            continue
        d_num += 1
        if d_num == 1:
            if _owner(piece) == turn:
                d_piece = d
            else:
                d_flag = True
        if d_num == 2:
            if turn == 0 and (piece == 10 or piece == 11):
                new_array[d_piece] = 1
            elif turn == 1 and (piece == 4 or piece == 5):
                new_array[d_piece] = 1
            d_flag = True
    return new_array


def _left_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    l_ = king_point
    l_num = 0
    l_piece = -1
    l_flag = False
    for i in range(7):
        l_ -= 8
        if l_flag:
            continue
        if not _is_in_board(l_) or not _is_same_row(king_point, l_):
            continue
        piece = bs[l_]
        if piece == 0:
            continue
        l_num += 1
        if l_num == 1:
            if _owner(piece) == turn:
                l_piece = l_
            else:
                l_flag = True
        if l_num == 2:
            if turn == 0 and (piece == 10 or piece == 11):
                new_array[l_piece] = 2
            elif turn == 1 and (piece == 4 or piece == 5):
                new_array[l_piece] = 2
            l_flag = True
    return new_array


def _right_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    r = king_point
    r_num = 0
    r_piece = -1
    r_flag = False
    for i in range(7):
        r += 8
        if r_flag:
            continue
        if not _is_in_board(r) or not _is_same_row(king_point, r):
            continue
        piece = bs[r]
        if piece == 0:
            continue
        r_num += 1
        if r_num == 1:
            if _owner(piece) == turn:
                r_piece = r
            else:
                r_flag = True
        if r_num == 2:
            if turn == 0 and (piece == 10 or piece == 11):
                new_array[r_piece] = 2
            elif turn == 1 and (piece == 4 or piece == 5):
                new_array[r_piece] = 2
            r_flag = True
    return new_array


def _up_right_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    ur = king_point
    ur_num = 0
    ur_piece = -1
    ur_flag = False
    for i in range(7):
        ur += 9
        if ur_flag:
            continue
        if not _is_in_board(ur) or not _is_same_rising(king_point, ur):
            continue
        piece = bs[ur]
        if piece == 0:
            continue
        ur_num += 1
        if ur_num == 1:
            if _owner(piece) == turn:
                ur_piece = ur
            else:
                ur_flag = True
        # 2枚目
        if ur_num == 2:
            # Bishop, Queenによってピン
            if turn == 0 and (piece == 9 or piece == 11):
                new_array[ur_piece] = 3
            elif turn == 1 and (piece == 3 or piece == 5):
                new_array[ur_piece] = 3
            ur_flag = True
    return new_array


def _down_left_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    dl = king_point
    dl_num = 0
    dl_piece = -1
    dl_flag = False
    for i in range(7):
        dl -= 9
        if dl_flag:
            continue
        if not _is_in_board(dl) or not _is_same_rising(king_point, dl):
            continue
        piece = bs[dl]
        if piece == 0:
            continue
        dl_num += 1
        if dl_num == 1:
            if _owner(piece) == turn:
                dl_piece = dl
            else:
                dl_flag = True
        if dl_num == 2:
            if turn == 0 and (piece == 9 or piece == 11):
                new_array[dl_piece] = 3
            elif turn == 1 and (piece == 3 or piece == 5):
                new_array[dl_piece] = 3
            dl_flag = True
    return new_array


def _up_left_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    ul = king_point
    ul_num = 0
    ul_piece = -1
    ul_flag = False
    for i in range(7):
        ul -= 7
        if ul_flag:
            continue
        if not _is_in_board(ul) or not _is_same_row(king_point, ul):
            continue
        piece = bs[ul]
        if piece == 0:
            continue
        ul_num += 1
        if ul_num == 1:
            if _owner(piece) == turn:
                ul_piece = ul
            else:
                ul_flag = True
        if ul_num == 2:
            if turn == 0 and (piece == 9 or piece == 11):
                new_array[ul_piece] = 4
            elif turn == 1 and (piece == 3 or piece == 5):
                new_array[ul_piece] = 4
            ul_flag = True
    return new_array


def _down_right_pin(
    bs: np.ndarray, turn: int, king_point: int, array: np.ndarray
) -> np.ndarray:
    new_array = array
    dr = king_point
    dr_num = 0
    dr_piece = -1
    dr_flag = False
    for i in range(7):
        dr += 7
        if dr_flag:
            continue
        if not _is_in_board(dr) or not _is_same_row(king_point, dr):
            continue
        piece = bs[dr]
        if piece == 0:
            continue
        dr_num += 1
        if dr_num == 1:
            if _owner(piece) == turn:
                dr_piece = dr
            else:
                dr_flag = True
        if dr_num == 2:
            if turn == 0 and (piece == 9 or piece == 11):
                new_array[dr_piece] = 4
            elif turn == 1 and (piece == 3 or piece == 5):
                new_array[dr_piece] = 4
            dr_flag = True
    return new_array


def _pin(state: ChessState, kp: int):
    bs = _board_status(state)
    turn = state.turn
    pins = np.zeros(64, dtype=np.int32)
    pins = _up_pin(bs, turn, kp, pins)
    pins = _up_left_pin(bs, turn, kp, pins)
    pins = _up_right_pin(bs, turn, kp, pins)
    pins = _left_pin(bs, turn, kp, pins)
    pins = _right_pin(bs, turn, kp, pins)
    pins = _down_pin(bs, turn, kp, pins)
    pins = _down_left_pin(bs, turn, kp, pins)
    pins = _down_right_pin(bs, turn, kp, pins)
    return pins
