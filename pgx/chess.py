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
    is_promote: int = 0


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


# actionがプロモーションかどうか
def _is_promotion(action: ChessAction):
    if action.piece == 1 and action.to % 8 == 7:
        return True
    if action.piece == 7 and action.to % 8 == 0:
        return True
    return False


def _move(state: ChessState, action: ChessAction) -> ChessState:
    s = copy.deepcopy(state)
    s.board[action.piece][action.from_] = 0
    s.board[0][action.from_] = 1
    s.board[0: 13][action.to] = 0
    p = action.piece
    # プロモーションの場合
    if _is_promotion(action):
        # Queenの場合
        if action.is_promote == 0:
            p += 4
        # Rook
        elif action.is_promote == 1:
            p += 3
        # Knight
        elif action.is_promote == 2:
            p += 1
        # Bishop
        elif action.is_promote == 3:
            p += 2
    s.board[p][action.to] = 1
    # 各種フラグの更新
    return s
