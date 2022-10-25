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
    is_promote: int


# 盤面のdataclass
@dataclass
class ChessState:
    # turn 先手番なら0 後手番なら1
    turn: int = 0
    # board 盤面の駒。
    # 空白,BPawn,BKnight,BBishop,BRook,BQueen,BKing,WPawn,WKnight,WBishop,WRook,WQueen,WKing
    # の順で駒がどの位置にあるかをone_hotで記録
    board: np.ndarray = np.zeros((13, 64), dtype=np.int32)
    # en_passant 直前にポーンが2マス動いていた場合、通過した地点でもそのポーンは取れる
    # 直前の動きがポーンを2マス進めるものだった場合は、位置を記録
    # 普段は-1
    en_passant: int = -1


# intのactionを向きと距離に変換
def _separate_int_action(action: int) -> Tuple[int, int]:
    if action <= 55:
        return action // 8, action % 8 + 1
    # Knightのaction
    # 8~15に割り振る
    if action <= 63:
        return action - 48, 1
    # promotionのaction
    # 二つ目の返り値をpromotionの駒種にする
    # 直進
    if action <= 66:
        return 16, action - 62
    # 右斜め
    if action <= 69:
        return 17, action - 65
    # 左斜め
    if action <= 72:
        return 18, action - 68


def _direction_to_dif(direction: int) -> int:
    # 前
    if direction == 0:
        return -1
    # 後ろ
    if direction == 1:
        return 1
    # 右
    if direction == 2:
        return -8
    # 左
    if direction == 3:
        return 8
    # 右上
    if direction == 4:
        return -9
    # 左下
    if direction == 5:
        return 9
    # 右下
    if direction == 6:
        return -7
    # 左上
    if direction == 7:
        return 7
    # Knightのdif
    if direction == 8:
        return 6
    if direction == 9:
        return -10
    if direction == 10:
        return -17
    if direction == 11:
        return -15
    if direction == 12:
        return -6
    if direction == 13:
        return 10
    if direction == 14:
        return 17
    if direction == 15:
        return 15


def convert_to(action: int, turn: int):
    from_ = action % 64
    dir, dis = _separate_int_action(action // 64)
    if dir <= 15:
        return from_ + _direction_to_dif(dir) * dis
    if dir == 16:
        if turn == 0:
            return from_ - 1
        else:
            return from_ + 1
    if dir == 17:
        if turn == 0:
            return from_ - 9
        else:
            return from_ - 7
    if dir == 18:
        if turn == 0:
            return from_ + 7
        else:
            return from_ + 9


def int_to_action(state: ChessState, action: int):
    from_ = action % 64
    to = convert_to(action, state.turn)
    return ChessAction(_piece_type(state, from_), from_, to)


def _piece_type(state: ChessState, position: int):
    return state.board[:, position].argmax()
