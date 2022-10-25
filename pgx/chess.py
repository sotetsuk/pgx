import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# 指し手のdataclass
@dataclass
class ChessAction:
    # piece: 動かした駒の種類
    piece: int
    # to: 移動後の座標
    to: int
    # 移動前の座標
    from_: int
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: int
    # promotion: プロモーションで成る駒を選ぶ
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
