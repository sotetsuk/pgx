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
    for i in range(13):
        s.board[i][action.to] = 0
    p = action.piece
    # プロモーションの場合
    if _is_promotion(action):
        # Queenの場合
        if action.promotion == 0:
            p += 4
        # Rook
        elif action.promotion == 1:
            p += 3
        # Knight
        elif action.promotion == 2:
            p += 1
        # Bishop
        elif action.promotion == 3:
            p += 2
    s.board[p][action.to] = 1
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
    if action.piece % 6 == 1 and abs(action.from_ - action.to) == 2:
        s.en_passant = action.to
    else:
        s.en_passant = -1
    return s


def _board_status(state: ChessState):
    bs = np.zeros(64, dtype=np.int32)
    for i in range(64):
        bs[i] = _piece_type(state, i)
    return bs
