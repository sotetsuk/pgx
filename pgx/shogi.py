import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# 指し手のdataclass
@dataclass
class ShogiAction:
    # 上の3つは移動と駒打ちで共用
    # 下の3つは移動でのみ使用
    # 駒打ちかどうか
    is_drop: bool
    # piece: 動かした(打った)駒の種類
    piece: int
    # final: 移動後の座標
    to: int
    # 移動前の座標
    from_: int = 0
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: int = 0
    # is_promote: 駒を成るかどうかの判定
    is_promote: bool = False


# 盤面のdataclass
@dataclass
class ShogiState:
    # turn 先手番なら0 後手番なら1
    turn: int = 0
    # board 盤面の駒。
    # 空白,先手歩,先手香車,先手桂馬,先手銀,先手角,先手飛車,先手金,先手玉,先手と,先手成香,先手成桂,先手成銀,先手馬,先手龍,
    # 後手歩,後手香車,後手桂馬,後手銀,後手角,後手飛車,後手金,後手玉,後手と,後手成香,後手成桂,後手成銀,後手馬,後手龍
    # の順で駒がどの位置にあるかをone_hotで記録
    board: np.ndarray = np.zeros((29, 81), dtype=np.int32)
    # hand 持ち駒。先手歩,先手香車,先手桂馬,先手銀,先手角,先手飛車,先手金,後手歩,後手香車,後手桂馬,後手銀,後手角,後手飛車,後手金
    # の14種の値を増減させる
    hand: np.ndarray = np.zeros(14, dtype=np.int32)
    # legal_actions_black/white: 自殺手や王手放置などの手も含めた合法手の一覧
    # move/dropによって変化させる
    # もしかしたら香車や大駒の動きは別で追加した方が良いかも？
    # legal_actions_black: np.ndarray = np.zeros(180, dtype=np.int32)
    # legal_actions_white: np.ndarray = np.zeros(180, dtype=np.int32)
    # checked: ターンプレイヤーの王に王手がかかっているかどうか
    is_check: int = 0
    # checking_piece: ターンプレイヤーに王手をかけている駒の座標
    checking_piece: np.ndarray = np.zeros(81, dtype=np.int32)


# BLACK/WHITE/(NONE)_○○_MOVEは22にいるときの各駒の動き
# 端にいる場合は対応するところに0をかけていけないようにする
# BLACK_PAWN_MOVE = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
# WHITE_PAWN_MOVE = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
# BLACK_GOLD_MOVE = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
# WHITE_GOLD_MOVE = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
# ROOK_MOVE = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
# BISHOP_MOVE = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]])
# KING_MOVE = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])


def init():
    board = _make_init_board()
    return ShogiState(board=board)


def _make_init_board():
    array = np.zeros((29, 81), dtype=np.int32)
    for i in range(81):
        if i % 9 == 6:
            # 先手歩
            p = 1
        elif i == 8 or i == 80:
            # 先手香車
            p = 2
        elif i == 17 or i == 71:
            # 先手桂馬
            p = 3
        elif i == 26 or i == 62:
            # 先手銀
            p = 4
        elif i == 70:
            # 先手角
            p = 5
        elif i == 16:
            # 先手飛車
            p = 6
        elif i == 35 or i == 53:
            # 先手金
            p = 7
        elif i == 44:
            # 先手玉
            p = 8
        elif i % 9 == 2:
            # 後手歩
            p = 15
        elif i == 72 or i == 0:
            # 後手香車
            p = 16
        elif i == 63 or i == 9:
            # 後手桂馬
            p = 17
        elif i == 54 or i == 18:
            # 後手銀
            p = 18
        elif i == 10:
            # 後手角
            p = 19
        elif i == 64:
            # 後手飛車
            p = 20
        elif i == 45 or i == 27:
            # 後手金
            p = 21
        elif i == 36:
            # 後手玉
            p = 22
        else:
            # 空白
            p = 0
        array[p][i] = 1
    return array


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
def _dlshogi_action(direction: int, to: int) -> int:
    return direction * 81 + to


# from, toが同じ列にあるかどうか
def _is_same_column(_from: int, to: int) -> bool:
    return _from // 9 == to // 9


# from, toが同じ行にあるかどうか
def _is_same_row(_from: int, to: int) -> bool:
    return _from % 9 == to % 9


# from, toが右肩上がりの斜め方向で同じ筋にあるか
def _is_same_rising(_from: int, to: int) -> bool:
    return _from // 9 - _from % 9 == to // 9 - to % 9


# from, toが右肩下がりの斜め方向で同じ筋にあるか
def _is_same_declining(_from: int, to: int) -> bool:
    return _from // 9 + _from % 9 == to // 9 + to % 9


# fromの座標とtoの座標からdirを生成
def _point_to_direction(_from: int, to: int, promote: bool, turn: int) -> int:
    direction = -1
    dis = to - _from
    # 後手番の動きは反転させる
    if turn == 1:
        dis = -dis
    # UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT, UP_PROMOTE...
    # の順でdirを割り振る
    # PROMOTEの場合は+10する処理を入れる
    if _is_same_column(_from, to) and dis < 0:
        direction = 0
    if _is_same_declining(_from, to) and dis > 0:
        direction = 1
    if _is_same_rising(_from, to) and dis < 0:
        direction = 2
    if _is_same_row(_from, to) and dis > 0:
        direction = 3
    if _is_same_row(_from, to) and dis < 0:
        direction = 4
    if _is_same_column(_from, to) and dis > 0:
        direction = 5
    if _is_same_rising(_from, to) and dis > 0:
        direction = 6
    if _is_same_declining(_from, to) and dis < 0:
        direction = 7
    if dis == 7:
        direction = 8
    if dis == -11:
        direction = 9
    if promote:
        direction += 10
    return direction


# 打った駒の種類をdirに変換
def _hand_to_direction(piece: int) -> int:
    # 移動のdirはPROMOTE_UP_RIGHT2の19が最大なので20以降に配置
    # 20: 先手歩 21: 先手香車... 33: 後手金　に対応させる
    if piece <= 14:
        return 19 + piece
    else:
        return 5 + piece


# AnimalShogiActionをdlshogiのint型actionに変換
def _action_to_dlaction(action: ShogiAction, turn: int) -> int:
    if action.is_drop:
        return _dlshogi_action(_hand_to_direction(action.piece), action.to)
    else:
        return _dlshogi_action(
            _point_to_direction(
                action.from_, action.to, action.is_promote, turn
            ),
            action.to,
        )


# dlshogiのint型actionをdirectionとtoに分解
def _separate_dlaction(action: int) -> Tuple[int, int]:
    # direction, to の順番
    return action // 81, action % 81


# directionからfromがtoからどれだけ離れてるかを返す
def _direction_to_dif(direction: int, turn: int) -> int:
    dif = 0
    if direction % 10 == 0:
        dif = -1
    if direction % 10 == 1:
        dif = 8
    if direction % 10 == 2:
        dif = -10
    if direction % 10 == 3:
        dif = 9
    if direction % 10 == 4:
        dif = -9
    if direction % 10 == 5:
        dif = 1
    if direction % 10 == 6:
        dif = 10
    if direction % 10 == 7:
        dif = -8
    if direction % 10 == 8:
        dif = 7
    if direction % 10 == 9:
        dif = -11
    if turn == 0:
        return dif
    else:
        return -dif


# directionとto,stateから大駒含めた移動のfromの位置を割り出す
# 成りの移動かどうかも返す
def _direction_to_from(direction: int, to: int, state: ShogiState) -> Tuple[int, bool]:
    dif = _direction_to_dif(direction, state.turn)
    f = to
    _from = -1
    for i in range(9):
        f -= dif
        if 80 >= f >= 0 and _from == -1 and _piece_type(state, f) != 0:
            _from = f
    if direction >= 10:
        return _from, True
    else:
        return _from, False


def _direction_to_hand(direction: int) -> int:
    if direction <= 26:
        return direction - 19
    else:
        return direction - 5


def _piece_type(state: ShogiState, point: int) -> int:
    return state.board[:, point].argmax()


def _dlaction_to_action(action: int, state: ShogiState) -> ShogiAction:
    direction, to = _separate_dlaction(action)
    if direction <= 19:
        # 駒の移動
        _from, is_promote = _direction_to_from(direction, to, state)
        piece = _piece_type(state, _from)
        captured = _piece_type(state, to)
        return ShogiAction(False, piece, to, _from, captured, is_promote)
    else:
        # 駒打ち
        piece = _direction_to_hand(direction)
        return ShogiAction(True, piece, to)
