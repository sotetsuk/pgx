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
    legal_actions_black: np.ndarray = np.zeros(2754, dtype=np.int32)
    legal_actions_white: np.ndarray = np.zeros(2754, dtype=np.int32)


def init():
    board = _make_init_board()
    state = ShogiState(board=board)
    return _init_legal_actions(state)


def step(state: ShogiState, action: int) -> Tuple[ShogiState, int, bool]:
    # state, 勝敗判定,終了判定を返す
    s = copy.deepcopy(state)
    legal_actions = _legal_actions(s)
    _action = _dlaction_to_action(action, s)
    # actionのfromが盤外の場合は非合法手なので負け
    if not _is_in_board(_action.from_):
        print("an illegal action")
        return s, _turn_to_reward(_another_color(s)), True
    # legal_actionsにactionがない場合、そのactionは非合法手
    if legal_actions[action] == 0:
        print("an illegal action2")
        return s, _turn_to_reward(_another_color(s)), True
    # 合法手の場合
    # 駒打ち
    if _action.is_drop:
        s = _update_legal_drop_actions(s, _action)
        s = _drop(s, _action)
        print("drop: piece =", _action.piece, ", to =", _action.to)
    # 駒の移動
    else:
        s = _update_legal_move_actions(s, _action)
        s = _move(s, _action)
        print("move: piece =", _action.piece, ", to =", _action.to)
    # 王手がかかったままの場合、王手放置またｈ自殺手で負け
    if _is_check(s):
        print("check is remained")
        return s, _turn_to_reward(_another_color(s)), True
    # その他の反則
    if _is_double_pawn(s):
        print("two pawns in the same file")
        return s, _turn_to_reward(_another_color(s)), True
    if _is_stuck(s):
        print("some pieces are stuck")
        return s, _turn_to_reward(_another_color(s)), True
    s.turn = _another_color(s)
    # 相手に合法手がない場合→詰み
    if _is_mate(s):
        # actionのis_dropがTrueかつpieceが歩の場合、打ち歩詰めで負け
        if _action.is_drop and (_action.piece == 1 or _action.piece == 15):
            print("mate by dropped pawn")
            return s, _turn_to_reward(s.turn), True
        # そうでなければ普通の詰みで勝ち
        else:
            print("mate")
            return s, _turn_to_reward(_another_color(s)), True
    else:
        return s, 0, False


def _turn_to_reward(turn: int) -> int:
    if turn == 0:
        return 1
    else:
        return -1


def _make_init_board() -> np.ndarray:
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


def _pawn_move(turn: int) -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    if turn == 0:
        array[4][3] = 1
    else:
        array[4][5] = 1
    return array


def _knight_move(turn: int) -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    if turn == 0:
        array[3][2] = 1
        array[5][2] = 1
    else:
        array[3][6] = 1
        array[5][6] = 1
    return array


def _silver_move(turn: int) -> np.ndarray:
    array = _pawn_move(turn)
    array[3][3] = 1
    array[3][5] = 1
    array[5][3] = 1
    array[5][5] = 1
    return array


def _gold_move(turn: int) -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    if turn == 0:
        array[3][3] = 1
        array[5][3] = 1
    else:
        array[3][5] = 1
        array[5][5] = 1
    array[4][3] = 1
    array[3][4] = 1
    array[4][5] = 1
    array[5][4] = 1
    return array


def _king_move() -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            array[3 + i][3 + j] = 1
    array[4][4] = 0
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
    if dis == 7 and not _is_same_column(_from, to):
        direction = 8
    if dis == -11 and not _is_same_column(_from, to):
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
        return 12 + piece


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
def _direction_to_from(
    direction: int, to: int, state: ShogiState
) -> Tuple[int, bool]:
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
        # direction:20が先手の歩（pieceの1）に対応
        return direction - 19
    else:
        # direction:27が後手の歩（pieceの15）に対応
        return direction - 12


def _piece_type(state: ShogiState, point: int):
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


# 手番側でない色を返す
def _another_color(state: ShogiState) -> int:
    return (state.turn + 1) % 2


# pointが盤面内かどうか
def _is_in_board(point: int) -> bool:
    return 0 <= point <= 80


# 相手の駒を同じ種類の自分の駒に変換する
def _convert_piece(piece: int) -> int:
    if piece == 0:
        return -1
    p = (piece + 14) % 28
    if p == 0:
        return 28
    else:
        return p


# 駒から持ち駒への変換
# 先手歩が0、後手金が13
def _piece_to_hand(piece: int) -> int:
    if piece % 14 == 0 or piece % 14 >= 9:
        p = piece - 8
    else:
        p = piece
    if p < 15:
        return p - 1
    else:
        return p - 8


# ある駒の持ち主を返す
def _owner(piece: int) -> int:
    if piece == 0:
        return 2
    return (piece - 1) // 14


# 盤面のどこに何の駒があるかをnp.arrayに移したもの
# 同じ座標に複数回piece_typeを使用する場合はこちらを使った方が良い
def _board_status(state: ShogiState) -> np.ndarray:
    board = np.zeros(81, dtype=np.int32)
    for i in range(81):
        board[i] = _piece_type(state, i)
    return board


# 駒の持ち主の判定
def _pieces_owner(state: ShogiState) -> np.ndarray:
    board = np.zeros(81, dtype=np.int32)
    for i in range(81):
        piece = _piece_type(state, i)
        board[i] = _owner(piece)
    return board


# 駒の移動の盤面変換
def _move(
    state: ShogiState,
    action: ShogiAction,
) -> ShogiState:
    s = copy.deepcopy(state)
    s.board[action.piece][action.from_] = 0
    s.board[0][action.from_] = 1
    s.board[action.captured][action.to] = 0
    if action.is_promote:
        s.board[action.piece + 8][action.to] = 1
    else:
        s.board[action.piece][action.to] = 1
    if action.captured != 0:
        s.hand[_piece_to_hand(_convert_piece(action.captured))] += 1
    return s


def _drop(state: ShogiState, action: ShogiAction) -> ShogiState:
    s = copy.deepcopy(state)
    s.hand[_piece_to_hand(action.piece)] -= 1
    s.board[action.piece][action.to] = 1
    s.board[0][action.to] = 0
    return s


def _is_side(point: int) -> Tuple[bool, bool, bool, bool]:
    is_up = point % 9 == 0
    is_down = point % 9 == 8
    is_left = point >= 72
    is_right = point <= 8
    return is_up, is_down, is_left, is_right


# 桂馬用
def _is_second_line(point: int) -> Tuple[bool, bool]:
    u = point % 9 <= 1
    d = point % 9 >= 7
    return u, d


# point(0~80)を座標((0, 0)~(8, 8))に変換
def _point_to_location(point: int) -> Tuple[int, int]:
    return point // 9, point % 9


def _cut_outside(array: np.ndarray, point: int) -> np.ndarray:
    new_array = copy.deepcopy(array)
    u, d, l, r = _is_side(point)
    u2, d2 = _is_second_line(point)
    # (4, 4)での動きを基準にはみ出すところをカットする
    if u:
        new_array[:, 3] *= 0
    if d:
        new_array[:, 5] *= 0
    if r:
        new_array[3, :] *= 0
    if l:
        new_array[5, :] *= 0
    if u2:
        new_array[:, 2] *= 0
    if d2:
        new_array[:, 6] *= 0
    return new_array


def _action_board(array: np.ndarray, point: int) -> np.ndarray:
    new_array = copy.deepcopy(array)
    y, t = _point_to_location(point)
    new_array = _cut_outside(new_array, point)
    return np.roll(new_array, (y - 4, t - 4), axis=(0, 1)).reshape(81)


def _lance_move(state: ShogiState, point: int, turn: int) -> np.ndarray:
    array = np.zeros(81, dtype=np.int32)
    pieces_owner = _pieces_owner(state)
    flag = True
    for i in range(8):
        if turn == 0:
            p = point - 1 - i
        else:
            p = point + 1 + i
        # pが盤内、かつ同じ列、かつフラグが折れていない場合は利きが通っている
        if _is_in_board(p) and _is_same_column(point, p) and flag:
            array[p] = 1
        # pが盤外、または空白でない場合はフラグを折ってそれより奥に到達できないようにする
        if not _is_in_board(p) or pieces_owner[p] != 2:
            flag = False
    return array


def _bishop_move(state: ShogiState, point: int) -> np.ndarray:
    array = np.zeros(81, dtype=np.int32)
    pieces_owner = _pieces_owner(state)
    ur_flag = True
    ul_flag = True
    dr_flag = True
    dl_flag = True
    for i in range(8):
        ur = point - 10 * (1 + i)
        ul = point + 8 * (1 + i)
        dr = point - 8 * (1 + i)
        dl = point + 10 * (1 + i)
        if _is_in_board(ur) and _is_same_rising(point, ur) and ur_flag:
            array[ur] = 1
        if _is_in_board(ul) and _is_same_declining(point, ul) and ul_flag:
            array[ul] = 1
        if _is_in_board(dr) and _is_same_declining(point, dr) and dr_flag:
            array[dr] = 1
        if _is_in_board(dl) and _is_same_rising(point, dl) and dl_flag:
            array[dl] = 1
        if not _is_in_board(ur) or pieces_owner[ur] != 2:
            ur_flag = False
        if not _is_in_board(ul) or pieces_owner[ul] != 2:
            ul_flag = False
        if not _is_in_board(dr) or pieces_owner[dr] != 2:
            dr_flag = False
        if not _is_in_board(dl) or pieces_owner[dl] != 2:
            dl_flag = False
    return array


def _rook_move(state: ShogiState, point: int) -> np.ndarray:
    array = np.zeros(81, dtype=np.int32)
    pieces_owner = _pieces_owner(state)
    u_flag = True
    d_flag = True
    r_flag = True
    l_flag = True
    for i in range(8):
        u = point - 1 * (1 + i)
        d = point + 1 * (1 + i)
        r = point - 9 * (1 + i)
        l_ = point + 9 * (1 + i)
        if _is_in_board(u) and _is_same_column(point, u) and u_flag:
            array[u] = 1
        if _is_in_board(d) and _is_same_column(point, d) and d_flag:
            array[d] = 1
        if _is_in_board(r) and _is_same_row(point, r) and r_flag:
            array[r] = 1
        if _is_in_board(l_) and _is_same_row(point, l_) and l_flag:
            array[l_] = 1
        if not _is_in_board(u) or pieces_owner[u] != 2:
            u_flag = False
        if not _is_in_board(d) or pieces_owner[d] != 2:
            d_flag = False
        if not _is_in_board(r) or pieces_owner[r] != 2:
            r_flag = False
        if not _is_in_board(l_) or pieces_owner[l_] != 2:
            l_flag = False
    return array


def _horse_move(state: ShogiState, point: int) -> np.ndarray:
    array = _bishop_move(state, point)
    u, d, r, l_ = _is_side(point)
    if not u:
        array[point - 1] = 1
    if not d:
        array[point + 1] = 1
    if not r:
        array[point - 9] = 1
    if not l_:
        array[point + 9] = 1
    return array


def _dragon_move(state: ShogiState, point: int) -> np.ndarray:
    array = _rook_move(state, point)
    u, d, r, l_ = _is_side(point)
    if not u:
        if not r:
            array[point - 10] = 1
        if not l_:
            array[point + 8] = 1
    if not d:
        if not r:
            array[point - 8] = 1
        if not l_:
            array[point + 10] = 1
    return array


# 駒種と位置から到達できる場所を返す
def _piece_moves(state: ShogiState, piece: int, point: int) -> np.ndarray:
    turn = _owner(piece)
    piece_type = piece % 14
    # 歩の動き
    if piece_type == 1:
        return _action_board(_pawn_move(turn), point)
    # 香車の動き
    if piece_type == 2:
        return _lance_move(state, point, turn)
    # 桂馬の動き
    if piece_type == 3:
        return _action_board(_knight_move(turn), point)
    # 銀の動き
    if piece_type == 4:
        return _action_board(_silver_move(turn), point)
    # 角の動き
    if piece_type == 5:
        return _bishop_move(state, point)
    # 飛車の動き
    if piece_type == 6:
        return _rook_move(state, point)
    # 金および成金の動き
    if piece_type == 7 or 9 <= piece_type <= 12:
        return _action_board(_gold_move(turn), point)
    # 玉の動き
    if piece_type == 8:
        return _action_board(_king_move(), point)
    # 馬の動き
    if piece_type == 13:
        return _horse_move(state, point)
    # 龍の動き
    # piece_typeが0になってしまっているのでpieceで判別
    if piece == 14 or piece == 28:
        return _dragon_move(state, point)
    return np.zeros(81, dtype=np.int32)


# 小駒のactionのみを返すpiece_moves
def _small_piece_moves(piece: int, point: int) -> np.ndarray:
    turn = _owner(piece)
    piece_type = piece % 14
    # 歩の動き
    if piece_type == 1:
        return _action_board(_pawn_move(turn), point)
    # 桂馬の動き
    if piece_type == 3:
        return _action_board(_knight_move(turn), point)
    # 銀の動き
    if piece_type == 4:
        return _action_board(_silver_move(turn), point)
    # 金および成金の動き
    if piece_type == 7 or 9 <= piece_type <= 12:
        return _action_board(_gold_move(turn), point)
    # 玉の動き
    if piece_type == 8:
        return _action_board(_king_move(), point)
    return np.zeros(81, dtype=np.int32)


# 敵陣かどうか
def _is_enemy_zone(turn: int, point: int) -> bool:
    if turn == 0:
        return point % 9 <= 2
    else:
        return point % 9 >= 6


def _can_promote(piece: int, _from: int, to: int) -> bool:
    # pieceが飛車以下でないと成れない
    if piece % 14 > 6 or piece % 14 == 0:
        return False
    # _fromとtoのどちらかが敵陣であれば成れる
    return _is_enemy_zone(_owner(piece), _from) or _is_enemy_zone(
        _owner(piece), to
    )


def _create_actions(piece: int, _from: int, to: np.ndarray) -> np.ndarray:
    actions = np.zeros(2754, dtype=np.int32)
    for i in range(81):
        if to[i] == 0:
            continue
        if _can_promote(piece, _from, i):
            pro_dir = _point_to_direction(_from, i, True, _owner(piece))
            pro_act = _dlshogi_action(pro_dir, i)
            actions[pro_act] = 1
        normal_dir = _point_to_direction(_from, i, False, _owner(piece))
        normal_act = _dlshogi_action(normal_dir, i)
        actions[normal_act] = 1
    return actions


def _create_piece_actions(piece: int, _from: int) -> np.ndarray:
    moves = _small_piece_moves(piece, _from)
    return _create_actions(piece, _from, moves)


# actionを追加する
def _add_action(add_array: np.ndarray, origin_array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(origin_array)
    for i in range(2754):
        if add_array[i] == 1:
            new_array[i] = 1
    return new_array


# 駒の種類と位置から生成できるactionのフラグを立てる
def _add_move_actions(piece: int, _from: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(piece, _from)
    for i in range(2754):
        if actions[i] == 1:
            new_array[i] = 1
    return new_array


# 駒の種類と位置から生成できるactionのフラグを折る
def _filter_move_actions(
    piece: int, _from: int, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(piece, _from)
    for i in range(2754):
        if actions[i] == 1:
            new_array[i] = 0
    return new_array


# 駒打ちのactionを追加する
def _add_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(81):
        action = _dlshogi_action(direction, i)
        new_array[action] = 1
    return new_array


# 駒打ちのactionのフラグを折る
def _filter_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(81):
        action = _dlshogi_action(direction, i)
        new_array[action] = 0
    return new_array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は使わないがlegal_actionsが設定されていない場合に使用
def _init_legal_actions(state: ShogiState) -> ShogiState:
    s = copy.deepcopy(state)
    bs = _board_status(s)
    # 移動の追加
    for i in range(81):
        piece = bs[i]
        if piece == 0:
            continue
        if piece <= 14:
            s.legal_actions_black = _add_move_actions(
                piece, i, s.legal_actions_black
            )
        else:
            s.legal_actions_white = _add_move_actions(
                piece, i, s.legal_actions_white
            )
    # 駒打ちの追加
    for i in range(7):
        if s.hand[i] != 0:
            s.legal_actions_black = _add_drop_actions(
                1 + i, s.legal_actions_black
            )
        if s.hand[i + 7] != 0:
            s.legal_actions_white = _add_drop_actions(
                15 + i, s.legal_actions_white
            )
    return s


# 駒の移動によるlegal_actionsの更新
def _update_legal_move_actions(
    state: ShogiState, action: ShogiAction
) -> ShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
        enemy_actions = s.legal_actions_white
    else:
        player_actions = s.legal_actions_white
        enemy_actions = s.legal_actions_black
    # 元の位置にいたときのフラグを折る
    new_player_actions = _filter_move_actions(
        action.piece, action.from_, player_actions
    )
    new_enemy_actions = enemy_actions
    # 移動後の位置からの移動のフラグを立てる
    new_piece = action.piece
    if action.is_promote:
        new_piece += 8
    new_player_actions = _add_move_actions(
        new_piece, action.to, new_player_actions
    )
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if action.captured != 0:
        new_enemy_actions = _filter_move_actions(
            action.captured, action.to, new_enemy_actions
        )
        captured = _convert_piece(action.captured)
        # 成駒の場合成る前の駒に変換
        if captured % 14 == 0 or captured % 14 >= 9:
            captured -= 8
        new_player_actions = _add_drop_actions(captured, new_player_actions)
    if s.turn == 0:
        s.legal_actions_black = new_player_actions
        s.legal_actions_white = new_enemy_actions
    else:
        s.legal_actions_black = new_enemy_actions
        s.legal_actions_white = new_player_actions
    return s


# 駒打ちによるlegal_actionsの更新
def _update_legal_drop_actions(
    state: ShogiState, action: ShogiAction
) -> ShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
    else:
        player_actions = s.legal_actions_white
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = _add_move_actions(
        action.piece, action.to, player_actions
    )
    # 持ち駒がもうない場合、その駒を打つフラグを折る
    if s.hand[_piece_to_hand(action.piece)] == 1:
        new_player_actions = _filter_drop_actions(
            action.piece, new_player_actions
        )
    if s.turn == 0:
        s.legal_actions_black = new_player_actions
    else:
        s.legal_actions_white = new_player_actions
    return s


# 自分の駒がある位置への移動を除く
def _filter_my_piece_move_actions(
    turn: int, owner: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    for i in range(81):
        if owner[i] != turn:
            continue
        for j in range(20):
            new_array[81 * j + i] = 0
    return new_array


# 駒がある地点への駒打ちを除く
def _filter_occupied_drop_actions(
    turn: int, owner: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    for i in range(81):
        if owner[i] == 2:
            continue
        for j in range(7):
            new_array[81 * (j + 20 + 7 * turn) + i] = 0
    return new_array


# boardのlegal_actionsを利用して合法手を生成する
# 大駒や香車の利きはboardのlegal_actionsに追加していないので、ここで追加する
# 自殺手や反則手はここでは除かない
def _legal_actions(state: ShogiState) -> np.ndarray:
    if state.turn == 0:
        action_array = copy.deepcopy(state.legal_actions_black)
    else:
        action_array = copy.deepcopy(state.legal_actions_white)
    own = _pieces_owner(state)
    for i in range(81):
        piece = _piece_type(state, i)
        # 香車の動きを追加
        if piece == 2 + 14 * state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _lance_move(state, i, state.turn)),
                action_array,
            )
        # 角の動きを追加
        if piece == 5 + 14 * state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _bishop_move(state, i)), action_array
            )
        # 飛車の動きを追加
        if piece == 6 + 14 * state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _rook_move(state, i)), action_array
            )
        # 馬の動きを追加
        if piece == 13 + 14 * state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _horse_move(state, i)), action_array
            )
        # 龍の動きを追加
        if piece == 14 + 14 * state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _dragon_move(state, i)), action_array
            )
    # 自分の駒がある位置への移動actionを除く
    action_array = _filter_my_piece_move_actions(state.turn, own, action_array)
    # 駒がある地点への駒打ちactionを除く
    action_array = _filter_occupied_drop_actions(state.turn, own, action_array)
    return action_array


# 王手判定
def _is_check(state: ShogiState) -> bool:
    is_check = False
    king_point = state.board[8 + 14 * state.turn, :].argmax()
    bs = _board_status(state)
    for i in range(81):
        piece = bs[i]
        if _owner(piece) != _another_color(state):
            continue
        if _piece_moves(state, piece, i)[king_point] == 1:
            is_check = True
    return is_check


# 二歩判定
# 手番側でチェックする
def _is_double_pawn(state: ShogiState) -> bool:
    is_double_pawn = False
    bs = _board_status(state)
    for i in range(9):
        num_pawn = 0
        for j in range(9):
            if bs[9 * i + j] == 1 + state.turn * 14:
                num_pawn += 1
        if num_pawn >= 2:
            is_double_pawn = True
    return is_double_pawn


# 行き所のない駒判定
def _is_stuck(state: ShogiState) -> bool:
    is_stuck = False
    bs = _board_status(state)
    for i in range(9):
        for j in range(9):
            piece = bs[9 * i + j]
            if (
                state.turn == 0
                and (piece == 1 or piece == 2 or piece == 3)
                and j == 0
            ):
                is_stuck = True
            if state.turn == 0 and piece == 3 and j == 1:
                is_stuck = True
            if (
                state.turn == 1
                and (piece == 15 or piece == 16 or piece == 17)
                and j == 8
            ):
                is_stuck = True
            if state.turn == 1 and piece == 17 and j == 7:
                is_stuck = True
    return is_stuck


def _is_mate(state: ShogiState) -> bool:
    # 王手がかかっていない場合は無視
    if not _is_check(state):
        return False
    is_mate = True
    player_actions = _legal_actions(state)
    for i in range(2754):
        if player_actions[i] == 0:
            continue
        action = _dlaction_to_action(i, state)
        if action.is_drop:
            s1 = _drop(state, action)
        else:
            s1 = _move(state, action)
        if not _is_check(s1):
            is_mate = False
    return is_mate
