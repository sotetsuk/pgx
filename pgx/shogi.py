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


def init() -> ShogiState:
    board = _make_init_board()
    state = ShogiState(board=board, hand=np.zeros(14, dtype=np.int32))
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
    cn, cnp, cf, cfp = _is_check(s)
    if cn + cf != 0:
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


# turnから報酬計算
def _turn_to_reward(turn: int) -> int:
    if turn == 0:
        return 1
    else:
        return -1


# 初期盤面生成
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


# 盤面の情報をStateに変換
def _make_board(bs: np.ndarray) -> ShogiState:
    board = np.zeros((29, 81), dtype=np.int32)
    for i in range(81):
        board[0][i] = 0
        board[bs[i]][i] = 1
    return ShogiState(board=board)


def _pawn_move(turn: int) -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    array[4][4 - _turn_to_reward(turn)] = 1
    return array


def _knight_move(turn: int) -> np.ndarray:
    array = np.zeros((9, 9), dtype=np.int32)
    array[3][4 - 2 * _turn_to_reward(turn)] = 1
    array[5][4 - 2 * _turn_to_reward(turn)] = 1
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
    array[3:6, 3:6] = 1
    array[4][4] = 0
    return array


# 端判定
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


# 端にいる駒の動けない地点へのフラグを折る
def _cut_outside(array: np.ndarray, point: int) -> np.ndarray:
    new_array = array
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


# 駒種と位置から到達できる地点をすべて生成
def _action_board(array: np.ndarray, point: int) -> np.ndarray:
    new_array = array
    y, t = _point_to_location(point)
    new_array = _cut_outside(new_array, point)
    return np.roll(new_array, (y - 4, t - 4), axis=(0, 1)).reshape(81)


# 大駒・香車以外の位置ごとの移動できる地点を記録
POINT_MOVES = np.zeros((81, 29, 81), dtype=np.int32)
for i in range(81):
    POINT_MOVES[i][1] = _action_board(_pawn_move(0), i)
    POINT_MOVES[i][3] = _action_board(_knight_move(0), i)
    POINT_MOVES[i][4] = _action_board(_silver_move(0), i)
    POINT_MOVES[i][7] = _action_board(_gold_move(0), i)
    POINT_MOVES[i][8] = _action_board(_king_move(), i)
    POINT_MOVES[i][9] = POINT_MOVES[i][7]
    POINT_MOVES[i][10] = POINT_MOVES[i][7]
    POINT_MOVES[i][11] = POINT_MOVES[i][7]
    POINT_MOVES[i][12] = POINT_MOVES[i][7]
    POINT_MOVES[i][15] = _action_board(_pawn_move(1), i)
    POINT_MOVES[i][17] = _action_board(_knight_move(1), i)
    POINT_MOVES[i][18] = _action_board(_silver_move(1), i)
    POINT_MOVES[i][21] = _action_board(_gold_move(1), i)
    POINT_MOVES[i][22] = _action_board(_king_move(), i)
    POINT_MOVES[i][23] = POINT_MOVES[i][21]
    POINT_MOVES[i][24] = POINT_MOVES[i][21]
    POINT_MOVES[i][25] = POINT_MOVES[i][21]
    POINT_MOVES[i][26] = POINT_MOVES[i][21]


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
def _dlshogi_action(direction: int, to: int) -> int:
    return direction * 81 + to


# from, toが同じ縦列にあるかどうか
def _is_same_column(_from: int, to: int) -> bool:
    return _from // 9 == to // 9


# from, toが同じ横列にあるかどうか
def _is_same_row(_from: int, to: int) -> bool:
    return _from % 9 == to % 9


# from, toが右肩上がりの斜め方向で同じ筋にあるか
def _is_same_rising(_from: int, to: int) -> bool:
    return _from // 9 - _from % 9 == to // 9 - to % 9


# from, toが右肩下がりの斜め方向で同じ筋にあるか
def _is_same_declining(_from: int, to: int) -> bool:
    return _from // 9 + _from % 9 == to // 9 + to % 9


# from, からdirectionの方向に行ったときにtoに到達できるか
def _is_same_line(from_: int, to: int, direction: int) -> bool:
    dir = direction % 10
    if dir == 0 or dir == 5:
        return _is_same_column(from_, to)
    elif dir == 3 or dir == 4:
        return _is_same_row(from_, to)
    elif dir == 2 or dir == 6:
        return _is_same_rising(from_, to)
    elif dir == 1 or dir == 7:
        return _is_same_declining(from_, to)
    else:
        return False


# fromからある方向に移動させたときの位置と距離の関係
# それ以上その方向に動かせないなら全部0
def _dis_direction_array(from_: int, turn: int, direction: int) -> np.ndarray:
    array = np.zeros(81, dtype=np.int32)
    dif = _direction_to_dif(direction, turn)
    for i in range(8):
        dis = 1 + i
        to = from_ + dif * dis
        if _is_in_board(to) and _is_same_line(from_, to, direction):
            array[to] = dis
    return array


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


# ShogiActionをdlshogiのint型actionに変換
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
    for i in range(8):
        f -= dif
        if _is_in_board(f) and _from == -1 and _piece_type(state, f) != 0:
            _from = f
    return _from, direction >= 10


def _direction_to_hand(direction: int) -> int:
    if direction <= 26:
        # direction:20が先手の歩（pieceの1）に対応
        return direction - 19
    else:
        # direction:27が後手の歩（pieceの15）に対応
        return direction - 12


def _piece_type(state: ShogiState, point: int) -> int:
    return int(state.board[:, point].argmax())


# dlshogiのactionの情報をShogiActionの情報に変換
def _dlshogi_move_action(
    direction: int, to: int, state: ShogiState
) -> ShogiAction:
    _from, is_promote = _direction_to_from(direction, to, state)
    piece = _piece_type(state, _from)
    captured = _piece_type(state, to)
    return ShogiAction(False, piece, to, _from, captured, is_promote)


def _dlshogi_drop_action(
    direction: int, to: int, state: ShogiState
) -> ShogiAction:
    piece = _direction_to_hand(direction)
    return ShogiAction(True, piece, to)


def _dlaction_to_action(action: int, state: ShogiState) -> ShogiAction:
    direction, to = _separate_dlaction(action)
    if direction <= 19:
        # 移動のaction
        return _dlshogi_move_action(direction, to, state)
    else:
        # 駒打ち
        return _dlshogi_drop_action(direction, to, state)


# 手番側でない色を返す
def _another_color(state: ShogiState) -> int:
    return (state.turn + 1) % 2


# pointが盤面内かどうか
def _is_in_board(point: int) -> bool:
    return 0 <= point <= 80


# 相手の駒を同じ種類の自分の駒に変換する
def _convert_piece(piece: int) -> int:
    if piece == 0:
        return 0
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
    else:
        return (piece - 1) // 14


# 盤面のどこに何の駒があるかをnp.arrayに移したもの
# 同じ座標に複数回piece_typeを使用する場合はこちらを使った方が良い
def _board_status(state: ShogiState) -> np.ndarray:
    return state.board.argmax(axis=0)


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
    s = state
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
    s = state
    s.hand[_piece_to_hand(action.piece)] -= 1
    s.board[action.piece][action.to] = 1
    s.board[0][action.to] = 0
    return s


# 方向ごとの大ごまの動きのなかで一番奥の地点を返す
def _inner_point(bs_one: np.ndarray, from_: int, direction: int) -> int:
    dir_array = _dis_direction_array(from_, 0, direction)
    # 途中で駒にぶつからない場合
    if np.all(dir_array * bs_one == 0):
        return from_ + _direction_to_dif(direction, 0) * np.max(dir_array)
    else:
        return _nearest_position(from_, direction, bs_one)


# fromからtoまでの地点をdifごとに1に置き換える
# 最大8回
# fromは0、toは1に
def _change_between(from_: int, to: int, dif: int) -> np.ndarray:
    array = np.zeros(81, dtype=np.int32)
    point = from_
    flag = from_ != to
    for i in range(8):
        point += dif
        if flag and _is_in_board(point):
            array[point] = 1
        if point == to:
            flag = False
    return array


# 香車の動き
def _lance_move(bs: np.ndarray, from_: int, turn: int) -> np.ndarray:
    bs_one = np.where(bs == 0, 0, 1)
    direction = 5 * turn
    point = _inner_point(bs_one, from_, direction)
    return _change_between(from_, point, _direction_to_dif(0, turn))


# 角の動き
def _bishop_move(bs: np.ndarray, from_: int) -> np.ndarray:
    to = np.zeros(81, dtype=np.int32)
    bs_one = np.where(bs == 0, 0, 1)
    ur_point = _inner_point(bs_one, from_, 2)
    to += _change_between(from_, ur_point, -10)
    ul_point = _inner_point(bs_one, from_, 1)
    to += _change_between(from_, ul_point, 8)
    dr_point = _inner_point(bs_one, from_, 7)
    to += _change_between(from_, dr_point, -8)
    dl_point = _inner_point(bs_one, from_, 6)
    to += _change_between(from_, dl_point, 10)
    return to


# 飛車の動き
def _rook_move(bs: np.ndarray, from_: int) -> np.ndarray:
    to = np.zeros(81, dtype=np.int32)
    bs_one = np.where(bs == 0, 0, 1)
    u_point = _inner_point(bs_one, from_, 0)
    to += _change_between(from_, u_point, -1)
    d_point = _inner_point(bs_one, from_, 5)
    to += _change_between(from_, d_point, 1)
    r_point = _inner_point(bs_one, from_, 4)
    to += _change_between(from_, r_point, -9)
    l_point = _inner_point(bs_one, from_, 3)
    to += _change_between(from_, l_point, 9)
    return to


# 馬の動き
def _horse_move(bs: np.ndarray, from_: int) -> np.ndarray:
    # 角の動き＋玉の動き
    to = _bishop_move(bs, from_) + POINT_MOVES[from_][8]
    return np.where(to == 0, 0, 1)


# 龍の動き
def _dragon_move(bs: np.ndarray, from_: int) -> np.ndarray:
    # 飛車の動き＋玉の動き
    to = _rook_move(bs, from_) + POINT_MOVES[from_][8]
    return np.where(to == 0, 0, 1)


# 駒種と位置から到達できる場所を返す
def _piece_moves(bs: np.ndarray, piece: int, point: int) -> np.ndarray:
    moves = POINT_MOVES[point][piece]
    # 香車の動き
    if piece == 2:
        moves = _lance_move(bs, point, 0)
    if piece == 16:
        moves = _lance_move(bs, point, 1)
    # 角の動き
    if piece == 5 or piece == 19:
        moves = _bishop_move(bs, point)
    # 飛車の動き
    if piece == 6 or piece == 20:
        moves = _rook_move(bs, point)
    # 馬の動き
    if piece == 13 or piece == 27:
        moves = _horse_move(bs, point)
    # 龍の動き
    if piece == 14 or piece == 28:
        moves = _dragon_move(bs, point)
    return moves


# 小駒のactionのみを返すpiece_moves
def _small_piece_moves(piece: int, point: int) -> np.ndarray:
    return POINT_MOVES[point][piece]


# 敵陣かどうか
def _is_enemy_zone(turn: int, point: int) -> bool:
    if turn == 0:
        return point % 9 <= 2
    else:
        return point % 9 >= 6


# 成れるかどうか
def _can_promote(piece: int, _from: int, to: int) -> bool:
    # pieceが飛車以下でないと成れない
    if piece % 14 > 6 or piece % 14 == 0:
        return False
    # _fromとtoのどちらかが敵陣であれば成れる
    else:
        return _is_enemy_zone(_owner(piece), _from) or _is_enemy_zone(
            _owner(piece), to
        )


def _create_one_piece_actions(
    piece: int, _from: int, to: int, actions: np.ndarray
) -> np.ndarray:
    new_actions = actions
    normal_dir = _point_to_direction(_from, to, False, _owner(piece))
    normal_act = _dlshogi_action(normal_dir, to)
    new_actions[normal_act] = 1
    pro_dir = _point_to_direction(_from, to, True, _owner(piece))
    pro_act = _dlshogi_action(pro_dir, to)
    if _can_promote(piece, _from, to):
        new_actions[pro_act] = 1
    return new_actions


def _create_actions(piece: int, _from: int, to: np.ndarray) -> np.ndarray:
    actions = np.zeros(2754, dtype=np.int32)
    for i in range(81):
        if to[i] != 0:
            actions = _create_one_piece_actions(piece, _from, i, actions)
    return actions


def _create_piece_actions(piece: int, _from: int) -> np.ndarray:
    return _create_actions(piece, _from, POINT_MOVES[_from][piece])


# actionを追加する
def _add_action(add_array: np.ndarray, origin_array: np.ndarray) -> np.ndarray:
    return np.where(add_array == 1, 1, origin_array)


# actionを削除する
def _filter_action(
    filter_array: np.ndarray, origin_array: np.ndarray
) -> np.ndarray:
    return np.where(filter_array == 1, 0, origin_array)


# 駒の種類と位置から生成できるactionのフラグを立てる
def _add_move_actions(piece: int, _from: int, array: np.ndarray) -> np.ndarray:
    actions = _create_piece_actions(piece, _from)
    return np.where(actions == 1, 1, array)


# 駒の種類と位置から生成できるactionのフラグを折る
def _filter_move_actions(
    piece: int, _from: int, array: np.ndarray
) -> np.ndarray:
    actions = _create_piece_actions(piece, _from)
    return np.where(actions == 1, 0, array)


# 駒打ちのactionを追加する
def _add_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = array
    direction = _hand_to_direction(piece)
    np.put(new_array, np.arange(81 * direction, 81 * (direction + 1)), 1)
    return new_array


# 駒打ちのactionのフラグを折る
def _filter_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = array
    direction = _hand_to_direction(piece)
    np.put(new_array, np.arange(81 * direction, 81 * (direction + 1)), 0)
    return new_array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は使わないがlegal_actionsが設定されていない場合に使用
def _init_legal_actions(state: ShogiState) -> ShogiState:
    s = state
    bs = _board_status(s)
    # 移動の追加
    for i in range(81):
        piece = bs[i]
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


# 成駒を成る前の駒に変更
def _degeneration_piece(piece: int) -> int:
    if piece % 14 >= 9 or piece == 14 or piece == 28:
        return piece - 8
    else:
        return piece


# 駒の移動によるlegal_actionsの更新
def _update_legal_move_actions(
    state: ShogiState, action: ShogiAction
) -> ShogiState:
    s = state
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
    # 取った駒を自分の持ち駒に変換
    # 取っていない場合は0
    captured = _degeneration_piece(_convert_piece(action.captured))
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if action.captured != 0:
        new_enemy_actions = _filter_move_actions(
            action.captured, action.to, new_enemy_actions
        )
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
    s = state
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
    new_array = array
    for i in range(81):
        if owner[i] != turn:
            continue
        np.put(new_array, np.arange(i, 1620 + i, 81), 0)
    return new_array


# 駒がある地点への駒打ちを除く
def _filter_occupied_drop_actions(
    turn: int, owner: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = array
    for i in range(81):
        if owner[i] == 2:
            continue
        if turn == 0:
            np.put(
                new_array,
                np.arange(81 * 20 + i, 81 * 27 + i, 81),
                0,
            )
        else:
            np.put(
                new_array,
                np.arange(81 * 27 + i, 81 * 34 + i, 81),
                0,
            )
    return new_array


# boardのlegal_actionsを利用して合法手を生成する
# 大駒や香車の利きはboardのlegal_actionsに追加していないので、ここで追加する
# 自殺手や反則手はここでは除かない
def _legal_actions(state: ShogiState) -> np.ndarray:
    if state.turn == 0:
        action_array = state.legal_actions_black
    else:
        action_array = state.legal_actions_white
    bs = _board_status(state)
    own = _pieces_owner(state)
    for i in range(81):
        piece = bs[i]
        if _owner(piece) == state.turn:
            action_array = _add_action(
                _create_actions(piece, i, _piece_moves(bs, piece, i)),
                action_array,
            )
    # 自分の駒がある位置への移動actionを除く
    action_array = _filter_my_piece_move_actions(state.turn, own, action_array)
    # 駒がある地点への駒打ちactionを除く
    action_array = _filter_occupied_drop_actions(state.turn, own, action_array)
    return action_array


# 王手判定
# 密接・遠隔の王手で分ける
def _is_check(state: ShogiState) -> Tuple[int, np.ndarray, int, np.ndarray]:
    # そもそも王がいない場合
    if np.all(state.board[8 + 14 * state.turn] == 0):
        return 0, np.zeros(81, dtype=np.int32), 0, np.zeros(81, dtype=np.int32)
    else:
        return _is_check_(state)


# 玉がいる前提
def _is_check_(state: ShogiState) -> Tuple[int, np.ndarray, int, np.ndarray]:
    check = 0
    checking_point = np.zeros((2, 81), dtype=np.int32)
    king_point = int(state.board[8 + 14 * state.turn, :].argmax())
    near_king = _small_piece_moves(8 + 14 * state.turn, king_point)
    bs = _board_status(state)
    for i in range(81):
        piece = bs[i]
        if (
            _owner(piece) == _another_color(state)
            and _piece_moves(bs, piece, i)[king_point] == 1
        ):
            # 桂馬の王手も密接としてカウント
            if near_king[i] == 1 or piece % 14 == 3:
                check += 10
                checking_point[0][i] = 1
            else:
                # 遠隔の王手は9以上ありえない
                check += 1
                checking_point[1][i] = 1
    return check // 10, checking_point[0], check % 10, checking_point[1]


# 二歩判定
# 手番側でチェックする
def _is_double_pawn(state: ShogiState) -> bool:
    is_double_pawn = False
    bs = _board_status(state)
    for i in range(9):
        num_pawn = np.count_nonzero(
            bs[9 * i : 9 * (i + 1)] == 1 + state.turn * 14
        )
        if num_pawn >= 2:
            is_double_pawn = True
    return is_double_pawn


# 行き所のない駒判定
def _is_stuck(state: ShogiState) -> bool:
    is_stuck = False
    bs = _board_status(state)
    line1 = bs[0::9]
    if np.any(line1 == 1):
        is_stuck = True
    if np.any(line1 == 2):
        is_stuck = True
    if np.any(line1 == 3):
        is_stuck = True
    line2 = bs[1::9]
    if np.any(line2 == 3):
        is_stuck = True
    line8 = bs[7::9]
    if np.any(line8 == 17):
        is_stuck = True
    line9 = bs[8::9]
    if np.any(line9 == 15):
        is_stuck = True
    if np.any(line9 == 16):
        is_stuck = True
    if np.any(line9 == 17):
        is_stuck = True
    return is_stuck


# pinされている方向から縦、横、右斜めうえ、右斜め下の四方向に分類
def _direction_to_pin(direction: int) -> int:
    if direction == 0 or direction == 5:
        return 1
    if direction == 1 or direction == 7:
        return 2
    if direction == 2 or direction == 6:
        return 3
    if direction == 3 or direction == 4:
        return 4
    return 0


# それぞれの方向について、1番fromに近い駒の位置を返す
def _nearest_position(from_: int, direction: int, bs_one: np.ndarray) -> int:
    flag1 = True
    point1 = -1
    dif = _direction_to_dif(direction, 0)
    for i in range(8):
        point = from_ + (1 + i) * dif
        if (
            flag1
            and _is_in_board(point)
            and _is_same_line(from_, point, direction)
            and bs_one[point] == 1
        ):
            point1 = point
            flag1 = False
    return point1


# pinされている駒の位置と方向を記録(1方向のみ)
def _direction_pin(
    bs: np.ndarray,
    turn: int,
    king_point: int,
    direction: int,
    array: np.ndarray,
) -> np.ndarray:
    new_array = array
    e_turn = (turn + 1) % 2
    bs_one = np.where(bs == 0, 0, 1)
    # 玉に一番近い駒の位置
    point1 = _nearest_position(king_point, direction, bs_one)
    # 2番目に近い駒の位置
    point2 = _nearest_position(point1, direction, bs_one)
    if point1 == -1 or point2 == -1:
        return new_array
    piece1 = bs[point1]
    piece2 = bs[point2]
    if _owner(piece1) != turn:
        return new_array
    flag = False
    if piece2 == 2 + 14 * e_turn and direction == 5 * turn:
        flag = True
    if (piece2 == 6 + e_turn * 14 or piece2 == 14 + e_turn * 14) and (
        direction == 0 or direction == 3 or direction == 4 or direction == 5
    ):
        flag = True
    if (piece2 == 5 + e_turn * 14 or piece2 == 13 + e_turn * 14) and (
        direction == 1 or direction == 2 or direction == 6 or direction == 7
    ):
        flag = True
    if flag:
        new_array[point1] = _direction_to_pin(direction)
    return new_array


# pinされている駒の位置と方向を記録
def _pin(state: ShogiState) -> np.ndarray:
    bs = _board_status(state)
    turn = state.turn
    pins = np.zeros(81, dtype=np.int32)
    king_point = int(state.board[8 + 14 * turn, :].argmax())
    for i in range(8):
        pins = _direction_pin(bs, turn, king_point, i, pins)
    return pins


# 特定の方向の動きだけを除いたactionを生成する
def _eliminate_direction(actions: np.ndarray, direction: int) -> np.ndarray:
    new_array = actions
    # 2方向と成・不成の4方向のフラグを折る
    dir1 = 0
    dir2 = 0
    # 縦
    if direction == 1:
        dir1 = 0
        dir2 = 5
    # 右下がり
    if direction == 2:
        dir1 = 1
        dir2 = 7
    # 右上がり
    if direction == 3:
        dir1 = 2
        dir2 = 6
    # 横
    if direction == 4:
        dir1 = 3
        dir2 = 4
    pro_dir1 = dir1 + 8
    pro_dir2 = dir2 + 8
    new_array[dir1 * 81 : (dir1 + 1) * 81] = 0
    new_array[dir2 * 81 : (dir2 + 1) * 81] = 0
    new_array[pro_dir1 * 81 : (pro_dir1 + 1) * 81] = 0
    new_array[pro_dir2 * 81 : (pro_dir2 + 1) * 81] = 0
    return new_array


# 利きの判定
# 玉の位置を透過する（玉をいないものとして扱う）ことで香車や角などの利きを玉の奥まで通す
def _kingless_effected_positions(
    bs: np.ndarray, king_point: int, turn: int
) -> np.ndarray:
    all_effect = np.zeros(81, dtype=np.int32)
    bs[king_point] = 0
    for i in range(81):
        if _owner(bs[i]) == turn:
            all_effect += _piece_moves(bs, bs[i], i)
    return all_effect


# 玉が移動する手に合法なものがあるかを調べる
def _king_escape(state: ShogiState) -> bool:
    king_point = int(state.board[8 + 14 * state.turn, :].argmax())
    bs = _board_status(state)
    effects = _kingless_effected_positions(
        bs, king_point, _another_color(state)
    )
    king_moves = POINT_MOVES[king_point][8]
    flag = False
    for i in range(81):
        if (
            king_moves[i] == 1
            and _owner(bs[i]) != state.turn
            and effects[i] == 0
        ):
            flag = True
    return flag


# point同士の間の位置にフラグを立てる
def _between(point1: int, point2: int) -> np.ndarray:
    direction = _point_to_direction(point1, point2, False, 0)
    if direction == -1:
        bet = np.zeros(81, dtype=np.int32)
    else:
        bet = _change_between(point1, point2, _direction_to_dif(direction, 0))
    bet[point2] = 0
    return bet


# pinされている駒の非合法手を弾いてlegal_actionを返す
def _eliminate_pin_actions(
    bs: np.ndarray, pins: np.ndarray, l_actions: np.ndarray
):
    for i in range(81):
        if pins[i] != 0:
            l_actions = _filter_action(
                _eliminate_direction(_create_piece_actions(bs[i], i), pins[i]),
                l_actions,
            )
    return l_actions


# 玉が逃げる以外の手に王手回避の手があるかをチェック
# 存在しない場合True
# 両王手は考えない(事前にはじく)
def _is_avoid_check(
    cn: int,
    cnp: np.ndarray,
    cfp: np.ndarray,
    king_point: int,
    legal_actions: np.ndarray,
):
    # 密接の王手
    if cn == 1:
        # 玉が逃げる手以外の合法手は王手をかけた駒がある座標への移動のみ
        point = int(cnp.argmax())
        points = cnp
    # 開き王手
    else:
        point = int(cfp.argmax())
        # pointとking_pointの間。ここに駒を打ったり移動させたりする手は合法
        points = _between(king_point, point)
    for i in range(81):
        if points[i] == 0 and point != i:
            np.put(legal_actions, np.arange(i, 81 * 34 + i, 81), 0)
    return (legal_actions == 0).all()


# 詰み判定関数
# 王手がかかっていないならFalse
def _is_mate(state: ShogiState) -> bool:
    cn, cnp, cf, cfp = _is_check(state)
    # 王手がかかっていないならFalse
    if cn + cf == 0:
        return False
    else:
        return _is_mate_noncheck(cn, cnp, cf, cfp, state)


# 王手の有無にかかわらず詰みを判定する
def _is_mate_noncheck(
    cn: int, cnp: np.ndarray, cf: int, cfp: np.ndarray, state: ShogiState
):
    legal_actions = _legal_actions(state)
    bs = _board_status(state)
    king_point = int(state.board[8 + 14 * state.turn, :].argmax())
    # 玉が逃げる手以外の合法手
    legal_actions = _filter_move_actions(
        8 + 14 * state.turn, king_point, legal_actions
    )
    # ピンされている駒の非合法な動きをのぞく
    legal_actions = _eliminate_pin_actions(bs, _pin(state), legal_actions)
    # 玉が逃げる手が合法なら詰んでない
    if _king_escape(state):
        return False
    else:
        # 両王手がかかっている場合はTrue, そうでなければis_avoid_check参照
        return cn + cf >= 2 or _is_avoid_check(
            cn, cnp, cfp, king_point, legal_actions
        )
