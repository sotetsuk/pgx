import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# 指し手のdataclass
@dataclass
class AnimalShogiAction:
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
class AnimalShogiState:
    # turn 先手番なら0 後手番なら1
    turn: int = 0
    # board 盤面の駒。
    # 空白,先手ヒヨコ,先手キリン,先手ゾウ,先手ライオン,先手ニワトリ,後手ヒヨコ,後手キリン,後手ゾウ,後手ライオン,後手ニワトリ
    # の順で駒がどの位置にあるかをone_hotで記録
    # ヒヨコ: Pawn, キリン: Rook, ゾウ: Bishop, ライオン: King, ニワトリ: Gold　と対応
    board: np.ndarray = np.zeros((11, 12), dtype=np.int32)
    # hand 持ち駒。先手ヒヨコ,先手キリン,先手ゾウ,後手ヒヨコ,後手キリン,後手ゾウの6種の値を増減させる
    hand: np.ndarray = np.zeros(6, dtype=np.int32)
    # legal_actions_black/white: 自殺手や王手放置などの手も含めた合法手の一覧
    # move/dropによって変化させる
    legal_actions_black: np.ndarray = np.zeros(180, dtype=np.int32)
    legal_actions_white: np.ndarray = np.zeros(180, dtype=np.int32)
    # checked: ターンプレイヤーの王に王手がかかっているかどうか
    is_check: int = 0
    # checking_piece: ターンプレイヤーに王手をかけている駒の座標
    checking_piece: np.ndarray = np.zeros(12, dtype=np.int32)


# BLACK/WHITE/(NONE)_○○_MOVEは22にいるときの各駒の動き
# 端にいる場合は対応するところに0をかけていけないようにする
BLACK_PAWN_MOVE = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
WHITE_PAWN_MOVE = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
BLACK_GOLD_MOVE = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
WHITE_GOLD_MOVE = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
ROOK_MOVE = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
BISHOP_MOVE = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]])
KING_MOVE = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])


INIT_BOARD = AnimalShogiState(
    turn=0,
    board=np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    hand=np.array([0, 0, 0, 0, 0, 0]),
)


def init() -> AnimalShogiState:
    return _init_legal_actions(copy.deepcopy(INIT_BOARD))


def step(
    state: AnimalShogiState, action: int
) -> Tuple[AnimalShogiState, int, bool]:
    # state, 勝敗判定,終了判定を返す
    # 勝敗判定は勝者側のturnを返す（決着がついていない・引き分けの場合は2を返す）
    s = copy.deepcopy(state)
    legal_actions = _legal_actions(s)
    # 合法手が存在しない場合、手番側の負けで終了
    if np.all(legal_actions == 0):
        print("no legal actions.")
        return s, _turn_to_reward(_another_color(s)), True
    # actionが合法手でない場合、手番側の負けで終了
    _action = _dlaction_to_action(action, s)
    # actionのfromが盤外に存在すると挙動がおかしくなるのでここではじいておく
    if _action.from_ > 11 or _action.from_ < 0:
        print("an illegal action")
        return s, _turn_to_reward(_another_color(s)), True
    if legal_actions[_action_to_dlaction(_action, s.turn)] == 0:
        print("an illegal action")
        return s, _turn_to_reward(_another_color(s)), True
    # actionが合法手の場合
    # 駒打ちの場合の操作
    if _action.is_drop:
        s = _update_legal_drop_actions(s, _action)
        s = _drop(s, _action)
        print("drop: piece =", _action.piece, ", to =", _action.to)
    # 駒の移動の場合の操作
    else:
        s = _update_legal_move_actions(s, _action)
        s = _move(s, _action)
        print("move: piece =", _action.piece, ", to =", _action.to)
        # トライ成功時には手番側の勝ちで終了
        if _is_try(_action):
            print("try")
            return s, _turn_to_reward(s.turn), True
    s.turn = _another_color(s)
    s.is_check = _is_check(s)
    # 王手をかけている駒は直前に動かした駒
    if s.is_check:
        # 王手返しの王手の場合があるので一度リセットする
        s.checking_piece = np.zeros(12, dtype=np.int32)
        s.checking_piece[_action.to] = 1
        print("checked!")
    else:
        s.checking_piece = np.zeros(12, dtype=np.int32)
    return s, 0, False


def _turn_to_reward(turn: int) -> int:
    if turn == 0:
        return 1
    else:
        return -1


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
def _dlshogi_action(direction: int, to: int) -> int:
    return direction * 12 + to


# fromの座標とtoの座標からdirを生成
def _point_to_direction(_from: int, to: int, promote: bool, turn: int) -> int:
    direction = -1
    dis = to - _from
    # 後手番の動きは反転させる
    if turn == 1:
        dis = -dis
    # UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP_PROMOTE... の順でdirを割り振る
    # PROMOTEの場合は+8する処理を入れるが、どうぶつ将棋ではUP_PROMOTEしか存在しない(はず)
    if dis == -1:
        direction = 0
    if dis == 3:
        direction = 1
    if dis == -5:
        direction = 2
    if dis == 4:
        direction = 3
    if dis == -4:
        direction = 4
    if dis == 1:
        direction = 5
    if dis == 5:
        direction = 6
    if dis == -3:
        direction = 7
    if promote:
        direction += 8
    return direction


# 打った駒の種類をdirに変換
def _hand_to_direction(piece: int) -> int:
    # 移動のdirはPROMOTE_UPの8が最大なので9以降に配置
    # 9: 先手ヒヨコ 10: 先手キリン... 14: 後手ゾウ　に対応させる
    if piece <= 5:
        return 8 + piece
    else:
        return 6 + piece


# AnimalShogiActionをdlshogiのint型actionに変換
def _action_to_dlaction(action: AnimalShogiAction, turn: int) -> int:
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
    return action // 12, action % 12


# directionからfromがtoからどれだけ離れてるかと成りを含む移動かを得る
# 手番の情報が必要
def _direction_to_from(direction: int, to: int, turn: int) -> Tuple[int, bool]:
    dif = 0
    if direction == 0 or direction == 8:
        dif = -1
    if direction == 1:
        dif = 3
    if direction == 2:
        dif = -5
    if direction == 3:
        dif = 4
    if direction == 4:
        dif = -4
    if direction == 5:
        dif = 1
    if direction == 6:
        dif = 5
    if direction == 7:
        dif = -3
    if turn == 0:
        if direction >= 8:
            return to - dif, True
        else:
            return to - dif, False
    else:
        if direction >= 8:
            return to + dif, True
        else:
            return to + dif, False


def _direction_to_hand(direction: int) -> int:
    if direction <= 11:
        return direction - 8
    else:
        return direction - 6


def _dlaction_to_action(
    action: int, state: AnimalShogiState
) -> AnimalShogiAction:
    direction, to = _separate_dlaction(action)
    if direction <= 8:
        # 駒の移動
        _from, is_promote = _direction_to_from(direction, to, state.turn)
        piece = _piece_type(state, _from)
        captured = _piece_type(state, to)
        return AnimalShogiAction(False, piece, to, _from, captured, is_promote)
    else:
        # 駒打ち
        piece = _direction_to_hand(direction)
        return AnimalShogiAction(True, piece, to)


# 手番側でない色を返す
def _another_color(state: AnimalShogiState) -> int:
    return (state.turn + 1) % 2


# 相手の駒を同じ種類の自分の駒に変換する
def _convert_piece(piece: int) -> int:
    p = (piece + 5) % 10
    if p == 0:
        return 10
    else:
        return p


# 駒から持ち駒への変換
# 先手ひよこが0、後手ぞうが5
def _piece_to_hand(piece: int) -> int:
    if piece % 5 == 0:
        p = piece - 4
    else:
        p = piece
    if p < 6:
        return p - 1
    else:
        return p - 3


#  移動の処理
def _move(
    state: AnimalShogiState,
    action: AnimalShogiAction,
) -> AnimalShogiState:
    s = copy.deepcopy(state)
    s.board[action.piece][action.from_] = 0
    s.board[0][action.from_] = 1
    s.board[action.captured][action.to] = 0
    if action.is_promote:
        s.board[action.piece + 4][action.to] = 1
    else:
        s.board[action.piece][action.to] = 1
    if action.captured != 0:
        s.hand[_piece_to_hand(_convert_piece(action.captured))] += 1
    return s


#  駒打ちの処理
def _drop(
    state: AnimalShogiState, action: AnimalShogiAction
) -> AnimalShogiState:
    s = copy.deepcopy(state)
    s.hand[_piece_to_hand(action.piece)] -= 1
    s.board[action.piece][action.to] = 1
    s.board[0][action.to] = 0
    return s


#  ある座標に存在する駒種を返す
def _piece_type(state: AnimalShogiState, point: int) -> int:
    if point < 0 or point > 11:
        return 0
    return int(state.board[:, point].argmax())


# ある駒の持ち主を返す
def _owner(piece: int) -> int:
    if piece == 0:
        return 2
    return (piece - 1) // 5


# 盤面のどこに何の駒があるかをnp.arrayに移したもの
# 同じ座標に複数回piece_typeを使用する場合はこちらを使った方が良い
def _board_status(state: AnimalShogiState) -> np.ndarray:
    board = np.zeros(12, dtype=np.int32)
    for i in range(12):
        board[i] = _piece_type(state, i)
    return board


# 駒の持ち主の判定
def _pieces_owner(state: AnimalShogiState) -> np.ndarray:
    board = np.zeros(12, dtype=np.int32)
    for i in range(12):
        piece = _piece_type(state, i)
        if piece == 0:
            board[i] = 2
        else:
            board[i] = (piece - 1) // 5
    return board


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
def _is_side(point: int) -> Tuple[bool, bool, bool, bool]:
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


# point(0~11)を座標((0, 0)~(2, 3))に変換
def _point_to_location(point: int) -> Tuple[int, int]:
    return point // 4, point % 4


# はみ出す部分をカットする
def _cut_outside(array: np.ndarray, point: int) -> np.ndarray:
    new_array = copy.deepcopy(array)
    u, d, l, r = _is_side(point)
    if u:
        new_array[:, 0] *= 0
    if d:
        new_array[:, 2] *= 0
    if r:
        new_array[0, :] *= 0
    if l:
        new_array[2, :] *= 0
    return new_array


def _action_board(array: np.ndarray, point: int) -> np.ndarray:
    new_array = copy.deepcopy(array)
    y, t = _point_to_location(point)
    new_array = _cut_outside(new_array, point)
    return np.roll(new_array, (y - 1, t - 1), axis=(0, 1))


# 各駒の動き
def _black_pawn_move(point: int) -> np.ndarray:
    return _action_board(np.copy(BLACK_PAWN_MOVE), point)


def _white_pawn_move(point: int) -> np.ndarray:
    return _action_board(np.copy(WHITE_PAWN_MOVE), point)


def _black_gold_move(point: int) -> np.ndarray:
    return _action_board(np.copy(BLACK_GOLD_MOVE), point)


def _white_gold_move(point: int) -> np.ndarray:
    return _action_board(np.copy(WHITE_GOLD_MOVE), point)


def _rook_move(point: int) -> np.ndarray:
    return _action_board(np.copy(ROOK_MOVE), point)


def _bishop_move(point: int) -> np.ndarray:
    return _action_board(np.copy(BISHOP_MOVE), point)


def _king_move(point: int) -> np.ndarray:
    return _action_board(np.copy(KING_MOVE), point)


#  座標と駒の種類から到達できる座標を列挙する関数
def _point_moves(piece: int, point: int) -> np.ndarray:
    if piece == 1:
        return _black_pawn_move(point)
    if piece == 6:
        return _white_pawn_move(point)
    if piece % 5 == 2:
        return _rook_move(point)
    if piece % 5 == 3:
        return _bishop_move(point)
    if piece % 5 == 4:
        return _king_move(point)
    if piece == 5:
        return _black_gold_move(point)
    if piece == 10:
        return _white_gold_move(point)
    return np.zeros(12, dtype=np.int32)


# 利きの判定
def _effected_positions(state: AnimalShogiState, turn: int) -> np.ndarray:
    all_effect = np.zeros(12)
    board = _board_status(state)
    piece_owner = _pieces_owner(state)
    for i in range(12):
        own = piece_owner[i]
        if own != turn:
            continue
        piece = board[i]
        effect = _point_moves(piece, i).reshape(12)
        all_effect += effect
    return all_effect


# 王手の判定(turn側の王に王手がかかっているかを判定)
def _is_check(state: AnimalShogiState) -> bool:
    effects = _effected_positions(state, _another_color(state))
    king_location = state.board[4 + 5 * state.turn, :].argmax()
    return effects[king_location] != 0


# 成る動きが合法かどうかの判定
def _can_promote(to: int, piece: int) -> bool:
    if piece == 1 and to % 4 == 0:
        return True
    if piece == 6 and to % 4 == 3:
        return True
    return False


# 駒の種類と位置から生成できるactionのフラグを立てる
def _create_piece_actions(_from: int, piece: int) -> np.ndarray:
    turn = _owner(piece)
    actions = np.zeros(180, dtype=np.int32)
    motion = _point_moves(piece, _from).reshape(12)
    for i in range(12):
        if motion[i] == 0:
            continue
        if _can_promote(i, piece):
            pro_dir = _point_to_direction(_from, i, True, turn)
            pro_act = _dlshogi_action(pro_dir, i)
            actions[pro_act] = 1
        normal_dir = _point_to_direction(_from, i, False, turn)
        normal_act = _dlshogi_action(normal_dir, i)
        actions[normal_act] = 1
    return actions


# 駒の種類と位置から生成できるactionのフラグを立てる
def _add_move_actions(_from: int, piece: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(_from, piece)
    for i in range(180):
        if actions[i] == 1:
            new_array[i] = 1
    return new_array


# 駒の種類と位置から生成できるactionのフラグを折る
def _filter_move_actions(
    _from: int, piece: int, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    actions = _create_piece_actions(_from, piece)
    for i in range(180):
        if actions[i] == 1:
            new_array[i] = 0
    return new_array


# 駒打ちのactionを追加する
def _add_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(12):
        action = _dlshogi_action(direction, i)
        new_array[action] = 1
    return new_array


# 駒打ちのactionを消去する
def _filter_drop_actions(piece: int, array: np.ndarray) -> np.ndarray:
    new_array = copy.deepcopy(array)
    direction = _hand_to_direction(piece)
    for i in range(12):
        action = _dlshogi_action(direction, i)
        new_array[action] = 0
    return new_array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は使わないがlegal_actionsが設定されていない場合に使用
def _init_legal_actions(state: AnimalShogiState) -> AnimalShogiState:
    s = copy.deepcopy(state)
    bs = _board_status(s)
    # 移動の追加
    for i in range(12):
        piece = bs[i]
        if piece == 0:
            continue
        if piece <= 5:
            s.legal_actions_black = _add_move_actions(
                i, piece, s.legal_actions_black
            )
        else:
            s.legal_actions_white = _add_move_actions(
                i, piece, s.legal_actions_white
            )
    # 駒打ちの追加
    for i in range(3):
        if s.hand[i] != 0:
            s.legal_actions_black = _add_drop_actions(
                1 + i, s.legal_actions_black
            )
        if s.hand[i + 3] != 0:
            s.legal_actions_white = _add_drop_actions(
                6 + i, s.legal_actions_white
            )
    return s


# 駒の移動によるlegal_actionsの更新
def _update_legal_move_actions(
    state: AnimalShogiState, action: AnimalShogiAction
) -> AnimalShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
        enemy_actions = s.legal_actions_white
    else:
        player_actions = s.legal_actions_white
        enemy_actions = s.legal_actions_black
    # 元の位置にいたときのフラグを折る
    new_player_actions = _filter_move_actions(
        action.from_, action.piece, player_actions
    )
    new_enemy_actions = enemy_actions
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = _add_move_actions(
        action.to, action.piece, new_player_actions
    )
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if action.captured != 0:
        new_enemy_actions = _filter_move_actions(
            action.to, action.captured, new_enemy_actions
        )
        captured = _convert_piece(action.captured)
        # にわとりの場合ひよこに変換
        if captured % 5 == 0:
            captured -= 4
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
    state: AnimalShogiState, action: AnimalShogiAction
) -> AnimalShogiState:
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
    else:
        player_actions = s.legal_actions_white
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = _add_move_actions(
        action.to, action.piece, player_actions
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
    for i in range(12):
        if owner[i] != turn:
            continue
        for j in range(9):
            new_array[12 * j + i] = 0
    return new_array


# 駒がある地点への駒打ちを除く
def _filter_occupied_drop_actions(
    turn: int, owner: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    for i in range(12):
        if owner[i] == 2:
            continue
        for j in range(3):
            new_array[12 * (j + 9 + 3 * turn) + i] = 0
    return new_array


# 自殺手を除く
def _filter_suicide_actions(
    turn: int, king_sq: int, effects: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    moves = _king_move(king_sq).reshape(12)
    for i in range(12):
        if moves[i] == 0:
            continue
        if effects[i] == 0:
            continue
        direction = _point_to_direction(king_sq, i, False, turn)
        action = _dlshogi_action(direction, i)
        new_array[action] = 0
    return new_array


# 王手放置を除く
def _filter_leave_check_actions(
    turn: int, king_sq: int, check_piece: np.ndarray, array: np.ndarray
) -> np.ndarray:
    new_array = copy.deepcopy(array)
    moves = _king_move(king_sq).reshape(12)
    for i in range(12):
        # 王手をかけている駒の位置以外への移動は王手放置
        for j in range(15):
            # 駒打ちのフラグは全て折る
            if j > 8:
                new_array[12 * j + i] = 0
            # 王手をかけている駒の場所以外への移動ははじく
            if check_piece[i] == 0:
                new_array[12 * j + i] = 0
        # 玉の移動はそれ以外でも可能だがフラグが折れてしまっているので立て直す
        if moves[i] == 0:
            continue
        direction = _point_to_direction(king_sq, i, False, turn)
        action = _dlshogi_action(direction, i)
        new_array[action] = 1
    return new_array


# boardのlegal_actionsを利用して合法手を生成する
def _legal_actions(state: AnimalShogiState) -> np.ndarray:
    if state.turn == 0:
        action_array = copy.deepcopy(state.legal_actions_black)
    else:
        action_array = copy.deepcopy(state.legal_actions_white)
    king_sq = state.board[4 + 5 * state.turn].argmax()
    # 王手放置を除く
    if state.is_check:
        action_array = _filter_leave_check_actions(
            state.turn, king_sq, state.checking_piece, action_array
        )
    own = _pieces_owner(state)
    # 自分の駒がある位置への移動actionを除く
    action_array = _filter_my_piece_move_actions(state.turn, own, action_array)
    # 駒がある地点への駒打ちactionを除く
    action_array = _filter_occupied_drop_actions(state.turn, own, action_array)
    # 自殺手を除く
    effects = _effected_positions(state, _another_color(state))
    action_array = _filter_suicide_actions(
        state.turn, king_sq, effects, action_array
    )
    # その他の反則手を除く
    # どうぶつ将棋の場合はなし
    return action_array


# トライルールによる勝利判定
# 王が最奥に動くactionならTrue
def _is_try(action: AnimalShogiAction) -> bool:
    if action.piece == 4 and action.to % 4 == 0:
        return True
    if action.piece == 9 and action.to % 4 == 3:
        return True
    return False
