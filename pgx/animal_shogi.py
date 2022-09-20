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
    checked: int = 0
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
    return create_legal_actions(copy.deepcopy(INIT_BOARD))


def step(
    state: AnimalShogiState, action: int
) -> Tuple[AnimalShogiState, int, bool]:
    # state, 勝敗判定,終了判定を返す
    # 勝敗判定は勝者側のturnを返す（決着がついていない・引き分けの場合は2を返す）
    s = copy.deepcopy(state)
    _legal_actions = legal_actions2(s)
    # 合法手が存在しない場合、手番側の負けで終了
    if np.all(_legal_actions == 0):
        print("no legal actions.")
        return s, turn_to_reward(another_color(s)), True
    # actionが合法手でない場合、手番側の負けで終了
    _action = int_to_action(action, s)
    if _legal_actions[action_to_int(_action, s.turn)] == 0:
        print("an illegal action")
        return s, turn_to_reward(another_color(s)), True
    # actionが合法手の場合
    # 駒打ちの場合の操作
    if _action.is_drop:
        s = update_legal_actions_drop(s, _action)
        s = drop(s, _action)
    # 駒の移動の場合の操作
    else:
        s = update_legal_actions_move(s, _action)
        s = move(s, _action)
    s.turn = another_color(s)
    s.checked = is_check(s)
    # 王手をかけている駒は直前に動かした駒
    if s.checked:
        # 王手返しの王手の場合があるので一度リセットする
        s.checking_piece = np.zeros(12, dtype=np.int32)
        s.checking_piece[_action.to] = 1
    else:
        s.checking_piece = np.zeros(12, dtype=np.int32)
    return s, 0, False


def turn_to_reward(turn: int):
    if turn == 0:
        return 1
    else:
        return -1


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
def dlshogi_action(direction: int, to: int):
    return direction * 12 + to


# fromの座標とtoの座標からdirを生成
def point_to_direction(_from: int, to: int, promote: bool, turn: int):
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
def hand_piece_to_dir(piece: int):
    # 移動のdirはPROMOTE_UPの8が最大なので9以降に配置
    # 9: 先手ヒヨコ 10: 先手キリン... 14: 後手ゾウ　に対応させる
    if piece <= 5:
        return 8 + piece
    else:
        return 6 + piece


# AnimalShogiActionをdlshogiのint型actionに変換
def action_to_int(action: AnimalShogiAction, turn: int):
    if action.is_drop:
        return dlshogi_action(hand_piece_to_dir(action.piece), action.to)
    else:
        return dlshogi_action(
            point_to_direction(
                action.from_, action.to, action.is_promote, turn
            ),
            action.to,
        )


# dlshogiのint型actionをdirectionとtoに分解
def separate_int(action: int):
    # direction, to の順番
    return action // 12, action % 12


# directionからfromがtoからどれだけ離れてるかと成りを含む移動かを得る
# 手番の情報が必要
def direction_to_from(direction: int, to: int, turn: int):
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


def direction_to_hand_piece(direction: int):
    if direction <= 11:
        return direction - 8
    else:
        return direction - 6


def int_to_action(action: int, state: AnimalShogiState):
    direction, to = separate_int(action)
    if direction <= 8:
        # 駒の移動
        _from, is_promote = direction_to_from(direction, to, state.turn)
        piece = piece_type(state, _from)
        captured = piece_type(state, to)
        return AnimalShogiAction(False, piece, to, _from, captured, is_promote)
    else:
        # 駒打ち
        piece = direction_to_hand_piece(direction)
        return AnimalShogiAction(True, piece, to)


# 手番側でない色を返す
def another_color(state: AnimalShogiState):
    return (state.turn + 1) % 2


# 相手の駒を同じ種類の自分の駒に変換する
def convert_piece(piece: int):
    p = (piece + 5) % 10
    if p == 0:
        return 10
    else:
        return p


#  移動の処理
def move(
    state: AnimalShogiState,
    action: AnimalShogiAction,
):
    s = copy.deepcopy(state)
    s.board[action.piece][action.from_] = 0
    s.board[0][action.from_] = 1
    s.board[action.captured][action.to] = 0
    if action.is_promote:
        s.board[action.piece + 4][action.to] = 1
    else:
        s.board[action.piece][action.to] = 1
    if action.captured != 0:
        if s.turn == 0:
            s.hand[(action.captured - 6) % 4] += 1
        else:
            s.hand[action.captured % 4 + 2] += 1
    return s


#  駒打ちの処理
def drop(state: AnimalShogiState, action: AnimalShogiAction):
    s = copy.deepcopy(state)
    s.hand[action.piece - 1 - 2 * state.turn] -= 1
    s.board[action.piece][action.to] = 1
    s.board[0][action.to] = 0
    return s


#  ある座標に存在する駒種を返す
def piece_type(state: AnimalShogiState, point: int):
    return state.board[:, point].argmax()


# ある駒の持ち主を返す
def owner(piece: int):
    if piece == 0:
        return 2
    return (piece - 1) // 5


# 盤面のどこに何の駒があるかをnp.arrayに移したもの
# 同じ座標に複数回piece_typeを使用する場合はこちらを使った方が良い
def board_status(state: AnimalShogiState):
    board = np.zeros(12, dtype=np.int32)
    for i in range(12):
        board[i] = piece_type(state, i)
    return board


# 駒の持ち主の判定
def pieces_owner(state: AnimalShogiState):
    board = np.zeros(12, dtype=np.int32)
    for i in range(12):
        piece = piece_type(state, i)
        if piece == 0:
            board[i] = 2
        else:
            board[i] = (piece - 1) // 5
    return board


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
def is_side(point: int):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


# point(0~11)を座標(00~23)に変換
def convert_point(point: int):
    return point // 4, point % 4


# はみ出す部分をカットする
def cut_outside(array: np.ndarray, point: int):
    new_array = copy.deepcopy(array)
    u, d, l, r = is_side(point)
    if u:
        new_array[:, 0] *= 0
    if d:
        new_array[:, 2] *= 0
    if r:
        new_array[0, :] *= 0
    if l:
        new_array[2, :] *= 0
    return new_array


def return_board(array: np.ndarray, point: int):
    new_array = copy.deepcopy(array)
    y, t = convert_point(point)
    new_array = cut_outside(new_array, point)
    return np.roll(new_array, (y - 1, t - 1), axis=(0, 1))


# 各駒の動き
def black_pawn_move(point: int):
    return return_board(np.copy(BLACK_PAWN_MOVE), point)


def white_pawn_move(point: int):
    return return_board(np.copy(WHITE_PAWN_MOVE), point)


def black_gold_move(point: int):
    return return_board(np.copy(BLACK_GOLD_MOVE), point)


def white_gold_move(point: int):
    return return_board(np.copy(WHITE_GOLD_MOVE), point)


def rook_move(point: int):
    return return_board(np.copy(ROOK_MOVE), point)


def bishop_move(point: int):
    return return_board(np.copy(BISHOP_MOVE), point)


def king_move(point: int):
    return return_board(np.copy(KING_MOVE), point)


#  座標と駒の種類から到達できる座標を列挙する関数
def point_moves(piece: int, point: int):
    if piece == 1:
        return black_pawn_move(point)
    if piece == 6:
        return white_pawn_move(point)
    if piece % 5 == 2:
        return rook_move(point)
    if piece % 5 == 3:
        return bishop_move(point)
    if piece % 5 == 4:
        return king_move(point)
    if piece == 5:
        return black_gold_move(point)
    if piece == 10:
        return white_gold_move(point)


# 利きの判定
def effected(state: AnimalShogiState, turn: int):
    all_effect = np.zeros(12)
    board = board_status(state)
    piece_owner = pieces_owner(state)
    for i in range(12):
        own = piece_owner[i]
        if own != turn:
            continue
        piece = board[i]
        effect = point_moves(piece, i).reshape(12)
        all_effect += effect
    return all_effect


# 自殺手判定
def is_suicide(piece: int, position: int, effects):
    # ライオン以外は関係ない
    if piece % 5 != 4:
        return False
    # 行先に相手の駒の利きがあるかどうか
    return effects[position] != 0


# 王手放置判定
def leave_check(piece, position, check, cp):
    if not check:
        return False
    # 玉が動いていればとりあえず放置ではない（自殺手の可能性はある）
    if piece % 5 == 4:
        return False
    # 両王手などについてはどうぶつ将棋では考えない
    # cp[position] が1のところに動いていれば、王手を回避できている
    return cp[position] == 0


# 王手の判定(turn側の王に王手がかかっているかを判定)
def is_check(state: AnimalShogiState):
    effects = effected(state, another_color(state))
    king_location = state.board[4 + 5 * state.turn, :].argmax()
    return effects[king_location] != 0


# 成る動きが合法かどうかの判定
def can_promote(to: int, piece: int):
    if piece == 1 and to & 4 == 0:
        return True
    if piece == 6 and to % 4 == 3:
        return True
    return False


# 駒の種類と位置から生成できるactionのフラグを立てる
def create_actions(_from: int, piece: int):
    turn = owner(piece)
    actions = np.zeros(180, dtype=np.int32)
    motion = point_moves(piece, _from).reshape(12)
    for i in range(12):
        if motion[i] == 0:
            continue
        if can_promote(i, piece):
            pro_dir = point_to_direction(_from, i, True, turn)
            pro_act = dlshogi_action(pro_dir, i)
            actions[pro_act] = 1
        normal_dir = point_to_direction(_from, i, False, turn)
        normal_act = dlshogi_action(normal_dir, i)
        actions[normal_act] = 1
    return actions


# 駒の種類と位置から生成できるactionのフラグを立てる
def add_actions(_from: int, piece: int, array: np.ndarray):
    new_array = copy.deepcopy(array)
    actions = create_actions(_from, piece)
    for i in range(180):
        if actions[i] == 1:
            new_array[i] = 1
    return new_array


# 駒の種類と位置から生成できるactionのフラグを折る
def break_actions(_from: int, piece: int, array: np.ndarray):
    new_array = copy.deepcopy(array)
    actions = create_actions(_from, piece)
    for i in range(180):
        if actions[i] == 1:
            new_array[i] = 0
    return new_array


# 駒打ちのactionを追加する
def add_drop(piece: int, array: np.ndarray):
    new_array = copy.deepcopy(array)
    direction = hand_piece_to_dir(piece)
    for i in range(12):
        action = dlshogi_action(direction, i)
        new_array[action] = 1
    return new_array


# 駒打ちのactionを消去する
def break_drop(piece: int, array: np.ndarray):
    new_array = copy.deepcopy(array)
    direction = hand_piece_to_dir(piece)
    for i in range(12):
        action = dlshogi_action(direction, i)
        new_array[action] = 0
    return new_array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は必要ないが途中の盤面から実行するときなどに必要
def create_legal_actions(state: AnimalShogiState):
    s = copy.deepcopy(state)
    bs = board_status(s)
    # 移動の追加
    for i in range(12):
        piece = bs[i]
        if piece == 0:
            continue
        if piece <= 5:
            s.legal_actions_black = add_actions(
                i, piece, s.legal_actions_black
            )
        else:
            s.legal_actions_white = add_actions(
                i, piece, s.legal_actions_white
            )
    # 駒打ちの追加
    for i in range(3):
        if s.hand[i] != 0:
            s.legal_actions_black = add_drop(1 + i, s.legal_actions_black)
        if s.hand[i + 3] != 0:
            s.legal_actions_white = add_drop(6 + i, s.legal_actions_white)
    return s


# 駒の移動によるlegal_actionsの更新
def update_legal_actions_move(
    state: AnimalShogiState, action: AnimalShogiAction
):
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
        enemy_actions = s.legal_actions_white
    else:
        player_actions = s.legal_actions_white
        enemy_actions = s.legal_actions_black
    # 元の位置にいたときのフラグを折る
    new_player_actions = break_actions(
        action.from_, action.piece, player_actions
    )
    new_enemy_actions = enemy_actions
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = add_actions(
        action.to, action.piece, new_player_actions
    )
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if action.captured != 0:
        new_enemy_actions = break_actions(
            action.to, action.captured, enemy_actions
        )
        captured = convert_piece(action.captured)
        # にわとりの場合ひよこに変換
        if captured % 5 == 0:
            captured -= 4
        # 持ち駒の種類が増えた場合、駒打ちのactionを追加する
        if s.hand[captured - 1 - 2 * s.turn] == 0:
            new_player_actions = add_drop(captured, new_player_actions)
    if s.turn == 0:
        s.legal_actions_black = new_player_actions
        s.legal_actions_white = new_enemy_actions
    else:
        s.legal_actions_black = new_enemy_actions
        s.legal_actions_white = new_player_actions
    return s


# 駒打ちによるlegal_actionsの更新
def update_legal_actions_drop(
    state: AnimalShogiState, action: AnimalShogiAction
):
    s = copy.deepcopy(state)
    if s.turn == 0:
        player_actions = s.legal_actions_black
    else:
        player_actions = s.legal_actions_white
    # 移動後の位置からの移動のフラグを立てる
    new_player_actions = add_actions(action.to, action.piece, player_actions)
    # 持ち駒がもうない場合、その駒を打つフラグを折る
    if s.hand[action.piece - 1 - 2 * s.turn] == 1:
        new_player_actions = break_drop(action.piece, new_player_actions)
    if s.turn == 0:
        s.legal_actions_black = new_player_actions
    else:
        s.legal_actions_white = new_player_actions
    return s


# 自殺手を除く
def break_suicide(
    turn: int, king_sq: int, effects: np.ndarray, array: np.ndarray
):
    new_array = copy.deepcopy(array)
    moves = king_move(king_sq).reshape(12)
    for i in range(12):
        if moves[i] == 0:
            continue
        if effects[i] == 0:
            continue
        direction = point_to_direction(king_sq, i, False, turn)
        action = dlshogi_action(direction, i)
        new_array[action] = 0
    return new_array


# 王手放置を除く
def break_leave_check(
    turn: int, king_sq: int, check_piece: np.ndarray, array: np.ndarray
):
    new_array = copy.deepcopy(array)
    moves = king_move(king_sq).reshape(12)
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
        direction = point_to_direction(king_sq, i, False, turn)
        action = dlshogi_action(direction, i)
        new_array[action] = 1
    return new_array


#  駒打ち以外の合法手を列挙する
def legal_moves(state: AnimalShogiState, action_array: np.ndarray):
    board = board_status(state)
    piece_owner = pieces_owner(state)
    for i in range(12):
        if piece_owner[i] != state.turn:
            continue
        piece = board[i]
        points = point_moves(piece, i).reshape(12)
        for p in range(12):
            if points[p] == 0:
                continue
            if piece_owner[p] == state.turn:
                continue
            piece2 = board[p]
            # ひよこが最奥までいった場合、成るactionも追加する
            if piece == 1 and p % 4 == 0:
                m = AnimalShogiAction(False, piece, p, i, piece2, True)
                after = move(state, m)
                if is_check(after):
                    continue
                action = action_to_int(m, state.turn)
                action_array[action] = 1
            elif piece == 6 and p % 4 == 3:
                m = AnimalShogiAction(False, piece, p, i, piece2, True)
                after = move(state, m)
                if is_check(after):
                    continue
                action = action_to_int(m, state.turn)
                action_array[action] = 1
            m = AnimalShogiAction(False, piece, p, i, piece2, False)
            # mを行った後の盤面（手番はそのまま）
            after = move(state, m)
            # mを行った後も自分の玉に王手がかかっていてはいけない
            if is_check(after):
                continue
            action = action_to_int(m, state.turn)
            action_array[action] = 1
    return action_array


# 駒打ちの合法手の生成
def legal_drop(state: AnimalShogiState, action_array: np.ndarray):
    #  打てるのはヒヨコ、キリン、ゾウの三種
    for i in range(3):
        piece = i + 1 + 5 * state.turn
        # 対応する駒を持ってない場合は打てない
        if state.hand[i + 3 * state.turn] == 0:
            continue
        for j in range(12):
            # 駒がある場合は打てない
            if state.board[0][j] == 0:
                continue
            d = AnimalShogiAction(True, piece, j)
            s = drop(state, d)
            # 自玉が取られるような手は打てない
            if is_check(s):
                continue
            action = action_to_int(d, state.turn)
            action_array[action] = 1
    return action_array


def legal_actions(state: AnimalShogiState):
    action_array = np.zeros(180, dtype=np.int32)
    action_array = legal_moves(state, action_array)
    action_array = legal_drop(state, action_array)
    return action_array


# boardのlegal_actionsを利用して合法手を生成する
def legal_actions2(state: AnimalShogiState):
    if state.turn == 0:
        action_array = copy.deepcopy(state.legal_actions_black)
    else:
        action_array = copy.deepcopy(state.legal_actions_white)
    king_sq = state.board[4 + 5 * state.turn].argmax()
    # 王手放置を除く
    if state.checked:
        action_array = break_leave_check(
            state.turn, king_sq, state.checking_piece, action_array
        )
    # toが自分の駒の場合はそのactionは不可
    # 駒打ちの場合は相手の駒でもダメ
    own = pieces_owner(state)
    for i in range(12):
        for j in range(15):
            # 移動かつ自分の駒
            if j <= 8 and own[i] == state.turn:
                action_array[j * 12 + i] = 0
            # 駒打ちかつ空白でない
            if j > 8 and own[i] != 2:
                action_array[j * 12 + i] = 0
    # 自殺手を除く
    effects = effected(state, another_color(state))
    action_array = break_suicide(state.turn, king_sq, effects, action_array)
    # その他の反則手を除く
    # どうぶつ将棋の場合はなし
    return action_array
