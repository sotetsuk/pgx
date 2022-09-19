import copy
from dataclasses import dataclass

import numpy as np


# 指し手のdataclass
@dataclass
class AnimalShogiAction:
    # 上の3つは移動と駒打ちで共用
    # 下の3つは移動でのみ使用
    # 駒打ちかどうか
    is_drop: int
    # piece: 動かした(打った)駒の種類
    piece: int
    # final: 移動後の座標
    final: int
    # 移動前の座標
    first: int = 0
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: int = 0
    # is_promote: 駒を成るかどうかの判定
    is_promote: int = 0


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


# dlshogiのactionはdirection(動きの方向)とto（駒の処理後の座標）に依存
def dlshogi_action(direction, to):
    return direction * 12 + to


# fromの座標とtoの座標からdirを生成
def point_to_direction(fro, to, promote, turn):
    direction = -1
    dis = to - fro
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
def hand_piece_to_dir(piece):
    # 移動のdirはPROMOTE_UPの8が最大なので9以降に配置
    # 9: 先手ヒヨコ 10: 先手キリン... 14: 後手ゾウ　に対応させる
    if piece <= 5:
        return 8 + piece
    else:
        return 6 + piece


# AnimalShogiActionをdlshogiのint型actionに変換
def action_to_int(act: AnimalShogiAction, turn):
    if act.is_drop == 0:
        return dlshogi_action(
            point_to_direction(act.first, act.final, act.is_promote, turn),
            act.final,
        )
    else:
        return dlshogi_action(hand_piece_to_dir(act.piece), act.final)


# dlshogiのint型actionをdirectionとtoに分解
def separate_int(act):
    # direction, to の順番
    return act // 12, act % 12


# directionからfromがtoからどれだけ離れてるかと成りを含む移動かを得る
# 手番の情報が必要
def direction_to_from(direction, to, turn):
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


def direction_to_hand_piece(direction):
    if direction <= 11:
        return direction - 8
    else:
        return direction - 6


def int_to_action(act, state: AnimalShogiState):
    direction, to = separate_int(act)
    if direction <= 8:
        # 駒の移動
        is_drop = 0
        fro, is_promote = direction_to_from(direction, to, state.turn)
        piece = piece_type(state, fro)
        captured = piece_type(state, to)
        return AnimalShogiAction(is_drop, piece, to, fro, captured, is_promote)
    else:
        # 駒打ち
        is_drop = 1
        piece = direction_to_hand_piece(direction)
        return AnimalShogiAction(is_drop, piece, to)


# 手番側でない色を返す
def another_color(state: AnimalShogiState):
    return (state.turn + 1) % 2


# 相手の駒を同じ種類の自分の駒に変換する
def convert_piece(piece):
    p = (piece + 5) % 10
    if p == 0:
        return 10
    else:
        return p


#  駒打ちでない移動の処理 手番変更、盤面書き換えなし
def move(
    state: AnimalShogiState,
    act: AnimalShogiAction,
):
    s = copy.deepcopy(state)
    s.board[act.piece][act.first] = 0
    s.board[0][act.first] = 1
    s.board[act.captured][act.final] = 0
    s.board[act.piece + 4 * act.is_promote][act.final] = 1
    if act.captured != 0:
        if s.turn == 0:
            s.hand[(act.captured - 6) % 4] += 1
        else:
            s.hand[act.captured % 4 + 2] += 1
    return s


#  駒打ちの処理 手番変更、盤面書き換えなし
def drop(state: AnimalShogiState, act: AnimalShogiAction):
    s = copy.deepcopy(state)
    s.hand[act.piece - 1 - 2 * state.turn] -= 1
    s.board[act.piece][act.final] = 1
    s.board[0][act.final] = 0
    return s


# stateとactを受け取りis_dropによって操作を分ける
# 手番、王手判定も更新。引数の盤面も書き換える
def action(state: AnimalShogiState, act: AnimalShogiAction):
    if act.is_drop == 1:
        state = drop(state, act)
    else:
        state = move(state, act)
    state.turn = another_color(state)
    state.checked = is_check(state)
    # 王手をかけている駒は直前に動かした駒
    if state.checked:
        state.checking_piece[act.final] = 1
    else:
        state.checking_piece = np.zeros(12, dtype=np.int32)
    return state


#  ある座標に存在する駒種を返す
def piece_type(state: AnimalShogiState, point: int):
    return state.board[:, point].argmax()


# ある駒の持ち主を返す
def owner(piece):
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
def is_side(point):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


# point(0~11)を座標(00~23)に変換
def convert_point(point):
    return point // 4, point % 4


# はみ出す部分をカットする
def cut_outside(array, point):
    u, d, l, r = is_side(point)
    if u:
        array[:, 0] *= 0
    if d:
        array[:, 2] *= 0
    if r:
        array[0, :] *= 0
    if l:
        array[2, :] *= 0


def return_board(array, point):
    y, t = convert_point(point)
    cut_outside(array, point)
    return np.roll(array, (y - 1, t - 1), axis=(0, 1))


# 各駒の動き
def black_pawn_move(point):
    return return_board(np.copy(BLACK_PAWN_MOVE), point)


def white_pawn_move(point):
    return return_board(np.copy(WHITE_PAWN_MOVE), point)


def black_gold_move(point):
    return return_board(np.copy(BLACK_GOLD_MOVE), point)


def white_gold_move(point):
    return return_board(np.copy(WHITE_GOLD_MOVE), point)


def rook_move(point):
    return return_board(np.copy(ROOK_MOVE), point)


def bishop_move(point):
    return return_board(np.copy(BISHOP_MOVE), point)


def king_move(point):
    return return_board(np.copy(KING_MOVE), point)


#  座標と駒の種類から到達できる座標を列挙する関数
def point_moves(piece, point):
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
def is_suicide(piece, position, effects):
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
def can_promote(to, piece):
    if piece == 1 and to & 4 == 0:
        return True
    if piece == 6 and to % 4 == 3:
        return True
    return False


# 駒の種類と位置から生成できるactionのフラグを立てる
def create_actions(fro, piece):
    turn = owner(piece)
    actions = np.zeros(180, dtype=np.int32)
    motion = point_moves(piece, fro).reshape(12)
    for i in range(12):
        if motion[i] == 0:
            continue
        if can_promote(i, piece):
            pro_dir = point_to_direction(fro, i, True, turn)
            pro_act = dlshogi_action(pro_dir, i)
            actions[pro_act] = 1
        normal_dir = point_to_direction(fro, i, False, turn)
        normal_act = dlshogi_action(normal_dir, i)
        actions[normal_act] = 1
    return actions


# 駒の種類と位置から生成できるactionのフラグを立てる
def add_actions(fro, piece, array):
    actions = create_actions(fro, piece)
    for i in range(180):
        if actions[i] == 1:
            array[i] = 1
    return array


# 駒の種類と位置から生成できるactionのフラグを折る
def break_actions(fro, piece, array):
    actions = create_actions(fro, piece)
    for i in range(180):
        if actions[i] == 1:
            array[i] = 0
    return array


# 駒打ちのactionを追加する
def add_drop(piece, array):
    direction = hand_piece_to_dir(piece)
    for i in range(12):
        act = dlshogi_action(direction, i)
        array[act] = 1
    return array


# 駒打ちのactionを消去する
def break_drop(piece, array):
    direction = hand_piece_to_dir(piece)
    for i in range(12):
        act = dlshogi_action(direction, i)
        array[act] = 0
    return array


# stateからblack,white両方のlegal_actionsを生成する
# 普段は必要ないが途中の盤面から実行するときなどに必要
def create_legal_actions(state: AnimalShogiState):
    bs = board_status(state)
    # 移動の追加
    for i in range(12):
        piece = bs[i]
        if piece == 0:
            continue
        if piece <= 5:
            state.legal_actions_black = add_actions(
                i, piece, state.legal_actions_black
            )
        else:
            state.legal_actions_white = add_actions(
                i, piece, state.legal_actions_white
            )
    # 駒打ちの追加
    for i in range(3):
        if state.hand[i] != 0:
            state.legal_actions_black = add_drop(
                1 + i, state.legal_actions_black
            )
        if state.hand[i + 3] != 0:
            state.legal_actions_white = add_drop(
                6 + i, state.legal_actions_white
            )
    return state


# 駒の移動によるlegal_actionsの更新
def update_legal_actions_move(
    act: AnimalShogiAction, player_actions, enemy_actions, hand
):
    # 元の位置にいたときのフラグを折る
    break_actions(act.first, act.piece, player_actions)
    # 移動後の位置からの移動のフラグを立てる
    add_actions(act.final, act.piece, player_actions)
    # 駒が取られた場合、相手の取られた駒によってできていたactionのフラグを折る
    if act.captured != 0:
        break_actions(act.final, act.captured, enemy_actions)
        # 持ち駒の種類が増えた場合、駒打ちのactionを追加する
        if hand == 0:
            add_drop(convert_piece(act.captured), player_actions)
    return player_actions, enemy_actions


# 駒打ちによるlegal_actionsの更新
def update_legal_actions_drop(
    act: AnimalShogiAction, player_actions, enemy_actions, hand
):
    # 移動後の位置からの移動のフラグを立てる
    add_actions(act.final, act.piece, player_actions)
    # 持ち駒が最後の一枚だった場合、駒打ちのactionを減らす
    if hand == 1:
        break_drop(act.piece, player_actions)
    return player_actions, enemy_actions


# 自殺手を除く
def break_suicide(turn, king_sq, effects, array):
    moves = king_move(king_sq).reshape(12)
    for i in range(12):
        if moves[i] == 0:
            continue
        if effects[i] == 0:
            continue
        direction = point_to_direction(king_sq, i, 0, turn)
        act = dlshogi_action(direction, i)
        array[act] = 0
    return array


# 王手放置を除く
def break_leave_check(turn, king_sq, check_piece, array):
    moves = king_move(king_sq).reshape(12)
    for i in range(12):
        # 王手をかけている駒の位置以外への移動は王手放置
        for j in range(15):
            # 駒打ちのフラグは全て折る
            if j > 8:
                array[12 * j + i] = 0
            # 王手をかけている駒の場所以外への移動ははじく
            if check_piece[i] == 0:
                array[12 * j + i] = 0
        # 玉の移動はそれ以外でも可能だがフラグが折れてしまっているので立て直す
        if moves[i] == 0:
            continue
        direction = point_to_direction(king_sq, i, 0, turn)
        act = dlshogi_action(direction, i)
        array[act] = 1
    return array


#  駒打ち以外の合法手を列挙する
def legal_moves(state: AnimalShogiState, action_array):
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
                m = AnimalShogiAction(0, piece, p, i, piece2, 1)
                after = move(state, m)
                if is_check(after):
                    continue
                act = action_to_int(m, state.turn)
                action_array[act] = 1
            elif piece == 6 and p % 4 == 3:
                m = AnimalShogiAction(0, piece, p, i, piece2, 1)
                after = move(state, m)
                if is_check(after):
                    continue
                act = action_to_int(m, state.turn)
                action_array[act] = 1
            m = AnimalShogiAction(0, piece, p, i, piece2, 0)
            # mを行った後の盤面（手番はそのまま）
            after = move(state, m)
            # mを行った後も自分の玉に王手がかかっていてはいけない
            if is_check(after):
                continue
            act = action_to_int(m, state.turn)
            action_array[act] = 1
    return action_array


# 駒打ちの合法手の生成
def legal_drop(state: AnimalShogiState, action_array):
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
            d = AnimalShogiAction(1, piece, j)
            s = drop(state, d)
            # 自玉が取られるような手は打てない
            if is_check(s):
                continue
            act = action_to_int(d, state.turn)
            action_array[act] = 1
    return action_array


def legal_actions(state: AnimalShogiState):
    action_array = np.zeros(180, dtype=np.int32)
    legal_moves(state, action_array)
    legal_drop(state, action_array)
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
