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


# 手番側でない色を返す
def another_color(state: AnimalShogiState):
    return (state.turn + 1) % 2


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


#  駒打ち以外の合法手を列挙する
def legal_moves(state: AnimalShogiState):
    board = board_status(state)
    piece_owner = pieces_owner(state)
    # 相手の駒の利き
    effects = effected(state, another_color(state))
    moves = []
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
            # ひよこが最奥までいった場合、強制的に成る
            # なぜかpieceが小数に変換されてしまうのでとりあえずintに変換しておく
            if piece == 1 and p % 4 == 0:
                m = AnimalShogiAction(0, piece, p, i, piece2, 1)
            elif piece == 6 and p % 4 == 3:
                m = AnimalShogiAction(0, piece, p, i, piece2, 1)
            else:
                m = AnimalShogiAction(0, piece, p, i, piece2, 0)
            # mを行った後の盤面（手番はそのまま）
            after = move(state, m)
            # mを行った後も自分の玉に王手がかかっていてはいけない
            if is_check(after):
                continue
            moves.append(m)
    return moves


# 駒打ちの合法手の生成
def legal_drop(state: AnimalShogiState):
    moves = []
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
            # ひよこは最奥には打てない
            if piece == 1 and j % 4 == 0:
                continue
            if piece == 6 and j % 4 == 3:
                continue
            d = AnimalShogiAction(1, piece, j)
            s = drop(state, d)
            # 自玉が取られるような手は打てない
            if is_check(s):
                continue
            moves.append(d)
    return moves
