from dataclasses import dataclass

import numpy as np


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


# 移動のdataclass
@dataclass
class AnimalShogiMove:
    # first: 移動前の座標
    first: int
    # final: 移動後の座標
    final: int
    # piece: 動かした駒の種類
    piece: int
    # captured: 取られた駒の種類。駒が取られていない場合は0
    captured: int
    # is_promote: 駒を成るかどうかの判定
    is_promote: int


# 駒打ちのdataclass
@dataclass
class AnimalShogiDrop:
    # piece: 打った駒の種類
    piece: int
    # point: 駒を打った座標
    point: int


# 手番を変更する
def turn_change(state: AnimalShogiState):
    state.turn = (state.turn + 1) % 2
    return state


#  駒打ちでない移動の処理
def move(
    state: AnimalShogiState,
    mov: AnimalShogiMove,
):
    state.board[mov.piece][mov.first] = 0
    state.board[0][mov.first] = 1
    state.board[mov.captured][mov.final] = 0
    state.board[mov.piece + 4 * mov.is_promote][mov.final] = 1
    if mov.captured != 0:
        if state.turn == 0:
            state.hand[(mov.captured - 6) % 4] += 1
        else:
            state.hand[mov.captured % 4 + 2] += 1
    state = turn_change(state)
    return state


#  駒打ちの処理
def drop(state: AnimalShogiState, dro: AnimalShogiDrop):
    state.hand[dro.piece - 1 - 2 * state.turn] -= 1
    state.board[dro.piece][dro.point] = 1
    state.board[0][dro.point] = 0
    state = turn_change(state)
    return state


#  ある座標に存在する駒種を返す
def piece_type(state: AnimalShogiState, point: int):
    return state.board[:, point].argmax()


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
def is_side(point):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


#  各駒の動き
def pawn_move(turn, point):
    #  最奥にいてはいけない
    if turn == 0:
        assert point % 4 != 0
        return [point - 1]
    else:
        assert point % 4 != 3
        return [point + 1]


def rook_move(point):
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        moves.append(point - 4)
    if not u:
        moves.append(point - 1)
    if not d:
        moves.append(point + 1)
    if not l:
        moves.append(point + 4)
    return moves


def bishop_move(point):
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        if not u:
            moves.append(point - 5)
        if not d:
            moves.append(point - 3)
    if not l:
        if not u:
            moves.append(point + 3)
        if not d:
            moves.append(point + 5)
    return moves


def king_move(point):
    #  座標が小さい順に並ぶようにする
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        if not u:
            moves.append(point - 5)
        moves.append(point - 4)
        if not d:
            moves.append(point - 3)
    if not u:
        moves.append(point - 1)
    if not d:
        moves.append(point + 1)
    if not l:
        if not u:
            moves.append(point + 3)
        moves.append(point + 4)
        if not d:
            moves.append(point + 5)
    return moves


def gold_move(turn, point):
    #  座標が小さい順に並ぶようにする
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        if not u and turn == 0:
            moves.append(point - 5)
        moves.append(point - 4)
        if not d and turn == 1:
            moves.append(point - 3)
    if not u:
        moves.append(point - 1)
    if not d:
        moves.append(point + 1)
    if not l:
        if not u and turn == 0:
            moves.append(point + 3)
        moves.append(point + 4)
        if not d and turn == 1:
            moves.append(point + 5)
    return moves


#  座標と駒の種類から到達できる座標を列挙する関数
def point_moves(piece, point):
    turn = (piece - 1) // 5
    if piece % 5 == 1:
        return pawn_move(turn, point)
    if piece % 5 == 2:
        return rook_move(point)
    if piece % 5 == 3:
        return bishop_move(point)
    if piece % 5 == 4:
        return king_move(point)
    if piece % 5 == 0:
        return gold_move(turn, point)


#  駒打ち以外の合法手を列挙する
def legal_moves(state: AnimalShogiState):
    moves = []
    for i in range(12):
        piece = piece_type(state, i)
        if (piece - 1) // 5 == state.turn:
            points = point_moves(piece, i)
            for p in points:
                piece2 = piece_type(state, p)
                # 自分の駒がある場所には動けない
                if (piece2 - 1) // 5 == state.turn:
                    continue
                # ひよこが最奥までいった場合、強制的に成る
                if piece == 1 and p % 4 == 0:
                    moves.append(AnimalShogiMove(i, p, piece, piece2, 1))
                elif piece == 6 and p % 4 == 3:
                    moves.append(AnimalShogiMove(i, p, piece, piece2, 1))
                else:
                    moves.append(AnimalShogiMove(i, p, piece, piece2, 0))
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
            # ひよこは最奥には打てない
            if piece == 1 and j % 4 == 0:
                continue
            if piece == 6 and j % 4 == 3:
                continue
            piece2 = piece_type(state, j)
            # お互いの駒がない地点(==piece2が0の地点)であれば打てる
            if piece2 == 0:
                moves.append(AnimalShogiDrop(j, piece))
    return moves
