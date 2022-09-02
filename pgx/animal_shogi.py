import numpy as np
from dataclasses import dataclass


@dataclass
class AnimalShogiState:
    # turn 先手番なら0 後手番なら1
    turn: int = 0
    # board 盤面の駒。空白、先手ヒヨコ、…、後手ライオン、後手ニワトリ、の順で駒がどの位置にあるかを記録
    board: np.ndarray = np.zeros((11, 12), dtype=np.int32)
    # hand 持ち駒。先手ヒヨコ～後手ゾウまでの6種類を枚数に応じて増減させる
    hand: np.ndarray = np.zeros(6, dtype=np.int32)


# 手番を変更する
def turn_change(state: AnimalShogiState):
    state.turn = (state.turn + 1) % 2
    return state


#  駒打ちでない移動の処理
#  first: 移動前の座標
#  final: 移動後の座標
#  piece: 動かした駒の種類
#  captured: 取られた駒の種類。駒が取られていない場合は0
#  is_promote: 駒を成るかどうかの判定
def move(state: AnimalShogiState, first: int, final: int, piece: int, captured: int, is_promote: int):
    state.board[piece][first] = 0
    state.board[0][first] = 1
    state.board[captured][final] = 0
    state.board[piece + 4 * is_promote][final] = 1
    if captured != 0:
        if state.turn == 0:
            state.hand[(captured - 6) % 4] += 1
        else:
            state.hand[captured % 4 + 2] += 1
    state = turn_change(state)
    return state


#  駒打ちの処理
#  point: 駒を打つ座標
#  piece: 打つ駒の種類。ライオン、ニワトリは打てないのでそれ以外の三種から選ぶ
def drop(state: AnimalShogiState, point: int, piece: int):
    state.hand[piece - 1 - 2 * state.turn] -= 1
    state.board[piece][point] = 1
    state.board[0][point] = 0
    state = turn_change(state)
    return state


#  ある座標に存在する駒の持ち主と種類を返す
#  持ち主はturnに対応させるため先手0後手1、駒が存在しない場合は2を返す
#  駒の種類は上のpieceと対応
def owner_piece(state: AnimalShogiState, point: int):
    for i in range(11):
        if state.board[i][point] == 1:
            return i


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
def is_side(point):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


#  各駒の動き
def hiyoko_move(turn, point):
    #  最奥にいてはいけない
    if turn == 0:
        assert point % 4 != 0
        return [point - 1]
    else:
        assert point % 4 != 3
        return [point + 1]


def kirin_move(point):
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


def zou_move(point):
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


def lion_move(point):
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


def niwatori_move(turn, point):
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
        return hiyoko_move(turn, point)
    if piece % 5 == 2:
        return kirin_move(point)
    if piece % 5 == 3:
        return zou_move(point)
    if piece % 5 == 4:
        return lion_move(point)
    if piece % 5 == 0:
        return niwatori_move(turn, point)


#  駒打ち以外の合法手を列挙する
def legal_moves(state: AnimalShogiState):
    moves = []
    for i in range(12):
        piece = owner_piece(state, i)
        if (piece - 1) // 5 == state.turn:
            points = point_moves(piece, i)
            for p in points:
                piece2 = owner_piece(state, p)
                # 自分の駒がある場所には動けない
                if (piece2 - 1) // 5 == state.turn:
                    continue
                # ひよこが最奥までいった場合、強制的に成る
                if piece == 1 and p % 4 == 0:
                    moves.append([i, p, piece, piece2, 1])
                elif piece == 6 and p % 4 == 3:
                    moves.append([i, p, piece, piece2, 1])
                else:
                    moves.append([i, p, piece, piece2, 0])
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
            piece2 = owner_piece(state, j)
            # お互いの駒がない地点(==piece2が0の地点)であれば打てる
            if piece2 == 0:
                moves.append([j, piece])
    return moves

