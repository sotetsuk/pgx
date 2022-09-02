import numpy as np
from dataclasses import dataclass


@dataclass
class AnimalShogiState:
    turn: int = 0
    board: np.ndarray = np.zeros((11, 12), dtype=np.int32)
    hand: np.ndarray = np.zeros(6, dtype=np.int32)


DEFAULT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# boardは19×11の構造
# 1~12行目は各座標に存在する駒種、13~18行目はお互いの持ち駒（3種×2）を表現。19行目は手番の情報
# turn 先手番なら0 後手番なら1


# 手番を変更する
def new_turn_change(state: AnimalShogiState):
    return (state.turn + 1) % 2


def turn_change(board, turn):
    b = np.copy(board)
    if turn == 0:
        b[18] = np.roll(b[18], 1)
    else:
        b[18] = np.roll(b[18], -1)
    return b


#  駒打ちでない移動の処理
#  board: 現在の盤面
#  turn: 現在の手番
#  fir_lo: 移動前の座標
#  fir_lo: 移動後の座標
#  piece: 動かした駒の種類（ヒヨコ1, キリン2, ゾウ3, ライオン4, ニワトリ5）
#  captured: 取られた駒の種類。駒が取られていない場合は0でそれ以外はpieceと同じ
#  is_promote: 駒を成るかどうかの判定
def new_move(state: AnimalShogiState, first: int, final: int, piece: int, captured: int, is_promote: int):
    state.board[piece][first] = 0
    state.board[0][first] = 1
    state.board[captured][final] = 0
    state.board[piece + 4 * is_promote][final] = 1
    if captured != 0:
        if state.turn == 0:
            state.hand[(captured - 6) % 4] += 1
        else:
            state.hand[captured % 4 + 2] += 1
    state.turn = new_turn_change(state)
    return state


def move(board, turn, fir_lo, fin_lo, piece, captured, is_promote):
    b = turn_change(board, turn)
    b[fir_lo] = DEFAULT
    b[fin_lo] = np.roll(DEFAULT, piece + 5 * turn + 4 * is_promote)
    if captured == 0:
        return b
    b[11 + captured % 4 + 3 * turn] = np.roll(
        b[11 + captured % 4 + 3 * turn], 1
    )
    return b


#  駒打ちの処理
#  point: 駒を打つ座標
#  piece: 打つ駒の種類。ライオン、ニワトリは打てないのでそれ以外の三種から選ぶ
def new_drop(state: AnimalShogiState, point: int, piece: int):
    state.hand[piece - 1 - 2 * state.turn] -= 1
    state.board[piece][point] = 1
    state.board[0][point] = 0
    state.turn = new_turn_change(state)


def drop(board, turn, point, piece):
    b = turn_change(board, turn)
    b[11 + piece + 3 * turn] = np.roll(b[11 + piece + 3 * turn], -1)
    b[point] = np.roll(b[point], piece + 5 * turn)
    return b


#  ある座標に存在する駒の持ち主と種類を返す
#  持ち主はturnに対応させるため先手0後手1、駒が存在しない場合は2を返す
#  駒の種類は上のpieceと対応
def owner_piece(board, point):
    ind = np.where(board[point] == 1)[0][0]
    # 駒がない位置
    if ind == 0:
        return 2, 0
    # 駒がある位置
    else:
        return (ind - 1) // 5, (ind - 1) % 5 + 1


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
def point_moves(turn, point, piece):
    if piece == 1:
        return hiyoko_move(turn, point)
    if piece == 2:
        return kirin_move(point)
    if piece == 3:
        return zou_move(point)
    if piece == 4:
        return lion_move(point)
    if piece == 5:
        return niwatori_move(turn, point)


#  駒打ち以外の合法手を列挙する
def legal_moves(board, turn):
    moves = []
    for i in range(12):
        owner, piece = owner_piece(board, i)
        if owner == turn:
            points = point_moves(turn, i, piece)
            for p in points:
                owner2, piece2 = owner_piece(board, p)
                # 自分の駒がある場所には動けない
                if owner2 == turn:
                    continue
                # ひよこが最奥までいった場合、強制的に成る
                if piece == 1 and turn == 0 and p % 4 == 0:
                    moves.append([i, p, piece, piece2, 1])
                elif piece == 1 and turn == 1 and p % 4 == 3:
                    moves.append([i, p, piece, piece2, 1])
                else:
                    moves.append([i, p, piece, piece2, 0])
    return moves


# 駒打ちの合法手の生成
def legal_drop(board, turn):
    moves = []
    #  打てるのはヒヨコ、キリン、ゾウの三種
    for i in range(3):
        piece = i + 1
        # 対応する駒を持ってない場合は打てない
        # 空白位置のベクトルと持ち駒を持っていないときのベクトルが同一であることを利用(DEFAULTとの比較演算ができなかった)
        if owner_piece(board, 11 + piece + turn * 3)[0] == 2:
            continue
        for j in range(12):
            # ひよこは最奥には打てない
            if piece == 1 and turn == 0 and j % 4 == 0:
                continue
            if piece == 1 and turn == 1 and j % 4 == 3:
                continue
            owner = owner_piece(board, j)[0]
            # お互いの駒がない地点(==ownerが2の地点)であれば打てる
            if owner == 2:
                moves.append([j, piece])
    return moves


#  全ての合法手の生成
def legal_drop_moves(board, turn):
    moves = legal_moves(board, turn)
    drops = legal_drop(board, turn)
    all_moves = []
    # 移動には0, 駒打ちには1でラベル付けをする
    for m in moves:
        all_moves.append((0, m))
    for d in drops:
        all_moves.append((1, d))
    return all_moves
