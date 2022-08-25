import numpy as np

# boardは19×11の構造
# 1~12行目は各座標に存在する駒種、13~18行目はお互いの持ち駒（3種×2）を表現。19行目は手番の情報
# turn 先手番なら0 後手番なら1
DEFAULT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
INIT_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 11(右上) 後手のゾウ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 空白
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 空白
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 14(右下) 先手のキリン
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 21 後手ライオン
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 22 後手ヒヨコ
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 23 先手ヒヨコ
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 24 先手ライオン
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 31 後手キリン
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 32 空白
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 33 空白
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 34 先手ゾウ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 先手ヒヨコ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 先手キリン
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 先手ゾウ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 後手ヒヨコ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 後手キリン
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 持ち駒 後手ゾウ
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 手番の情報
])


def turn_change(board, turn):
    """
    >>> turn_change(INIT_BOARD, 0)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    b = np.copy(board)
    if turn == 0:
        b[18] = np.roll(b[18], 1)
    else:
        b[18] = np.roll(b[18], -1)
    return b


def move(board, turn, fir_lo, fin_lo, piece, captured, is_promote):
    """
    >>> move(INIT_BOARD, 0, 6, 5, 1, 1, 0)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    b = turn_change(board, turn)
    b[fir_lo] = DEFAULT
    b[fin_lo] = np.roll(DEFAULT, piece+5*turn+4*is_promote)
    if captured == 0:
        return b
    b[11+captured % 4+3*turn] = np.roll(b[11+captured % 4+3*turn], 1)
    return b


def drop(board, turn, point, piece):
    """
    >>> drop(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 0, 2, 2)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    b = turn_change(board, turn)
    b[11+piece+3*turn] = np.roll(b[11+piece+3*turn], -1)
    b[point] = np.roll(b[point], piece + 5*turn)
    return b


def owner_piece(board, point):
    """
    >>> owner_piece(INIT_BOARD, 3)
    (0, 2)
    >>> owner_piece(INIT_BOARD, 5)
    (1, 1)
    >>> owner_piece(INIT_BOARD, 9)
    (2, 0)
    """
    ind = np.where(board[point] == 1)[0][0]
    # 駒がない位置
    if ind == 0:
        return 2, 0
    # 駒がある位置
    else:
        return (ind-1)//5, (ind-1) % 5 + 1


# 上下左右の辺に接しているかどうか
def is_side(point):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


def hiyoko_move(turn, point):
    #  最奥にいてはいけない
    if turn == 0:
        assert point % 4 != 0
        return [point-1]
    else:
        assert point % 4 != 3
        return [point+1]


def kirin_move(point):
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        moves.append(point-4)
    if not u:
        moves.append(point-1)
    if not d:
        moves.append(point+1)
    if not l:
        moves.append(point+4)
    return moves


def zou_move(point):
    u, d, l, r = is_side(point)
    moves = []
    if not r:
        if not u:
            moves.append(point-5)
        if not d:
            moves.append(point-3)
    if not l:
        if not u:
            moves.append(point+3)
        if not d:
            moves.append(point+5)
    return moves


def lion_move(point):
    m1 = kirin_move(point)
    m2 = zou_move(point)
    m1.extend(m2)
    return m1


def niwatori_move(turn, point):
    moves = kirin_move(point)
    u, d, l, r = is_side(point)
    if turn == 0:
        if not u:
            if not r:
                moves.append(point-5)
            if not l:
                moves.append(point+3)
    else:
        if not d:
            if not r:
                moves.append(point-3)
            if not l:
                moves.append(point+5)
    return moves


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


def legal_moves(board, turn):
    """
    >>> legal_moves(INIT_BOARD, 0)
    [[3, 2, 2, 0, 0], [6, 5, 1, 1, 0], [7, 2, 4, 0, 0], [7, 10, 4, 0, 0]]
    >>> legal_moves(INIT_BOARD, 1)
    [[4, 1, 4, 0, 0], [4, 9, 4, 0, 0], [5, 6, 1, 1, 0], [8, 9, 2, 0, 0]]
    """
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
                if piece == 1 and p % 4 == 0:
                    moves.append([i, p, piece, piece2, 1])
                else:
                    moves.append([i, p, piece, piece2, 0])
    return moves


def legal_drop(board, turn):
    moves = []
    #  打てるのはヒヨコ、キリン、ゾウの三種
    for i in range(3):
        piece = i + 1
        # 対応する駒を持ってない場合は打てない
        # 空白位置のベクトルと持ち駒を持っていないときのベクトルが同一であることを利用(DEFAULTとの比較演算ができなかった)
        if owner_piece(board, 11+piece+turn*3)[0] == 2:
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
