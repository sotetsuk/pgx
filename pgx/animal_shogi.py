import numpy as np

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


class Board:
    def __init__(self, board=INIT_BOARD, turn=0):
        # boardは19×11の構造
        # 1~12行目は各座標に存在する駒種、13~18行目はお互いの持ち駒（3種×2）を表現。19行目は手番の情報
        # turn 先手番なら0 後手番なら1
        self.board = board
        self.turn = turn

    def turn_change(self):
        if self.turn == 0:
            self.board[18] = np.roll(self.board[18], 1)
        else:
            self.board[18] = np.roll(self.board[18], -1)

    # fir_lo 動かす前の位置
    # fin_lo 動かしたあとの位置
    # piece, captured 動かす駒と取られる駒 ひよこが1 きりん2 ぞう3 ライオン4　にわとり5　capturedは駒取りでない場合0
    # is_promote 駒を成るかどうか
    def move(self, fir_lo, fin_lo, piece, captured, is_promote):
        self.turn_change()
        self.board[fir_lo] = DEFAULT
        self.board[fin_lo] = np.roll(DEFAULT, piece+5*self.turn+4*is_promote)
        if captured == 0:
            return
        self.board[11+captured % 4+3*self.turn] = np.roll(self.board[11+captured % 4+3*self.turn], 1)
        return

    def drop(self, point, piece):
        self.turn_change()
        self.board[11+piece+3*self.turn] = np.roll(self.board[11+piece+3*self.turn], -1)
        self.board[point] = np.roll(self.board[point], piece + 5*self.turn)
        return

    def owner_piece(self, point):
        ind = np.where(self.board[point] == 1)[0][0]
        # 駒がない位置
        if ind == 0:
            return 2, 0
        # 駒がある位置
        else:
            return (ind-1)//5, (ind-1) % 5 + 1

    # 上下左右の編に接しているかどうか
    def is_side(self, point):
        is_up = point % 4 == 0
        is_down = point % 4 == 3
        is_left = point >= 8
        is_right = point <= 3
        return is_up, is_down, is_left, is_right

    def hiyoko_move(self, point):
        if self.turn == 0:
            return [point-1]
        else:
            return [point+1]

    def kirin_move(self, point):
        u, d, l, r = self.is_side(point)
        moves = []
        if not u:
            moves.append(point-1)
        if not d:
            moves.append(point+1)
        if not l:
            moves.append(point+4)
        if not r:
            moves.append(point-4)
        return moves

    def zou_move(self, point):
        u, d, l, r = self.is_side(point)
        moves = []
        if not u:
            if not l:
                moves.append(point+3)
            if not r:
                moves.append(point-5)
        if not d:
            if not l:
                moves.append(point+5)
            if not r:
                moves.append(point-3)
        return moves

    def lion_move(self, point):
        m1 = self.kirin_move(point)
        m2 = self.zou_move(point)
        m1.extend(m2)
        return m1

    def niwatori_move(self, point):
        moves = self.kirin_move(point)
        u, d, l, r = self.is_side(point)
        if self.turn == 0:
            if not u:
                if not l:
                    moves.append(point+3)
                if not r:
                    moves.append(point-5)
        else:
            if not d:
                if not l:
                    moves.append(point+5)
                if not r:
                    moves.append(point-3)
        return moves

    def point_moves(self, point, piece):
        if piece == 1:
            return self.hiyoko_move(point)
        if piece == 2:
            return self.kirin_move(point)
        if piece == 3:
            return self.zou_move(point)
        if piece == 4:
            return self.lion_move(point)
        if piece == 5:
            return self.niwatori_move(point)

    def legal_moves(self):
        moves = []
        for i in range(12):
            owner, piece = self.owner_piece(i)
            if owner == self.turn:
                points = self.point_moves(i, piece)
                for p in points:
                    owner2, piece2 = self.owner_piece(p)
                    # 自分の駒がある場所には動けない
                    if owner2 == self.turn:
                        continue
                    # ひよこが最奥までいった場合、強制的に成る
                    if piece == 1 and p % 4 == 0:
                        moves.append([i, p, piece, piece2, 1])
                    else:
                        moves.append([i, p, piece, piece2, 0])
        return moves

    def legal_drop(self):
        moves = []
        for i in range(3):
            piece = i + 1
            # 対応する駒を持ってない場合は打てない
            # 空白位置のベクトルと持ち駒を持っていないときのベクトルが同一であることを利用
            if self.owner_piece(11+piece+self.turn*3)[0] == 2:
                continue
            for j in range(12):
                # ひよこは最奥には打てない
                if piece == 1 and self.turn == 0 and j % 4 == 0:
                    continue
                if piece == 1 and self.turn == 1 and j % 4 == 3:
                    continue
                owner = self.owner_piece(j)[0]
                # お互いの駒がない地点(==ownerが2の地点)であれば打てる
                if owner == 2:
                    moves.append([j, piece])
        return moves

    def legal_drop_moves(self):
        moves = self.legal_moves()
        drop = self.legal_drop()
        all_moves = []
        # 移動には0, 駒打ちには1でラベル付けをする
        for m in moves:
            all_moves.append((0, m))
        for d in drop:
            all_moves.append((1, d))
        return all_moves


board = Board()
print(board.legal_drop_moves())
