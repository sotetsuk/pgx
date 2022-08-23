import numpy as np

DEFAULT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class Board:
    def __init__(self, board, turn):
        # boardは18×11の構造
        # 1~12行目は各座標に存在する駒種、13~18行目はお互いの持ち駒（3種×2）を表現
        self.board = board
        self.turn = turn

    # teban 先手が0　後手が1
    # fir_lo 動かす前の位置
    # fin_lo 動かしたあとの位置
    # piece, captured 動かす駒と取られる駒 ひよこが1 きりん2 ぞう3 ライオン4　にわとり5　capturedは駒取りでない場合0
    # is_promote 駒を成るかどうか
    def move(self, fir_lo, fin_lo, piece, captured, is_promote):
        self.board[fir_lo] = DEFAULT
        self.board[fin_lo] = np.roll(DEFAULT, piece+5*+4*is_promote)
        if captured == 0:
            return
        if captured == 5:
            self.board[12+self.turn*3] = np.roll(self.board[12+self.turn*3], 1)
            return
        self.board[11+captured+self.turn*3] = np.roll(self.board[11+captured+self.turn*3], 1)
        return
