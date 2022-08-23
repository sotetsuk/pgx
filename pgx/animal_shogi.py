import numpy as np

DEFAULT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class Board:
    def __init__(self, board, turn):
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

