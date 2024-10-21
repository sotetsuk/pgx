import numpy as np
# import jax.numpy as jnp

# ピースの定義
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))  # ピース
PIECE_TYPES = [EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]

# ボードのサイズ
BOARD_SIZE = 64

# 4bitの情報に対応するためのシフト量
SHIFT_PIECE_TYPE = 0  # piece typeを表すのは0ビット右にずれる
SHIFT_COLOR = 3       # colorは左端のビット (3ビット分ずれる)


# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
INIT_BOARD = np.int32([4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4])  # fmt: skip


def to_bitboard(board):
    """
    64次元のboardをbitboardに変換する。
    bitboardは8つの32bit整数で表現され、各rankに対応。
    """
    # bitboardは8行なので、8つの32bit整数を持つ
    bitboard = np.zeros(8, dtype=np.uint32)

    for idx in range(BOARD_SIZE):
        piece = board[idx]
        if piece == EMPTY:
            # EMPTYならビット操作をスキップする
            continue

        # idxからrank (row) と file (col) を計算
        rank = idx % 8
        file = idx // 8

        # colorとpiece typeのビット表現を作る
        color = 1 if piece < 0 else 0  # opponentなら1, 自分(us)なら0
        piece_type = abs(piece)

        # rankに対応する32bit整数に4bitのピース情報を埋め込む
        bit_value = (color << SHIFT_COLOR) | (piece_type << SHIFT_PIECE_TYPE)

        # fileに対応する位置にピース情報を配置
        bitboard[rank] |= (bit_value << (4 * file))

    return bitboard

def to_board(bitboard):
    """
    bitboardを64次元のboardに変換する。
    boardは各マスごとにピースの種類と色を保持。
    """
    # 64マスのboardを初期化
    board = np.zeros(BOARD_SIZE, dtype=np.int8)

    for rank in range(8):
        # rankに対応する32bit整数
        rank_bits = bitboard[rank]

        for file in range(8):
            bit_value = (rank_bits >> (4 * file)) & 0b1111

            # 4bitからcolorとpiece typeを抽出
            color = (bit_value >> SHIFT_COLOR) & 1
            piece_type = bit_value & 0b111

            # ピースがemptyなら次へ
            if piece_type == EMPTY:
                continue

            # boardにcolorに応じたピースを配置
            board[file * 8 + rank] = -piece_type if color == 1 else piece_type

    return board

# --- テストコード ---

# テスト用のboard
board = INIT_BOARD

# board -> bitboard -> board のテスト
bitboard = to_bitboard(board)
reconstructed_board = to_board(bitboard)

print("Original board:")
print(board.reshape(8, 8))
print("\nBitboard:")
print(bitboard)
print("\nReconstructed board:")
print(reconstructed_board.reshape(8, 8))

# 再構築したboardが元のboardと同じかどうかを確認
assert np.array_equal(board, reconstructed_board)

