import jax
from jax import lax
import jax.numpy as jnp
from pgx.experimental.chess import from_fen

# ピースの定義
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))  # ピース
PIECE_TYPES = [EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]

# ボードのサイズ
BOARD_SIZE = 64

# 4bitの情報に対応するためのシフト量
SHIFT_PIECE_TYPE = 0  # piece typeを表すのは0ビット右にずれる
SHIFT_COLOR = 3       # colorは左端のビット (3ビット分ずれる)

@jax.jit
def to_bitboard(board):
    bitboard = jnp.zeros(8, dtype=jnp.int32)

    for idx in range(BOARD_SIZE):
        piece = board[idx]
        rank = idx % 8
        file = idx // 8
        color = piece < 0  # >=0: us, <0: opp
        piece_type = jnp.abs(piece)
        bit_value = (color << SHIFT_COLOR) | piece_type
        bit_value = bitboard[rank] | (bit_value << (4 * file))
        bitboard = bitboard.at[rank].set(bit_value)

    return bitboard


@jax.jit
def to_board(bitboard):
    return jax.vmap(get_bb, in_axes=(None, 0))(bitboard, jnp.arange(64))


def get_bb(bb, pos):
    rank, file = pos % 8, pos // 8
    rank_bb = bb[rank]
    bits = (rank_bb >> (4 * file)) & 0b1111
    color = (bits >> SHIFT_COLOR) & 1
    piece_type = bits & 0b111
    return jnp.int32([1, -1])[color] * piece_type

# --- テストコード ---
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
INIT_BOARD = jnp.int32([4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4])  # fmt: skip


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
assert jnp.array_equal(board, reconstructed_board)
