import jax
import jax.numpy as jnp
from flax.serialization import to_bytes

from pgx.shogi import *


def can_move_to(piece, from_, to):
    """Can <piece> move from <from_> to <to>?"""
    if from_ == to:
        return False
    x0, y0 = from_ // 9, from_ % 9
    x1, y1 = to // 9, to % 9
    dx = x1 - x0
    dy = y1 - y0
    if piece == PAWN:
        if dx == 0 and dy == -1:
            return True
        else:
            return False
    elif piece == LANCE:
        if dx == 0 and dy < 0:
            return True
        else:
            return False
    elif piece == KNIGHT:
        if dx in (-1, 1) and dy == -2:
            return True
        else:
            return False
    elif piece == SILVER:
        if dx in (-1, 0, 1) and dy == -1:
            return True
        elif dx in (-1, 1) and dy == 1:
            return True
        else:
            return False
    elif piece == BISHOP:
        if dx == dy or dx == -dy:
            return True
        else:
            return False
    elif piece == ROOK:
        if dx == 0 or dy == 0:
            return True
        else:
            return False
    if piece in (GOLD, PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER):
        if dx in (-1, 0, 1) and dy in (0, -1):
            return True
        elif dx == 0 and dy == 1:
            return True
        else:
            return False
    elif piece == KING:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        else:
            return False
    elif piece == HORSE:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        elif dx == dy or dx == -dy:
            return True
        else:
            return False
    elif piece == DRAGON:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        if dx == 0 or dy == 0:
            return True
        else:
            return False
    else:
        assert False

def show(piece, point):
    x = ""
    for i in range(81):
        if i % 9 == 0:
            print(x)
            x = ""
        if i == point:
            x += "x "
        else:
            x += str(int(can_move_to(piece, point, i))) + " "
    print(x)


# for piece in (PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING, HORSE, DRAGON):
#     print("===")
#     show(piece, 40)
#
#
# for piece in (PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING, HORSE, DRAGON):
#     print("===")
#     show(piece, 0)


RAW_EFFECT_BOARDS = jnp.zeros((14, 81, 81), dtype=jnp.bool_)
for piece in range(14):
    for from_ in range(81):
        for to in range(81):
            RAW_EFFECT_BOARDS = RAW_EFFECT_BOARDS.at[piece, from_, to].set(can_move_to(piece, from_, to))

print(f"RAW_EFFECT_BOARDS = {to_bytes(RAW_EFFECT_BOARDS)}")
