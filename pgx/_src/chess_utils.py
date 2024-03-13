# type: ignore
import jax.numpy as jnp
import jax.random
import numpy as np

TO_MAP = -np.ones((64, 73), dtype=np.int32)
PLANE_MAP = -np.ones((64, 64), dtype=np.int32)  # ignores underpromotion
# underpromotiona
for from_ in range(64):
    if (from_ % 8) not in (1, 6):
        continue
    for plane in range(9):
        dir_ = plane % 3
        to = -1
        if from_ % 8 == 6:
            # white
            # 8  7 15 23 31 39 47 55 63
            # 7  6 14 22 30 38 46 54 62
            # black
            # 2  6 14 22 30 38 46 54 62
            # 1  7 15 23 31 39 47 55 63
            to = from_ + [+1, +9, -7][dir_]
        if not (0 <= to < 64):
            continue
        TO_MAP[from_, plane] = to
# normal move
seq = list(range(1, 8))
zeros = [0 for _ in range(7)]
# 下
dr = [-x for x in seq[::-1]]
dc = [0 for _ in range(7)]
# 上
dr += [x for x in seq]
dc += [0 for _ in range(7)]
# 左
dr += [0 for _ in range(7)]
dc += [-x for x in seq[::-1]]
# 右
dr += [0 for _ in range(7)]
dc += [x for x in seq]
# 左下
dr += [-x for x in seq[::-1]]
dc += [-x for x in seq[::-1]]
# 右上
dr += [x for x in seq]
dc += [x for x in seq]
# 左上
dr += [x for x in seq[::-1]]
dc += [-x for x in seq[::-1]]
# 右下
dr += [-x for x in seq]
dc += [x for x in seq]
# knight moves
dr += [-1, +1, -2, +2, -1, +1, -2, +2]
dc += [-2, -2, -1, -1, +2, +2, +1, +1]
for from_ in range(64):
    for plane in range(9, 73):
        r, c = from_ % 8, from_ // 8
        r = r + dr[plane - 9]
        c = c + dc[plane - 9]
        if r < 0 or r >= 8 or c < 0 or c >= 8:
            continue
        to = c * 8 + r
        TO_MAP[from_, plane] = to
        PLANE_MAP[from_, to] = plane


CAN_MOVE = -np.ones((7, 64, 27), np.int32)
# usage: CAN_MOVE[piece, from_x, from_y]
# CAN_MOVE[0, :, :] are all -1
# Note that the board is not symmetric about the center (different from shogi)
# You can imagine that the viewpoint is always from the white side.
# Except PAWN, the moves are symmetric about the center.
# We define PAWN as a piece that can move up, down, and diagonally, and filter it according to the turn.


# PAWN
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if np.abs(r1 - r0) == 1 and np.abs(c1 - c0) <= 1:
            legal_dst.append(to)
        # init move
        if (r0 == 1 or r0 == 6) and (np.abs(c1 - c0) == 0 and np.abs(r1 - r0) == 2):
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE[1, from_, : len(legal_dst)] = legal_dst
# KNIGHT
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if np.abs(r1 - r0) == 1 and np.abs(c1 - c0) == 2:
            legal_dst.append(to)
        if np.abs(r1 - r0) == 2 and np.abs(c1 - c0) == 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE[2, from_, : len(legal_dst)] = legal_dst
# BISHOP
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if np.abs(r1 - r0) == np.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE[3, from_, : len(legal_dst)] = legal_dst
# ROOK
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if np.abs(r1 - r0) == 0 or np.abs(c1 - c0) == 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE[4, from_, : len(legal_dst)] = legal_dst
# QUEEN
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if np.abs(r1 - r0) == 0 or np.abs(c1 - c0) == 0:
            legal_dst.append(to)
        if np.abs(r1 - r0) == np.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE[5, from_, : len(legal_dst)] = legal_dst
# KING
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if (np.abs(r1 - r0) <= 1) and (np.abs(c1 - c0) <= 1):
            legal_dst.append(to)
    # castling
    # if from_ == 32:
    #     legal_dst += [16, 48]
    # if from_ == 39:
    #     legal_dst += [23, 55]
    assert len(legal_dst) <= 8
    CAN_MOVE[6, from_, : len(legal_dst)] = legal_dst

assert (CAN_MOVE[0, :, :] == -1).all()

CAN_MOVE_ANY = -np.ones((64, 35), np.int32)
for from_ in range(64):
    legal_dst = []
    for i in range(27):
        to = CAN_MOVE[5, from_, i]  # QUEEN
        if to >= 0:
            legal_dst.append(to)
    for i in range(27):
        to = CAN_MOVE[2, from_, i]  # KNIGHT
        if to >= 0:
            legal_dst.append(to)
    CAN_MOVE_ANY[from_, : len(legal_dst)] = legal_dst


# Between
BETWEEN = -np.ones((64, 64, 6), dtype=np.int32)
for from_ in range(64):
    for to in range(64):
        r0, c0 = from_ % 8, from_ // 8
        r1, c1 = to % 8, to // 8
        if not ((np.abs(r1 - r0) == 0 or np.abs(c1 - c0) == 0) or (np.abs(r1 - r0) == np.abs(c1 - c0))):
            continue
        dr = max(min(r1 - r0, 1), -1)
        dc = max(min(c1 - c0, 1), -1)
        r = r0
        c = c0
        bet = []
        while True:
            r += dr
            c += dc
            if r == r1 and c == c1:
                break
            bet.append(c * 8 + r)
        assert len(bet) <= 6
        BETWEEN[from_, to, : len(bet)] = bet

INIT_LEGAL_ACTION_MASK = np.zeros(64 * 73, dtype=np.bool_)
# fmt: off
ixs = [89, 90, 652, 656, 673, 674, 1257, 1258, 1841, 1842, 2425, 2426, 3009, 3010, 3572, 3576, 3593, 3594, 4177, 4178]
# fmt: on
INIT_LEGAL_ACTION_MASK[ixs] = True
assert INIT_LEGAL_ACTION_MASK.shape == (64 * 73,)
assert INIT_LEGAL_ACTION_MASK.sum() == 20

TO_MAP = jnp.array(TO_MAP)
PLANE_MAP = jnp.array(PLANE_MAP)
CAN_MOVE = jnp.array(CAN_MOVE)
CAN_MOVE_ANY = jnp.array(CAN_MOVE_ANY)
BETWEEN = jnp.array(BETWEEN)
INIT_LEGAL_ACTION_MASK = jnp.array(INIT_LEGAL_ACTION_MASK)
INIT_POSSIBLE_PIECE_POSITIONS = jnp.int32(
    [
        [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57],
        [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57],
    ]
)  # (2, 16)

key = jax.random.PRNGKey(238290)
key, subkey = jax.random.split(key)
ZOBRIST_BOARD = jax.random.randint(subkey, shape=(64, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
key, subkey = jax.random.split(key)
ZOBRIST_SIDE = jax.random.randint(subkey, shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)

key, subkey = jax.random.split(key)
ZOBRIST_CASTLING_QUEEN = jax.random.randint(subkey, shape=(2, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
key, subkey = jax.random.split(key)
ZOBRIST_CASTLING_KING = jax.random.randint(subkey, shape=(2, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
key, subkey = jax.random.split(key)
ZOBRIST_EN_PASSANT = jax.random.randint(
    subkey,
    shape=(
        65,
        2,
    ),
    minval=0,
    maxval=2**31 - 1,
    dtype=jnp.uint32,
)
