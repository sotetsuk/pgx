# type: ignore
import jax.numpy as jnp
import jax.random
import numpy as np

TO_MAP = -np.ones((64, 73), dtype=np.int32)
PLANE_MAP = -np.ones((64, 64), dtype=np.int32)  # ignores underpromotion
# underpromotiona
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
for from_ in range(64):
    for plane in range(9):
        to = from_ + [+1, +9, -7][plane % 3] if from_ % 8 == 6 else -1
        to = to if 0 <= to < 64 else -1
        TO_MAP[from_, plane] = to
# normal move
zeros, seq, rseq = [0] * 7, list(range(1, 8)), list(range(-7, 0))
#    down, up, left, right, down-left, down-right, up-right, up-left, knight, and knight
dr = rseq[::] + seq[::] + zeros[::] + zeros[::] + rseq[::] + seq[::] + seq[::-1] + rseq[::-1]
dc = zeros[::] + zeros[::] + rseq[::] + seq[::] + rseq[::] + seq[::] + rseq[::] + seq[::]
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


LEGAL_DEST = -np.ones((7, 64, 27), np.int32)
CAN_MOVE = np.zeros((7, 64, 64), dtype=np.bool_)
# usage: LEGAL_DEST[piece, from_x, from_y]
# LEGAL_DEST[0, :, :] are all -1
# Note that the board is not symmetric about the center (different from shogi)
# You can imagine that the viewpoint is always from the white side.
# Except PAWN, the moves are symmetric about the center.
# We define PAWN as a piece that can move left-up, up, right-up.


# PAWN
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if r1 - r0 == 1 and np.abs(c1 - c0) <= 1:
            legal_dst.append(to)
        # init move
        if r0 == 1 and r1 == 3 and np.abs(c1 - c0) == 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    LEGAL_DEST[1, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[1, from_, a] = True
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
    LEGAL_DEST[2, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[2, from_, a] = True
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
    LEGAL_DEST[3, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[3, from_, a] = True
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
    LEGAL_DEST[4, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[4, from_, a] = True
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
    LEGAL_DEST[5, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[5, from_, a] = True
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
    LEGAL_DEST[6, from_, : len(legal_dst)] = legal_dst
    for a in legal_dst:
        CAN_MOVE[6, from_, a] = True

assert (LEGAL_DEST[0, :, :] == -1).all()

LEGAL_DEST_ANY = -np.ones((64, 35), np.int32)
for from_ in range(64):
    legal_dst = []
    for i in range(27):
        to = LEGAL_DEST[5, from_, i]  # QUEEN
        if to >= 0:
            legal_dst.append(to)
    for i in range(27):
        to = LEGAL_DEST[2, from_, i]  # KNIGHT
        if to >= 0:
            legal_dst.append(to)
    LEGAL_DEST_ANY[from_, : len(legal_dst)] = legal_dst


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
LEGAL_DEST = jnp.array(LEGAL_DEST)
LEGAL_DEST_ANY = jnp.array(LEGAL_DEST_ANY)
CAN_MOVE = jnp.array(CAN_MOVE)
BETWEEN = jnp.array(BETWEEN)
INIT_LEGAL_ACTION_MASK = jnp.array(INIT_LEGAL_ACTION_MASK)

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
