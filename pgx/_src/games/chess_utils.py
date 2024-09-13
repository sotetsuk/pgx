# type: ignore
import jax.numpy as jnp
import jax.random
import numpy as np

TO_MAP = -np.ones((64, 73), dtype=np.int32)
PLANE_MAP = -np.ones((64, 64), dtype=np.int32)  # ignores underpromotion
zeros, seq, rseq = [0] * 7, list(range(1, 8)), list(range(-7, 0))
#    down, up, left, right, down-left, down-right, up-right, up-left, knight, and knight
dr = rseq[::] + seq[::] + zeros[::] + zeros[::] + rseq[::] + seq[::] + seq[::-1] + rseq[::-1]
dc = zeros[::] + zeros[::] + rseq[::] + seq[::] + rseq[::] + seq[::] + rseq[::] + seq[::]
dr += [-1, +1, -2, +2, -1, +1, -2, +2]
dc += [-2, -2, -1, -1, +2, +2, +1, +1]
for from_ in range(64):
    for plane in range(73):
        if plane < 9:  # underpromotion
            # 8  7 15 23 31 39 47 55 63
            # 7  6 14 22 30 38 46 54 62
            to = from_ + [+1, +9, -7][plane % 3] if from_ % 8 == 6 else -1
            if 0 <= to < 64:
                TO_MAP[from_, plane] = to
        else:  # normal moves
            r = from_ % 8 + dr[plane - 9]
            c = from_ // 8 + dc[plane - 9]
            if 0 <= r < 8 and 0 <= c < 8:
                to = c * 8 + r
                TO_MAP[from_, plane] = to
                PLANE_MAP[from_, to] = plane

LEGAL_DEST = -np.ones((7, 64, 27), np.int32)  # LEGAL_DEST[0, :, :] == -1
CAN_MOVE = np.zeros((7, 64, 64), dtype=np.bool_)
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))
for from_ in range(64):
    legal_dst = {p: [] for p in range(7)}
    for to in range(64):
        if from_ == to:
            continue
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if (r1 - r0 == 1 and abs(c1 - c0) <= 1) or ((r0, r1) == (1, 3) and abs(c1 - c0) == 0):
            legal_dst[PAWN].append(to)
        if (abs(r1 - r0) == 1 and abs(c1 - c0) == 2) or (abs(r1 - r0) == 2 and abs(c1 - c0) == 1):
            legal_dst[KNIGHT].append(to)
        if abs(r1 - r0) == abs(c1 - c0):
            legal_dst[BISHOP].append(to)
        if abs(r1 - r0) == 0 or abs(c1 - c0) == 0:
            legal_dst[ROOK].append(to)
        if (abs(r1 - r0) == 0 or abs(c1 - c0) == 0) or (abs(r1 - r0) == abs(c1 - c0)):
            legal_dst[QUEEN].append(to)
        if from_ != to and abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1:
            legal_dst[KING].append(to)
    for p in range(1, 7):
        LEGAL_DEST[p, from_, :len(legal_dst[p])] = legal_dst[p]
        CAN_MOVE[p, from_, legal_dst[p]] = True

assert (LEGAL_DEST[EMPTY, :, :] == -1).all()

LEGAL_DEST_ANY = -np.ones((64, 35), np.int32)
for from_ in range(64):
    legal_dst = [x for x in list(LEGAL_DEST[5, from_]) + list(LEGAL_DEST[2, from_]) if x >= 0]
    LEGAL_DEST_ANY[from_, :len(legal_dst)] = legal_dst


# Between
BETWEEN = -np.ones((64, 64, 6), dtype=np.int32)
for from_ in range(64):
    for to in range(64):
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if not (abs(r1 - r0) == 0 or abs(c1 - c0) == 0 or abs(r1 - r0) == abs(c1 - c0)):
            continue
        dr, dc = max(min(r1 - r0, 1), -1), max(min(c1 - c0, 1), -1)
        for i in range(6):
            r, c = r0 + dr * (i + 1), c0 + dc * (i + 1)
            if r == r1 and c == c1:
                break
            BETWEEN[from_, to, i] = c * 8 + r

INIT_LEGAL_ACTION_MASK = np.zeros(64 * 73, dtype=np.bool_)
# fmt: off
ixs = [89, 90, 652, 656, 673, 674, 1257, 1258, 1841, 1842, 2425, 2426, 3009, 3010, 3572, 3576, 3593, 3594, 4177, 4178]
# fmt: on
INIT_LEGAL_ACTION_MASK[ixs] = True
assert INIT_LEGAL_ACTION_MASK.shape == (64 * 73,)
assert INIT_LEGAL_ACTION_MASK.sum() == 20

TO_MAP, PLANE_MAP, LEGAL_DEST, LEGAL_DEST_ANY, CAN_MOVE, BETWEEN, INIT_LEGAL_ACTION_MASK = (jnp.array(x) for x in (TO_MAP, PLANE_MAP ,LEGAL_DEST,LEGAL_DEST_ANY,CAN_MOVE,BETWEEN ,INIT_LEGAL_ACTION_MASK))

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
ZOBRIST_EN_PASSANT = jax.random.randint(subkey,shape=(65, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
