# type: ignore
import jax
import jax.numpy as jnp
import numpy as np

TO_MAP = -np.ones((25, 49), dtype=np.int32)
PLANE_MAP = -np.ones((25, 25), dtype=np.int32)  # ignores underpromotions
# underpromotions
for from_ in range(25):
    if from_ % 5 != 3:  # 4th row in current player view
        continue
    for plane in range(9):
        dir_ = plane % 3
        # board index (white view)
        # 5  4  9 14 19 24
        # 4  3  8 13 18 23
        # board index (flipped black view)
        # 5  0  5 10 15 20
        # 4  1  6 11 16 21
        to = from_ + [+1, +6, -4][dir_]
        if not (0 <= to < 25):
            continue
        TO_MAP[from_, plane] = to
# normal move
# fmt: off
dr = [-4, -3, -2, -1,  1,  2,  3,  4,  0,  0,  0,  0,  0,  0,  0,  0, -4, -3, -2, -1,  1,  2,  3,  4,  4,  3,  2,  1, -1, -2, -3, -4, -1, +1, -2, +2, -1, +1, -2, +2]  # noqa
dc = [ 0,  0,  0,  0,  0,  0,  0,  0, -4, -3, -2, -1, +1, +2, +3, +4, -4, -3, -2, -1, +1, +2, +3, +4, -4, -3, -2, -1, +1, +2, +3, +4, -2, -2, -1, -1, +2, +2, +1, +1]  # noqa
# fmt: on
for from_ in range(25):
    for plane in range(9, 49):
        r, c = from_ % 5, from_ // 5
        r = r + dr[plane - 9]
        c = c + dc[plane - 9]
        if r < 0 or r >= 5 or c < 0 or c >= 5:
            continue
        to = c * 5 + r
        TO_MAP[from_, plane] = to
        PLANE_MAP[from_, to] = plane


CAN_MOVE = -np.ones((7, 25, 16), np.int32)
# usage: CAN_MOVE[piece, from_x, from_y]
# CAN_MOVE[0, :, :] are all -1
# Note that the board is not symmetric about the center (different from shogi)
# You can imagine that the viewpoint is always from the white side.
# Except PAWN, the moves are symmetric about the center.
# We define PAWN as a piece that can move up and diagonally.


# PAWN
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if r1 - r0 == 1 and np.abs(c1 - c0) <= 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 6, f"{from_=}, {to=}, {legal_dst=}"
    CAN_MOVE[1, from_, : len(legal_dst)] = legal_dst
# KNIGHT
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if np.abs(r1 - r0) == 1 and np.abs(c1 - c0) == 2:
            legal_dst.append(to)
        if np.abs(r1 - r0) == 2 and np.abs(c1 - c0) == 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE[2, from_, : len(legal_dst)] = legal_dst
# BISHOP
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if np.abs(r1 - r0) == np.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE[3, from_, : len(legal_dst)] = legal_dst
# ROOK
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if np.abs(r1 - r0) == 0 or np.abs(c1 - c0) == 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE[4, from_, : len(legal_dst)] = legal_dst
# QUEEN
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if np.abs(r1 - r0) == 0 or np.abs(c1 - c0) == 0:
            legal_dst.append(to)
        if np.abs(r1 - r0) == np.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 16
    CAN_MOVE[5, from_, : len(legal_dst)] = legal_dst
# KING
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if (np.abs(r1 - r0) <= 1) and (np.abs(c1 - c0) <= 1):
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE[6, from_, : len(legal_dst)] = legal_dst

assert (CAN_MOVE[0, :, :] == -1).all()

CAN_MOVE_ANY = -np.ones((25, 24), np.int32)
for from_ in range(25):
    legal_dst = []
    for i in range(16):
        to = CAN_MOVE[5, from_, i]  # QUEEN
        if to >= 0:
            legal_dst.append(to)
    for i in range(16):
        to = CAN_MOVE[2, from_, i]  # KNIGHT
        if to >= 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 24
    CAN_MOVE_ANY[from_, : len(legal_dst)] = legal_dst

BETWEEN = -np.ones((25, 25, 3), dtype=np.int32)
for from_ in range(25):
    for to in range(25):
        r0, c0 = from_ % 5, from_ // 5
        r1, c1 = to % 5, to // 5
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
            bet.append(c * 5 + r)
        assert len(bet) <= 3
        BETWEEN[from_, to, : len(bet)] = bet


INIT_LEGAL_ACTION_MASK = np.zeros(25 * 49, dtype=np.bool_)
# fmt: off
ixs = [62, 289, 293, 307, 552, 797, 1042]
# fmt: on
for ix in ixs:
    INIT_LEGAL_ACTION_MASK[ix] = True
assert INIT_LEGAL_ACTION_MASK.shape == (25 * 49,)
assert INIT_LEGAL_ACTION_MASK.sum() == 7

TO_MAP = jnp.array(TO_MAP)
PLANE_MAP = jnp.array(PLANE_MAP)
CAN_MOVE = jnp.array(CAN_MOVE)
CAN_MOVE_ANY = jnp.array(CAN_MOVE_ANY)
BETWEEN = jnp.array(BETWEEN)
INIT_LEGAL_ACTION_MASK = jnp.array(INIT_LEGAL_ACTION_MASK)


key = jax.random.PRNGKey(238942)
key, subkey = jax.random.split(key)
ZOBRIST_BOARD = jax.random.randint(subkey, shape=(25, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
key, subkey = jax.random.split(key)
ZOBRIST_SIDE = jax.random.randint(subkey, shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
