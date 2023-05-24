# type: ignore
import jax
import jax.numpy as jnp

TO_MAP = -jnp.ones((25, 49), dtype=jnp.int8)
PLANE_MAP = -jnp.ones((25, 25), dtype=jnp.int8)  # ignores underpromotions
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
        to = from_ + jnp.int8([+1, +6, -4])[dir_]
        if not (0 <= to < 25):
            continue
        TO_MAP = TO_MAP.at[from_, plane].set(to)
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
        to = jnp.int8(c * 5 + r)
        TO_MAP = TO_MAP.at[from_, plane].set(to)
        PLANE_MAP = PLANE_MAP.at[from_, to].set(jnp.int8(plane))


CAN_MOVE = -jnp.ones((7, 25, 16), jnp.int8)
# usage: CAN_MOVE[piece, from_x, from_y]
# CAN_MOVE[0, :, :]はすべて-1
# 将棋と違い、中央から点対称でないので、注意が必要。
# 視点は常に白側のイメージが良い。
# PAWN以外の動きは上下左右対称。


# PAWN
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if r1 - r0 == 1 and jnp.abs(c1 - c0) <= 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 6, f"{from_=}, {to=}, {legal_dst=}"
    CAN_MOVE = CAN_MOVE.at[1, from_, : len(legal_dst)].set(jnp.int8(legal_dst))
# KNIGHT
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if jnp.abs(r1 - r0) == 1 and jnp.abs(c1 - c0) == 2:
            legal_dst.append(to)
        if jnp.abs(r1 - r0) == 2 and jnp.abs(c1 - c0) == 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE = CAN_MOVE.at[2, from_, : len(legal_dst)].set(jnp.int8(legal_dst))
# BISHOP
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == jnp.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE = CAN_MOVE.at[3, from_, : len(legal_dst)].set(jnp.int8(legal_dst))
# ROOK
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE = CAN_MOVE.at[4, from_, : len(legal_dst)].set(jnp.int8(legal_dst))
# QUEEN
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0:
            legal_dst.append(to)
        if jnp.abs(r1 - r0) == jnp.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 16
    CAN_MOVE = CAN_MOVE.at[5, from_, : len(legal_dst)].set(jnp.int8(legal_dst))
# KING
for from_ in range(25):
    r0, c0 = from_ % 5, from_ // 5
    legal_dst = []
    for to in range(25):
        r1, c1 = to % 5, to // 5
        if from_ == to:
            continue
        if (jnp.abs(r1 - r0) <= 1) and (jnp.abs(c1 - c0) <= 1):
            legal_dst.append(to)
    assert len(legal_dst) <= 8
    CAN_MOVE = CAN_MOVE.at[6, from_, : len(legal_dst)].set(jnp.int8(legal_dst))

assert (CAN_MOVE[0, :, :] == -1).all()

CAN_MOVE_ANY = -jnp.ones((25, 24), jnp.int8)
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
    CAN_MOVE_ANY = CAN_MOVE_ANY.at[from_, : len(legal_dst)].set(
        jnp.int8(legal_dst)
    )

BETWEEN = -jnp.ones((25, 25, 3), dtype=jnp.int8)
for from_ in range(25):
    for to in range(25):
        r0, c0 = from_ % 5, from_ // 5
        r1, c1 = to % 5, to // 5
        if not (
            (jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0)
            or (jnp.abs(r1 - r0) == jnp.abs(c1 - c0))
        ):
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
        BETWEEN = BETWEEN.at[from_, to, : len(bet)].set(jnp.int8(bet))


INIT_LEGAL_ACTION_MASK = jnp.zeros(25 * 49, dtype=jnp.bool_)
# fmt: off
ixs = [62, 289, 293, 307, 552, 797, 1042]
# fmt: on
for ix in ixs:
    INIT_LEGAL_ACTION_MASK = INIT_LEGAL_ACTION_MASK.at[ix].set(True)
assert INIT_LEGAL_ACTION_MASK.shape == (25 * 49,)
assert INIT_LEGAL_ACTION_MASK.sum() == 7


key = jax.random.PRNGKey(238942)
key, subkey = jax.random.split(key)
ZOBRIST_BOARD = jax.random.randint(
    subkey, shape=(25, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32
)
key, subkey = jax.random.split(key)
ZOBRIST_SIDE = jax.random.randint(
    subkey, shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32
)
