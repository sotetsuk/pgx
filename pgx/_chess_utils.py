import jax.numpy as jnp


CAN_MOVE = -jnp.ones((6, 64, 27), jnp.int8)
# usage: CAN_MOVE[piece, from_x, from_y]
# CAN_MOVE[0, :, :]はすべて-1
# 将棋と違い、中央から点対称でないので、注意が必要。
# 視点は常に白側のイメージが良い。
# PAWN以外の動きは上下左右対称。PAWNは上下と斜めへ動ける駒と定義して、手番に応じてフィルタする。


# PAWN
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if jnp.abs(r1 - r0) == 1 and jnp.abs(c1 - c0) <= 1:
            legal_dst.append(to)
        # init move
        if (r0 == 1 or r0 == 6) and (jnp.abs(c1 - c0) == 0 and jnp.abs(r1 - r0) == 2):
            legal_dst.append(to)
    CAN_MOVE = CAN_MOVE.at[1, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
# KNIGHT
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if jnp.abs(r1 - r0) == 1 and jnp.abs(c1 - c0) == 2:
            legal_dst.append(to)
        if jnp.abs(r1 - r0) == 2 and jnp.abs(c1 - c0) == 1:
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[2, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
# BISHOP
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == jnp.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[3, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
# ROOK
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0:
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[4, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
# QUEEN
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0:
            legal_dst.append(to)
        if jnp.abs(r1 - r0) == jnp.abs(c1 - c0):
            legal_dst.append(to)
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[5, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
# KING
for from_ in range(64):
    r0, c0 = from_ % 8, from_ // 8
    legal_dst = []
    for to in range(64):
        r1, c1 = to % 8, to // 8
        if from_ == to:
            continue
        if jnp.abs(r1 - r0) <= 1 and jnp.abs(c1 - c0) <= 1:
            legal_dst.append(to)
    # castling
    # if from_ == 32:
    #     legal_dst += [16, 48]
    # if from_ == 39:
    #     legal_dst += [23, 55]
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[6, from_, :len(legal_dst)].set(jnp.int8(legal_dst))


# Between
BETWEEN = -jnp.ones((64, 64, 6), dtype=jnp.int8)
for from_ in range(64):
    for to in range(64):
        r0, c0 = from_ % 8, from_ // 8
        r1, c1 = to % 8, to // 8
        if not ((jnp.abs(r1 - r0) == 0 or jnp.abs(c1 - c0) == 0) or
                (jnp.abs(r1 - r0) == jnp.abs(c1 - c0))):
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
        BETWEEN = BETWEEN.at[from_, to, :len(bet)].set(jnp.int8(bet))

