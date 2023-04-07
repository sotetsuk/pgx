import jax.numpy as jnp


CAN_MOVE = -jnp.ones((6, 64, 27), jnp.int8)
# usage: CAN_MOVE[piece, from_x, from_y]
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
    CAN_MOVE = CAN_MOVE.at[0, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
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
    CAN_MOVE = CAN_MOVE.at[1, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
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
    CAN_MOVE = CAN_MOVE.at[2, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
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
    CAN_MOVE = CAN_MOVE.at[3, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
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
    CAN_MOVE = CAN_MOVE.at[4, from_, :len(legal_dst)].set(jnp.int8(legal_dst))
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
    assert len(legal_dst) <= 27
    CAN_MOVE = CAN_MOVE.at[5, from_, :len(legal_dst)].set(jnp.int8(legal_dst))

print(CAN_MOVE[5, 36])
