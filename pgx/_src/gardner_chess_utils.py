import jax.numpy as jnp

TO_MAP = -jnp.ones((25, 49), dtype=jnp.int8)
PLANE_MAP = -jnp.ones((5, 5), dtype=jnp.int8)  # ignores underpromotions
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
        if not (0 <= to < 64):
            continue
        TO_MAP = TO_MAP.at[from_, plane].set(to)
# normal move
seq = list(range(1, 5))
zeros = [0 for _ in range(4)]
# 下
dr = [-x for x in seq[::-1]]
dc = zeros
# 上
dr += [x for x in seq]
dc += [0 for _ in zeros]
# 左
dr += [0 for _ in zeros]
dc += [-x for x in seq[::-1]]
# 右
dr += [0 for _ in zeros]
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
