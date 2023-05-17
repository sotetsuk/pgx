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
# fmt off
dr = [-4, -3, -2, -1,  1,  2,  3,  4,  0,  0,  0,  0,  0,  0,  0,  0, -4, -3, -2, -1,  1,  2,  3,  4,  4,  3,  2,  1, -1, -2, -3, -4, -1, +1, -2, +2, -1, +1, -2, +2]
dc = [ 0,  0,  0,  0,  0,  0,  0,  0, -4, -3, -2, -1, +1, +2, +3, +4, -4, -3, -2, -1, +1, +2, +3, +4, -4, -3, -2, -1, +1, +2, +3, +4, -2, -2, -1, -1, +2, +2, +1, +1]
# fmt on
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
