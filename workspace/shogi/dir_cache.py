#! python3

from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes

#  dir, to, from
#
#  (10, 81, 81)
#
#  0 Up
#  1 Up left
#  2 Up right
#  3 Left
#  4 Right
#  5 Down
#  6 Down left
#  7 Down right
#  8 Up2 left
#  9 Up2 right


LEGAL_FROM_IDX = -jnp.ones((10, 81, 8), dtype=jnp.int32)


for dir_ in tqdm(range(10)):
    for to in range(81):
        x, y = to // 9, to % 9
        if dir_ == 0:  # Up
            dx, dy = 0, +1
        elif dir_ == 1:  # Up left
            dx, dy = -1, +1
        elif dir_ == 2:  # Up right
            dx, dy = +1, +1
        elif dir_ == 3:  # Left
            dx, dy = -1, 0
        elif dir_ == 4:  # Right
            dx, dy = +1, 0
        elif dir_ == 5:  # Down
            dx, dy = 0, -1
        elif dir_ == 6:  # Down left
            dx, dy = -1, -1
        elif dir_ == 7:  # Down right
            dx, dy = +1, -1
        elif dir_ == 8:  # Up2 left
            dx, dy = -1, +2
        elif dir_ == 9:  # Up2 right
            dx, dy = +1, +2
        for i in range(8):
            x += dx
            y += dy
            if x < 0 or 8 < x or y < 0 or 8 < y:
                break
            LEGAL_FROM_IDX = LEGAL_FROM_IDX.at[dir_, to, i].set(x * 9 + y)
            if dir_ == 8 or dir_ == 9:
                break

# print(jnp.rot90(jnp.arange(81).reshape(9, 9), k=3))


# check
# to = 40
# for dir_ in tqdm(range(10)):
#     print(dir_)
#     print(LEGAL_FROM_IDX[dir_, to, :])


print(f"BYTES = {to_bytes(LEGAL_FROM_IDX)}")

