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

LEGAL_FROM_MASK = jnp.zeros((10, 81, 81), dtype=jnp.bool_)

for dir_ in tqdm(range(10)):
    for from_ in tqdm(range(81), leave=False):
        for to in range(81):
            ok = False
            x0, y0 = from_ // 9, from_ % 9
            x1, y1 = to // 9, to % 9
            dx = x1 - x0
            dy = y1 - y0
            if dir_ == 0:  # Up
                ok = (dx == 0) and (dy < 0)
            elif dir_ == 1:  # Up left
                ok = (dx > 0) and (dy < 0) and (abs(dx) == abs(dy))
            elif dir_ == 2:  # Up right
                ok = (dx < 0) and (dy < 0) and (dx == dy)
            elif dir_ == 3:  # Left
                ok = (dx > 0) and (dy == 0)
            elif dir_ == 4:  # Right
                ok = (dx < 0) and (dy == 0)
            elif dir_ == 5:  # Down
                ok = (dx == 0) and (dy > 0)
            elif dir_ == 6:  # Down left
                ok = (dx > 0) and (dy > 0) and (dx == dy)
            elif dir_ == 7:  # Down right
                ok = (dx < 0) and (dy > 0) and (abs(dx) == abs(dy))
            elif dir_ == 8:  # Up2 left
                ok = (dx == 1) and (dy == -2)
            elif dir_ == 9:  # Up2 right
                ok = (dx == -1) and (dy == -2)
            LEGAL_FROM_MASK = LEGAL_FROM_MASK.at[dir_, to, from_].set(ok)


# check
# to = 40
# for dir_ in tqdm(range(10)):
#     print(dir_)
#     print(jnp.rot90(LEGAL_FROM_MASK[dir_, to].reshape(9, 9), k=3))


print(f"LEGAL_FROM_MASK_BYTES = {to_bytes(LEGAL_FROM_MASK)}")

