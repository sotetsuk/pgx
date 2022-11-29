import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import struct

NUM_TILES = 44
NUM_TILE_TYPES = 11
N_PLAYER = 3
MAX_RIVER_LENGTH = 10
WIN_CACHE = jnp.int8(
    [
        [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
        [4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 4, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 3],
        [4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3],
        [3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 3, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 3, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3],
        [3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 3, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 3, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 3, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 3, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 3, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 3, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 3, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 3, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 3, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 3, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 3, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 3, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 3, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 3, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 3, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 3],
        [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
    ]
)


@struct.dataclass
class State:
    curr_player: jnp.ndarray = jnp.int8(0)
    legal_action_mask: jnp.ndarray = jnp.zeros(9, jnp.bool_)
    terminated: jnp.ndarray = jnp.bool_(False)
    turn: jnp.ndarray = jnp.int8(0)  # 0 = dealer
    rivers: jnp.ndarray = -jnp.ones(
        (N_PLAYER, MAX_RIVER_LENGTH), dtype=jnp.int8
    )  # type type (0~10) is set
    last_discard: jnp.ndarray = jnp.int8(-1)  # type type (0~10) is set
    hands: jnp.ndarray = jnp.zeros(
        (N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8
    )  # type type (0~10) is set
    walls: jnp.ndarray = jnp.zeros(
        NUM_TILES, dtype=jnp.int8
    )  # tile id (0~43) is set
    draw_ix: jnp.ndarray = jnp.int8(N_PLAYER * 5)
    shuffled_players: jnp.ndarray = jnp.zeros(N_PLAYER)  # 0: dealer, ...
    dora: jnp.ndarray = jnp.int8(0)  # type type (0~10) is set


# TODO: avoid Tenhou
@jax.jit
def init(rng: jax.random.KeyArray):
    # shuffle players and walls
    key1, key2 = jax.random.split(rng)
    shuffled_players = jnp.arange(N_PLAYER)
    shuffled_players = jax.random.shuffle(key1, shuffled_players)
    walls = jnp.arange(NUM_TILES, dtype=jnp.int8)
    walls = jax.random.shuffle(key2, walls)
    curr_player = shuffled_players[0]  # dealer
    dora = walls[-1] // 4
    # set hands (hands[0] = dealer's hand)
    hands = jnp.zeros((N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int8)
    hands = lax.fori_loop(
        0, N_PLAYER * 5, lambda i, x: x.at[i // 5, walls[i] // 4].add(1), hands
    )
    # first draw
    draw_ix = jnp.int8(N_PLAYER * 5)
    hands = hands.at[0, walls[draw_ix] // 4].add(1)
    draw_ix += 1
    legal_action_mask = hands[0] > 0
    state = State(
        curr_player=curr_player,
        legal_action_mask=legal_action_mask,
        hands=hands,
        walls=walls,
        draw_ix=draw_ix,
        shuffled_players=shuffled_players,
        dora=dora,
    )  # type: ignore
    return curr_player, state


def _is_completed(hand: jnp.ndarray):
    x = jnp.abs(hand - WIN_CACHE).sum(axis=-1).min()
    return x == 0


@jax.jit
def _check_ron(state: State) -> jnp.ndarray:
    # TODO: furiten
    # TODO: 5-fan limit
    winning_players = jax.lax.fori_loop(
        0, N_PLAYER,
        lambda i, x: x.at[i].set(_is_completed(state.hands.at[i, state.last_discard].add(1)[i])),
        jnp.zeros(N_PLAYER, dtype=jnp.bool_)
    )
    winning_players = winning_players.at[state.turn].set(False)
    return winning_players


@jax.jit
def _check_tsumo(state: State) -> jnp.ndarray:
    return _is_completed(state.hands[state.turn])


@jax.jit
def _step_by_ron(state: State):
    winning_players = _check_ron(state)
    r_by_turn = winning_players.astype(jnp.float16)
    r_by_turn = r_by_turn.at[state.turn % N_PLAYER].set(- r_by_turn.sum())
    r = lax.fori_loop(
        0, N_PLAYER,
        lambda i, x: x.at[state.shuffled_players[i]].set(r_by_turn[i]),
        jnp.zeros_like(r_by_turn)
    )
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
    )
    return curr_player, state, r


@jax.jit
def _step_by_tsumo(state):
    r = - jnp.ones(N_PLAYER, dtype=jnp.float16) * (1 / (N_PLAYER - 1))
    r = r.at[state.shuffled_players[state.turn]].set(1)
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
    )
    return curr_player, state, r


def _step_by_tie(state):
    curr_player = jnp.int8(-1)
    state = state.replace(  # type: ignore
        curr_player=curr_player,
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
    )
    r = jnp.zeros(3, dtype=jnp.float16)
    return curr_player, state, r


def _draw_tile(state: State) -> State:
    turn = state.turn + 1
    curr_player = state.shuffled_players[turn % N_PLAYER]
    hands = state.hands.at[turn % N_PLAYER, state.walls[state.draw_ix] // 4].add(1)
    draw_ix = state.draw_ix + 1
    legal_action_mask = hands[turn % N_PLAYER] > 0
    state = state.replace(  # type: ignore
        turn=turn,
        curr_player=curr_player,
        hands=hands,
        draw_ix=draw_ix,
        legal_action_mask=legal_action_mask,
        terminated=jnp.bool_(False),
    )
    return state


def _step_non_terminal(state: State):
    r = jnp.zeros(3, dtype=jnp.float16)
    return state.curr_player, state, r


def _step_non_tied(state: State):
    state = _draw_tile(state)
    is_tsumo = _check_tsumo(state)
    if is_tsumo:
        return _step_by_tsumo(state)
    else:
        return _step_non_terminal(state)


def step(state: State, action: jnp.ndarray):
    # discard tile
    hands = state.hands.at[state.turn % N_PLAYER, action].add(-1)
    rivers = state.rivers.at[
        state.turn % N_PLAYER, state.turn // N_PLAYER
    ].set(action)
    last_discard = action
    state = state.replace(  # type: ignore
        hands=hands,
        rivers=rivers,
        last_discard=last_discard,
    )

    win_players = _check_ron(state)
    if jnp.any(win_players):
        return _step_by_ron(state)
    else:
        if jnp.bool_(NUM_TILES - 1 <= state.draw_ix):
            return _step_by_tie(state)
        else:
            return _step_non_tied(state)


def _tile_type_to_str(tile_type) -> str:
    if tile_type < 9:
        s = str(tile_type + 1)
    elif tile_type == 9:
        s = "g"
    elif tile_type == 10:
        s = "r"
    return s


def _hand_to_str(hand: jnp.ndarray) -> str:
    s = ""
    for i in range(NUM_TILE_TYPES):
        for j in range(hand[i]):
            s += _tile_type_to_str(i)
    return s.ljust(6)


def _river_to_str(river: jnp.ndarray) -> str:
    s = ""
    for i in range(MAX_RIVER_LENGTH):
        tile_type = river[i]
        s += _tile_type_to_str(tile_type) if tile_type >= 0 else "x"
    return s


def _to_str(state: State):
    s = f"{'[terminated]' if state. terminated else ''} dora: {_tile_type_to_str(state.dora)}\n"
    for i in range(N_PLAYER):
        s += f"{'*' if not state.terminated and state.turn % N_PLAYER == i else ' '}[{state.shuffled_players[i]}] "
        s += f"{_hand_to_str(state.hands[i])}, "
        s += f"{_river_to_str(state.rivers[i])} "
        s += "\n"
    return s


def _is_valid(state: State) -> bool:
    if state.dora < 0 or 10 < state.dora:
        return False
    if 10 < state.last_discard:
        return False
    if state.last_discard < 0 and state.rivers[0, 0] >= 0:
        return False
    if jnp.any(state.hands < 0):
        return False
    counts = state.hands.sum(axis=0)
    for i in range(N_PLAYER):
        for j in range(MAX_RIVER_LENGTH):
            if state.rivers[i, j] >= 0:
                counts = counts.at[state.rivers[i, j]].add(1)
    if jnp.any(counts > 4):
        return False
    for i in range(NUM_TILE_TYPES):
        if state.legal_action_mask[i] and state.hands[state.turn, i] <= 0:
            return False
        if not state.legal_action_mask[i] and state.hands[state.turn, i] > 0:
            return False

    return True
