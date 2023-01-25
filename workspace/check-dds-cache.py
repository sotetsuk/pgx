import sys
import jax
import jax.numpy as jnp
import time


@jax.vmap
def to_key(x):
    """Convert board information to key
    >>> x = jnp.arange(52, dtype=jnp.int32) % 4
    >>> x[0:13]
    [0 1 2 3 0 1 2 3 0 1 2 3 0]
    >>> x[13:26]
    [1 2 3 0 1 2 3 0 1 2 3 0 1]
    >>> x[26:39]
    [2 3 0 1 2 3 0 1 2 3 0 1 2]
    >>> x[39:52]
    [3 0 1 2 3 0 1 2 3 0 1 2 3]
    >>> to_key(x)
    [15000804 20527417 38686286 60003219]
    """
    _to_binary = jax.vmap(to_binary)
    y = x.reshape(4, 13)
    return _to_binary(y)  # shape = (4,)


def to_binary(x) -> jnp.array:
    """Convert hand information to key
    >>> x = jnp.arange(13, dtype=jnp.int32) % 4
    >>> x
    [0 1 2 3 0 1 2 3 0 1 2 3 0]
    >>> to_binary(x)
    15000804
    >>> x = jnp.ones(13, dtype=jnp.int32) * 3
    [3 3 3 3 3 3 3 3 3 3 3 3 3]
    >>> x
    67108863
    """
    bases = jnp.int32([4 ** i for i in range(13)])
    return (x * bases).sum()  # shape = (1,)


# Only for testing
@jax.vmap
def to_board(rng):
    return jax.random.randint(rng, (52,), 0, 4)


HASH_SIZE = 2_500_000
N = int(sys.argv[1])

rng = jax.random.PRNGKey(0)
VALUES = jax.random.split(rng, HASH_SIZE)
BOARDS = to_board(VALUES)
KEYS = to_key(BOARDS)
# print(VALUES[:10])
# print(BOARDS[:10])
# print(KEYS[:10])


@jax.vmap
def find_key(key):
    ix = jnp.argmin(jnp.abs(KEYS - key).sum(axis=1))
    return VALUES[ix]


@jax.jit
@jax.vmap
def find_key2(key):
    mask = jnp.where((KEYS == key).all(axis=1),
                     jnp.ones(HASH_SIZE, dtype=jnp.bool_),
                     jnp.zeros(HASH_SIZE, dtype=jnp.bool_))
    ix = jnp.argmax(mask)
    return VALUES[ix]


st = time.time()
find_key2(KEYS[:N])
et = time.time()
print(f"{et - st:.5f} sec")
st = time.time()
find_key2(KEYS[:N])
et = time.time()
print(f"{et - st:.5f} sec")
results = find_key2(KEYS[:N])
# print(results[:10])
