import sys
from urllib.request import urlopen

import jax.numpy as jnp
from jax import Array


def xor_reduce(operand: Array, axis: int = 0) -> Array:
    """XOR-reduce an unsigned-integer array along ``axis`` via per-bit parity.

    Equivalent to ``lax.reduce(operand, 0, lax.bitwise_xor, (axis,))`` but built only from
    ``sum`` / shifts / bitwise-and, because the ``bitwise_xor`` reduction primitive fails to
    legalize on the Apple Metal (``jax-metal``) XLA backend
    (``UNIMPLEMENTED: failed to legalize operation 'mhlo.reduce'``). Numerically identical
    on every backend; used for Zobrist hashing so chess/go/etc. run on Apple Silicon GPUs.
    """
    nbits = jnp.iinfo(operand.dtype).bits
    bitpos = jnp.arange(nbits, dtype=operand.dtype)
    bits = (jnp.expand_dims(operand, -1) >> bitpos) & 1  # (..., nbits)
    parity = jnp.sum(bits, axis=axis) & 1  # reduce the requested axis, keep bit axis
    weights = jnp.ones((), operand.dtype) << bitpos
    return jnp.sum(parity.astype(operand.dtype) * weights, axis=-1).astype(operand.dtype)


def _download(url, filename):
    try:
        print(f"Downloading from {url} ...", file=sys.stderr)
        data = urlopen(url).read()
        with open(filename, mode="wb") as f:
            f.write(data)
    except Exception as e:
        print(f"Failed to downalod the data from {url}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
