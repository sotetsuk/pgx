import sys
from urllib.request import urlopen

import jax
import jax.numpy as jnp
from jax import Array, lax


def _xor_reduce_bitparity(operand: Array, axis: int) -> Array:
    # Metal fallback: the ``bitwise_xor`` reduction primitive fails to legalize on the
    # Apple Metal (``jax-metal``) XLA backend
    # (``UNIMPLEMENTED: failed to legalize operation 'mhlo.reduce'``). Compute the XOR via
    # per-bit parity using only ``sum`` / shifts / bitwise-and. Numerically identical to the
    # native reduction, but expands each value into its bits, so it is used only on Metal.
    nbits = jnp.iinfo(operand.dtype).bits
    bitpos = jnp.arange(nbits, dtype=operand.dtype)
    bits = (jnp.expand_dims(operand, -1) >> bitpos) & 1  # (..., nbits)
    parity = jnp.sum(bits, axis=axis) & 1  # reduce the requested axis, keep bit axis
    weights = jnp.ones((), operand.dtype) << bitpos
    return jnp.sum(parity.astype(operand.dtype) * weights, axis=-1).astype(operand.dtype)


def xor_reduce(operand: Array, axis: int = 0) -> Array:
    """XOR-reduce an unsigned-integer array along ``axis``.

    Uses the native ``bitwise_xor`` reduction on CPU/CUDA/TPU, and falls back to a
    numerically identical per-bit-parity implementation only on the Apple Metal
    (``jax-metal``) backend, where the native reduction fails to legalize. The native path
    avoids the per-bit expansion of the fallback (~15x faster on CPU, ~100x on CUDA), which
    matters because this runs once per env step for Zobrist hashing. The backend is resolved
    at trace time, so jitted code pays no runtime cost for the check.
    """
    if "metal" in jax.default_backend().lower():
        return _xor_reduce_bitparity(operand, axis)
    return lax.reduce(operand, jnp.zeros((), operand.dtype), lax.bitwise_xor, (axis,))


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
