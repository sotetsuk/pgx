import jax
import jax.numpy as jnp
from pgx.suzume_jong import _is_completed


def test_is_completed():
    hand = jnp.int8([0,0,1,1,1,0,0,0,0,3,0])
    assert _is_completed(hand)
    hand = jnp.int8([0,0,1,1,1,0,0,0,0,2,1])
    assert not _is_completed(hand)
    hand = jnp.int8([0,0,1,0,1,0,0,0,0,3,0])
    assert not _is_completed(hand)
    hand = jnp.int8([0,0,1,2,1,0,0,0,0,3,0])
    assert not _is_completed(hand)
