import jax.numpy as jnp

from pgx.shogi import init


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8
