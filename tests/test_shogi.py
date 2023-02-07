import jax.numpy as jnp

from pgx.shogi import init, _to_sfen


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_to_sfen():
    sfen = _to_sfen(init())
    assert (sfen == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
