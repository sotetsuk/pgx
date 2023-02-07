import jax.numpy as jnp

from pgx.shogi import State,  init,  to_sfen


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_to_sfen():
    sfen = to_sfen(init())
    assert (sfen == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    pb = jnp.int8([15, -1, -1, 14, -1, 0, -1, -1, 1, -1, -1, -1, 1, 14, -1, -1, -1, 5, 8, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 6, 14, -1, 0, -1, 6, -1, -1, -1, -1, 14, -1, -1, 0, -1, 7, -1, -1, -1, -1, -1, -1, 2, 3, -1, -1, -1, 20, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, 16, 5, 0, -1, -1, -1, 8, -1, -1, -1, 1, -1, -1, -1])
    hand = jnp.int8([[1, 0, 0, 1, 0, 0, 0], [6, 0, 2, 2, 2, 0, 1]])
    s = State(turn=1, piece_board=pb, hand=hand)
    sfen2 = to_sfen(s)
    assert (sfen2 == "6+P1l/+P8/2g2G3/4pp1Lp/1nk3Pp1/LR3P2P/1P1NP4/3S1G3/4K2RL w SP2bg2s2n6p 1")
