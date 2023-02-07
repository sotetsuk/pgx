import jax.numpy as jnp

from pgx.shogi import State, init, to_sfen


def test_init():
    s = init()
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8


def test_to_sfen():
    sfen = to_sfen(init())
    assert (sfen == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    pb = jnp.int8(
        [16, 0, 0, 15, 0, 1, 0, 0, 2, 0, 0, 0, 2, 15, 0, 0, 0, 6, 9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 7, 15, 0, 1, 0,
         7, 0, 0, 0, 0, 15, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 21, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 17,
         6, 1, 0, 0, 0, 9, 0, 0, 0, 2, 0, 0, 0])
    hand = jnp.int8([[1, 0, 0, 1, 0, 0, 0], [6, 0, 2, 2, 2, 0, 1]])
    s = State(turn=1, piece_board=pb, hand=hand)
    sfen2 = to_sfen(s)
    assert (sfen2 == "6+P1l/+P8/2g2G3/4pp1Lp/1nk3Pp1/LR3P2P/1P1NP4/3S1G3/4K2RL w SP2bg2s2n6p 1")
