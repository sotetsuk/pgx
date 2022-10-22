import numpy as np

from pgx.backgammon import init


def test_init():
    state = init()
    assert state.turn[0] == 1 or state.turn[0] == 0
