from functools import partial
import jax
import jax.numpy as jnp

from pgx._shogi_utils import _rotate
from pgx._shogi import Shogi


env = Shogi()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def xy2i(x, y):
    """
    >>> xy2i(2, 6)  # 26歩
    14
    """
    i = (x - 1) * 9 + (y - 1)
    return i


# check visualization results by image preview plugins
def visualize(state, fname="tests/assets/shogi/xxx.svg"):
    from pgx._visualizer import Visualizer
    v = Visualizer(color_theme="dark")
    v.save_svg(state, fname)


def update_board(state, piece_board, hand=None):
    state = state.replace(piece_board=piece_board)
    if hand is not None:
        state = state.replace(hand=hand)
    return state


def test_init():
    key = jax.random.PRNGKey(0)
    s = init(key)
    visualize(s, "tests/assets/shogi/tmp.svg")
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8
    assert s.legal_action_mask.sum() == 0


