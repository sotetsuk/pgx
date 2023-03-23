from functools import partial
import jax
import jax.numpy as jnp

from pgx._shogi_utils import *
from pgx._shogi_utils import _rotate
from pgx._shogi import Shogi, _is_legal_move


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
    # visualize(s, "tests/assets/shogi/init_board.svg")
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8
    assert s.legal_action_mask.sum() != 0
    # print(s.legal_action_mask.shape)
    # print(_rotate(s.legal_action_mask[6 * 81: 7*81].reshape(9,9)))
    # print(_rotate(s.legal_action_mask[8 * 81: 9*81].reshape(9,9)))
    # assert False


def test_is_legal_move():
    # King cannot move into opponent pieces' effect
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[xy2i(5, 5)].set(OPP_LANCE)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(6, 8)].set(KING)
        .at[xy2i(5, 9)].set(EMPTY)
    )
    visualize(s, "tests/assets/shogi/legal_moves_003.svg")
    from_, to = xy2i(6, 8), xy2i(5, 8)
    move = from_ * 81 + to
    assert not _is_legal_move(s.piece_board, move)