from functools import partial
import jax
import jax.numpy as jnp

from pgx._shogi_utils import *
from pgx._shogi_utils import _rotate
from pgx._shogi import Shogi, _is_legal_move, _is_legal_drop


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


def test_is_legal_drop():
    # 駒がある
    key = jax.random.PRNGKey(0)
    s = init(key)
    assert not _is_legal_drop(s.piece_board, s.hand.at[:].set(1), PAWN, xy2i(5, 7))

    # 持ってない
    s = init(key)
    assert not _is_legal_drop(s.piece_board, s.hand, PAWN, xy2i(5, 5))

    # 合駒
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[:].set(EMPTY)
        .at[xy2i(1, 9)].set(KING)
        .at[xy2i(1, 5)].set(OPP_LANCE),
        hand=s.hand.at[0, GOLD].set(1)
    )
    visualize(s, "tests/assets/shogi/legal_drops_005.svg")
    assert _is_legal_drop(s.piece_board, s.hand, GOLD, xy2i(1, 6))  # 合駒はOK
    assert _is_legal_drop(s.piece_board, s.hand, GOLD, xy2i(1, 7))  # 合駒はOK
    assert _is_legal_drop(s.piece_board, s.hand, GOLD, xy2i(1, 8))  # 合駒はOK
    assert not _is_legal_drop(s.piece_board, s.hand, GOLD, xy2i(2, 6))  # 合駒はOK

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
    visualize(s, "tests/assets/shogi2/legal_moves_001.svg")
    # 78はOK
    from_, to = xy2i(6, 8), xy2i(7, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move)
    # 58はNG
    from_, to = xy2i(6, 8), xy2i(5, 8)
    move = from_ * 81 + to
    assert not _is_legal_move(s.piece_board, move)

    # King must escape
    key = jax.random.PRNGKey(0)
    s = init(key)
    s = update_board(s,
        piece_board=s.piece_board.at[xy2i(5, 3)].set(EMPTY)
        .at[xy2i(5, 7)].set(EMPTY)
        .at[xy2i(5, 8)].set(OPP_PAWN)
    )
    visualize(s, "tests/assets/shogi2/legal_moves_002.svg")
    # 王が逃げるのはOK
    from_, to = xy2i(5, 9), xy2i(4, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move)
    from_, to = xy2i(5, 9), xy2i(5, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move)
    from_, to = xy2i(5, 9), xy2i(6, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move)
    # 放置はNG
    from_, to = xy2i(1, 7), xy2i(1, 6)
    move = from_ * 81 + to
    assert not _is_legal_move(s.piece_board, move)