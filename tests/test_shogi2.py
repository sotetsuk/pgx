import json
from functools import partial
import jax
import jax.numpy as jnp

from pgx._shogi_utils import *
from pgx._shogi_utils import _rotate
from pgx._shogi import Shogi, State, Action, _is_legal_move, _is_legal_drop


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

    legal_actions = jnp.int32([5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331])
    print(Action._from_dlshogi_action(s, 241))
    print(LEGAL_FROM_IDX[2, 79])
    assert jnp.nonzero(s.legal_action_mask)[0].shape[0] == legal_actions.shape[0], jnp.nonzero(s.legal_action_mask)[0]
    assert (jnp.nonzero(s.legal_action_mask)[0] == legal_actions).all(), jnp.nonzero(s.legal_action_mask)[0]


def test_step():
    # sfen = "9/4R4/9/9/9/9/9/9/9 b 2r2b4g3s4n4l17p 1"
    # state = State._from_sfen(sfen)
    # visualize(state, "tests/assets/shogi2/test_step_001.svg")
    # dlshogi_action = 846
    # state = step(state, dlshogi_action)
    # sfen = "4+R4/9/9/9/9/9/9/9/9 w 2r2b4g3s4n4l7p 1"
    # expected_state = State._from_sfen(sfen)
    # visualize(expected_state, "tests/assets/shogi2/test_step_002.svg")
    # assert (state.piece_board == expected_state.piece_board).all()

    data = """{"sfen_before": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", "action": 115, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331], "sfen_after": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL w - 2"}
{"sfen_before": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL w - 2", "action": 304, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331], "sfen_after": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL b - 3"}
{"sfen_before": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL b - 3", "action": 142, "legal_actions": [5, 7, 14, 23, 32, 41, 43, 50, 52, 59, 61, 68, 77, 79, 124, 133, 142, 187, 205, 214, 268, 331, 350, 593], "sfen_after": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL w - 4"}
{"sfen_before": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL w - 4", "action": 77, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 68, 77, 79, 115, 124, 133, 178, 187, 196, 205, 214, 331, 340, 349, 358, 367, 376], "sfen_after": "lnsgkgsnl/6rb1/pppppppp1/8p/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL b - 5"}
{"sfen_before": "lnsgkgsnl/6rb1/pppppppp1/8p/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL b - 5", "action": 32, "legal_actions": [5, 7, 14, 23, 32, 41, 43, 50, 59, 68, 77, 79, 124, 133, 187, 214, 268, 296, 331, 350, 376, 593], "sfen_after": "lnsgkgsnl/6rb1/pppppppp1/8p/9/5P3/PPPPP1PPP/1BG2S1R1/LNS1KG1NL w - 6"}"""

    for line in data.split("\n"):
        d = json.loads(line)
        sfen = d["sfen_before"]
        state = State._from_sfen(sfen)
        action = int(d["action"])
        state = step(state, action)
        sfen = d["sfen_after"]
        assert state._to_sfen() == sfen


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
    assert _is_legal_move(s.piece_board, move, FALSE)
    # 58はNG
    from_, to = xy2i(6, 8), xy2i(5, 8)
    move = from_ * 81 + to
    assert not _is_legal_move(s.piece_board, move, FALSE)

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
    assert _is_legal_move(s.piece_board, move, FALSE)
    from_, to = xy2i(5, 9), xy2i(5, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move, FALSE)
    from_, to = xy2i(5, 9), xy2i(6, 8)
    move = from_ * 81 + to
    assert _is_legal_move(s.piece_board, move, FALSE)
    # 放置はNG
    from_, to = xy2i(1, 7), xy2i(1, 6)
    move = from_ * 81 + to
    assert not _is_legal_move(s.piece_board, move, FALSE)