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
    assert jnp.unique(s.piece_board).shape[0] == 1 + 8 + 8
    assert s.legal_action_mask.sum() != 0
    legal_actions = jnp.int32([5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331])
    assert (jnp.nonzero(s.legal_action_mask)[0] == legal_actions).all()


def test_is_legal_drop():
    state = State._from_sfen("l2gg1b2/p1s1k3p/n1r2Pr+Pl/1G1p4P/2pP3n1/Pp2ppSB1/N2S3+pN/L1GK4L/2P3S2 b 3Pp 135")
    from_, to = 14, 46
    move = from_ * 81 + to
    assert _is_legal_move(state.piece_board, move, FALSE)
    from_, to = 14, 54
    move = from_ * 81 + to
    assert _is_legal_move(state.piece_board, move, FALSE)


def test_step():
    # data = """{"sfen_before": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", "action": 115, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331], "sfen_after": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL w - 2"}"""
    # for line in data.split("\n"):
    with open("tests/assets/shogi/random_play.json") as f:
        for line in f:
            d = json.loads(line)
            sfen = d["sfen_before"]
            state = State._from_sfen(sfen)
            expected_legal_actions = d["legal_actions"]
            legal_actions = jnp.nonzero(state.legal_action_mask)[0]
            ok = legal_actions.shape[0] == len(expected_legal_actions)
            if not ok:
                visualize(state, "tests/assets/shogi2/failed.svg")
                for a in legal_actions:
                    if a not in expected_legal_actions:
                        print(Action._from_dlshogi_action(state, a))
                for a in expected_legal_actions:
                    if a not in legal_actions:
                        print(Action._from_dlshogi_action(state, a))
                assert False, f"{legal_actions.shape[0]} != {len(expected_legal_actions)}, {sfen}"
            action = int(d["action"])
            state = step(state, action)
            sfen = d["sfen_after"]
            assert state._to_sfen() == sfen
