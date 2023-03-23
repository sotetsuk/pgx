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


def test_step():
    data = """{"sfen_before": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", "action": 115, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331], "sfen_after": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL w - 2"}
{"sfen_before": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL w - 2", "action": 304, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331], "sfen_after": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL b - 3"}
{"sfen_before": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1B3S1R1/LNSGKG1NL b - 3", "action": 142, "legal_actions": [5, 7, 14, 23, 32, 41, 43, 50, 52, 59, 61, 68, 77, 79, 124, 133, 142, 187, 205, 214, 268, 331, 350, 593], "sfen_after": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL w - 4"}
{"sfen_before": "lnsgkgsnl/6rb1/ppppppppp/9/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL w - 4", "action": 77, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 68, 77, 79, 115, 124, 133, 178, 187, 196, 205, 214, 331, 340, 349, 358, 367, 376], "sfen_after": "lnsgkgsnl/6rb1/pppppppp1/8p/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL b - 5"}
{"sfen_before": "lnsgkgsnl/6rb1/pppppppp1/8p/9/9/PPPPPPPPP/1BG2S1R1/LNS1KG1NL b - 5", "action": 32, "legal_actions": [5, 7, 14, 23, 32, 41, 43, 50, 59, 68, 77, 79, 124, 133, 187, 214, 268, 296, 331, 350, 376, 593], "sfen_after": "lnsgkgsnl/6rb1/pppppppp1/8p/9/5P3/PPPPP1PPP/1BG2S1R1/LNS1KG1NL w - 6"}
{"sfen_before": "lnsgkgsnl/6rb1/pppppppp1/8p/9/5P3/PPPPP1PPP/1BG2S1R1/LNS1KG1NL w - 6", "action": 34, "legal_actions": [5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 68, 76, 78, 79, 115, 124, 133, 159, 178, 187, 196, 205, 214, 331, 340, 349, 358, 367, 376, 726], "sfen_after": "lns1kgsnl/3g2rb1/pppppppp1/8p/9/5P3/PPPPP1PPP/1BG2S1R1/LNS1KG1NL b - 7"}
{"sfen_before": "lns1kgsnl/3g2rb1/pppppppp1/8p/9/5P3/PPPPP1PPP/1BG2S1R1/LNS1KG1NL b - 7", "action": 31, "legal_actions": [5, 7, 14, 23, 31, 33, 41, 43, 50, 59, 68, 77, 79, 124, 133, 187, 214, 268, 296, 331, 350, 376, 593], "sfen_after": "lns1kgsnl/3g2rb1/pppppppp1/8p/5P3/9/PPPPP1PPP/1BG2S1R1/LNS1KG1NL w - 8"}
{"sfen_before": "lns1kgsnl/3g2rb1/pppppppp1/8p/5P3/9/PPPPP1PPP/1BG2S1R1/LNS1KG1NL w - 8", "action": 14, "legal_actions": [5, 7, 14, 23, 25, 32, 41, 43, 50, 52, 59, 68, 76, 78, 79, 133, 159, 178, 205, 214, 286, 349, 359, 367, 376, 440, 726], "sfen_after": "lns1kgsnl/3g2rb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1R1/LNS1KG1NL b - 9"}
{"sfen_before": "lns1kgsnl/3g2rb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1R1/LNS1KG1NL b - 9", "action": 7, "legal_actions": [5, 7, 14, 23, 30, 33, 41, 43, 50, 59, 68, 77, 79, 124, 133, 187, 214, 268, 296, 331, 350, 376, 593], "sfen_after": "lns1kgsnl/3g2rb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1RL/LNS1KG1N1 w - 10"}
{"sfen_before": "lns1kgsnl/3g2rb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1RL/LNS1KG1N1 w - 10", "action": 52, "legal_actions": [5, 7, 13, 23, 25, 32, 41, 43, 50, 52, 59, 68, 76, 78, 79, 133, 159, 178, 205, 214, 286, 349, 359, 367, 376, 440, 726], "sfen_after": "lns1k1snl/3g1grb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1RL/LNS1KG1N1 b - 11"}
{"sfen_before": "lns1k1snl/3g1grb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG2S1RL/LNS1KG1N1 b - 11", "action": 43, "legal_actions": [5, 14, 23, 30, 33, 41, 43, 50, 59, 68, 77, 79, 124, 133, 187, 214, 268, 296, 350, 376, 593], "sfen_after": "lns1k1snl/3g1grb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG1KS1RL/LNS2G1N1 w - 12"}
{"sfen_before": "lns1k1snl/3g1grb1/p1pppppp1/1p6p/5P3/9/PPPPP1PPP/1BG1KS1RL/LNS2G1N1 w - 12", "action": 41, "legal_actions": [5, 7, 13, 23, 25, 32, 41, 43, 50, 59, 68, 76, 78, 79, 159, 178, 286, 296, 349, 359, 367, 440, 458, 726], "sfen_after": "lns1k1snl/3g1grb1/p1pp1ppp1/1p2p3p/5P3/9/PPPPP1PPP/1BG1KS1RL/LNS2G1N1 b - 13"}
{"sfen_before": "lns1k1snl/3g1grb1/p1pp1ppp1/1p2p3p/5P3/9/PPPPP1PPP/1BG1KS1RL/LNS2G1N1 b - 13", "action": 59, "legal_actions": [5, 14, 23, 30, 33, 41, 50, 59, 68, 77, 79, 187, 195, 214, 268, 287, 295, 350, 376, 449, 530, 539, 593], "sfen_after": "lns1k1snl/3g1grb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS2G1N1 w - 14"}
{"sfen_before": "lns1k1snl/3g1grb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS2G1N1 w - 14", "action": 43, "legal_actions": [5, 7, 13, 23, 25, 32, 40, 43, 50, 59, 68, 76, 78, 79, 123, 159, 178, 204, 286, 296, 349, 359, 367, 440, 458, 726], "sfen_after": "lns3snl/3gkgrb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS2G1N1 b - 15"}
{"sfen_before": "lns3snl/3gkgrb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS2G1N1 b - 15", "action": 287, "legal_actions": [5, 14, 23, 30, 33, 41, 50, 58, 60, 68, 77, 79, 182, 187, 192, 195, 202, 212, 214, 222, 268, 287, 295, 350, 376, 449, 530, 539, 593, 789, 992], "sfen_after": "lns3snl/3gkgrb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS1G2N1 w - 16"}
{"sfen_before": "lns3snl/3gkgrb1/p1pp1ppp1/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS1G2N1 w - 16", "action": 726, "legal_actions": [5, 7, 13, 23, 25, 32, 40, 42, 50, 59, 68, 76, 78, 79, 123, 159, 178, 204, 349, 440, 449, 458, 539, 602, 726], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pppn/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS1G2N1 b - 17"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pppn/1p2p3p/5P3/2P6/PP1PP1PPP/1BG1KS1RL/LNS1G2N1 b - 17", "action": 195, "legal_actions": [5, 14, 23, 30, 33, 41, 50, 58, 60, 68, 77, 79, 133, 182, 192, 195, 202, 212, 214, 222, 268, 295, 296, 359, 376, 539, 593, 602, 789, 992], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pppn/1p2p3p/5P3/2P6/PP1PPKPPP/1BG2S1RL/LNS1G2N1 w - 18"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pppn/1p2p3p/5P3/2P6/PP1PPKPPP/1BG2S1RL/LNS1G2N1 w - 18", "action": 68, "legal_actions": [5, 7, 13, 23, 25, 32, 40, 42, 50, 59, 68, 76, 79, 123, 178, 204, 349, 440, 449, 458, 539, 602, 796], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pp1n/1p2p2pp/5P3/2P6/PP1PPKPPP/1BG2S1RL/LNS1G2N1 b - 19"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pp1n/1p2p2pp/5P3/2P6/PP1PPKPPP/1BG2S1RL/LNS1G2N1 b - 19", "action": 58, "legal_actions": [5, 14, 23, 30, 32, 41, 43, 50, 58, 60, 68, 77, 79, 122, 133, 182, 185, 192, 202, 212, 214, 222, 268, 296, 359, 376, 529, 592, 593, 789, 992], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pp1n/1p2p2pp/2P2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 w - 20"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pp1n/1p2p2pp/2P2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 w - 20", "action": 13, "legal_actions": [5, 7, 13, 23, 25, 32, 40, 42, 50, 59, 67, 76, 79, 123, 178, 204, 349, 440, 449, 458, 539, 602, 796], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 b - 21"}"""

    for line in data.split("\n"):
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
            assert False, f"{legal_actions.shape[0]} != {len(expected_legal_actions)}"
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
    assert not jax.jit(_is_legal_move)(s.piece_board, move, FALSE)

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