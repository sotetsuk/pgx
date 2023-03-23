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
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pp1n/1p2p2pp/2P2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 w - 20", "action": 13, "legal_actions": [5, 7, 13, 23, 25, 32, 40, 42, 50, 59, 67, 76, 79, 123, 178, 204, 349, 440, 449, 458, 539, 602, 796], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 b - 21"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNS1G2N1 b - 21", "action": 296, "legal_actions": [5, 14, 23, 30, 32, 41, 43, 50, 57, 60, 68, 77, 79, 122, 133, 182, 185, 192, 202, 212, 214, 222, 268, 296, 359, 376, 529, 592, 593, 789, 992], "sfen_after": "lns3s1l/3gkgrb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNSG3N1 w - 22"}
{"sfen_before": "lns3s1l/3gkgrb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNSG3N1 w - 22", "action": 458, "legal_actions": [5, 7, 12, 23, 25, 32, 40, 42, 50, 59, 67, 76, 79, 123, 178, 204, 349, 440, 449, 458, 539, 602, 796], "sfen_after": "lns2gs1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNSG3N1 b - 23"}
{"sfen_before": "lns2gs1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1BG2S1RL/LNSG3N1 b - 23", "action": 376, "legal_actions": [5, 14, 23, 30, 32, 41, 50, 52, 57, 60, 68, 77, 79, 122, 182, 185, 192, 202, 205, 212, 214, 222, 268, 368, 376, 529, 530, 592, 593, 789, 992], "sfen_after": "lns2gs1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1B1G1S1RL/LNSG3N1 w - 24"}
{"sfen_before": "lns2gs1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1B1G1S1RL/LNSG3N1 w - 24", "action": 368, "legal_actions": [5, 7, 12, 23, 25, 32, 40, 42, 50, 52, 59, 67, 76, 79, 123, 178, 214, 295, 349, 368, 376, 440, 449, 602, 796], "sfen_after": "lns1g1s1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1B1G1S1RL/LNSG3N1 b - 25"}
{"sfen_before": "lns1g1s1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/9/PP1PPKPPP/1B1G1S1RL/LNSG3N1 b - 25", "action": 50, "legal_actions": [5, 14, 23, 30, 32, 41, 50, 57, 61, 68, 77, 79, 122, 141, 142, 182, 185, 192, 202, 205, 212, 222, 268, 304, 367, 368, 529, 530, 592, 593, 789, 992], "sfen_after": "lns1g1s1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1B1G1S1RL/LNSG3N1 w - 26"}
{"sfen_before": "lns1g1s1l/3gk1rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1B1G1S1RL/LNSG3N1 w - 26", "action": 602, "legal_actions": [5, 7, 12, 23, 25, 32, 40, 42, 50, 59, 67, 76, 79, 123, 133, 178, 214, 295, 296, 349, 359, 376, 440, 539, 602, 796], "sfen_after": "lnskg1s1l/3g2rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1B1G1S1RL/LNSG3N1 b - 27"}
{"sfen_before": "lnskg1s1l/3g2rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1B1G1S1RL/LNSG3N1 b - 27", "action": 61, "legal_actions": [5, 14, 23, 30, 32, 41, 49, 51, 57, 61, 68, 77, 79, 122, 141, 142, 185, 205, 222, 268, 304, 367, 368, 529, 530, 592, 593, 789], "sfen_after": "lnskg1s1l/3g2rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1BSG1S1RL/LN1G3N1 w - 28"}
{"sfen_before": "lnskg1s1l/3g2rb1/p1pp1pp1n/4p2pp/1pP2P3/3P5/PP2PKPPP/1BSG1S1RL/LN1G3N1 w - 28", "action": 5, "legal_actions": [5, 7, 12, 23, 25, 32, 40, 43, 50, 59, 67, 76, 79, 123, 124, 133, 178, 187, 214, 286, 296, 349, 367, 376, 796], "sfen_after": "lnskg1s1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/1BSG1S1RL/LN1G3N1 b - 29"}
{"sfen_before": "lnskg1s1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/1BSG1S1RL/LN1G3N1 b - 29", "action": 79, "legal_actions": [5, 14, 23, 30, 32, 41, 49, 51, 57, 60, 68, 77, 79, 122, 141, 185, 205, 213, 222, 268, 305, 367, 368, 529, 530, 592, 593, 629, 789], "sfen_after": "lnskg1s1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/LBSG1S1RL/1N1G3N1 w - 30"}
{"sfen_before": "lnskg1s1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/LBSG1S1RL/1N1G3N1 w - 30", "action": 133, "legal_actions": [4, 6, 7, 12, 23, 25, 32, 40, 43, 50, 59, 67, 76, 79, 123, 124, 133, 178, 187, 214, 286, 296, 349, 367, 376, 735, 796], "sfen_after": "lnsk2s1l/3g1grb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/LBSG1S1RL/1N1G3N1 b - 31"}
{"sfen_before": "lnsk2s1l/3g1grb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP2PKPPP/LBSG1S1RL/1N1G3N1 b - 31", "action": 51, "legal_actions": [5, 14, 23, 30, 32, 41, 49, 51, 57, 60, 68, 77, 122, 141, 185, 205, 213, 222, 268, 305, 367, 368, 529, 530, 566, 592, 593, 629, 789], "sfen_after": "lnsk2s1l/3g1grb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP1GPKPPP/LBS2S1RL/1N1G3N1 w - 32"}
{"sfen_before": "lnsk2s1l/3g1grb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP1GPKPPP/LBS2S1RL/1N1G3N1 w - 32", "action": 458, "legal_actions": [4, 6, 7, 12, 23, 25, 32, 40, 50, 59, 67, 76, 79, 123, 124, 178, 187, 204, 286, 287, 349, 367, 458, 735, 796], "sfen_after": "lnsk1gs1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP1GPKPPP/LBS2S1RL/1N1G3N1 b - 33"}
{"sfen_before": "lnsk1gs1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/3P5/PP1GPKPPP/LBS2S1RL/1N1G3N1 b - 33", "action": 140, "legal_actions": [5, 14, 23, 30, 32, 41, 49, 52, 57, 60, 68, 77, 122, 140, 185, 203, 205, 222, 268, 303, 305, 368, 457, 529, 530, 566, 592, 593, 629, 789], "sfen_after": "lnsk1gs1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/2GP5/PP2PKPPP/LBS2S1RL/1N1G3N1 w - 34"}
{"sfen_before": "lnsk1gs1l/3g2rb1/2pp1pp1n/p3p2pp/1pP2P3/2GP5/PP2PKPPP/LBS2S1RL/1N1G3N1 w - 34", "action": 32, "legal_actions": [4, 6, 7, 12, 23, 25, 32, 40, 50, 52, 59, 67, 76, 79, 123, 124, 178, 187, 205, 214, 286, 287, 349, 367, 368, 376, 735, 796], "sfen_after": "lnsk1gs1l/3g2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2PKPPP/LBS2S1RL/1N1G3N1 b - 35"}
{"sfen_before": "lnsk1gs1l/3g2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2PKPPP/LBS2S1RL/1N1G3N1 b - 35", "action": 529, "legal_actions": [5, 14, 23, 30, 32, 41, 49, 52, 57, 60, 68, 77, 122, 148, 185, 205, 211, 213, 222, 268, 305, 311, 368, 465, 529, 530, 566, 592, 593, 629, 789], "sfen_after": "lnsk1gs1l/3g2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2P1PPP/LBS1KS1RL/1N1G3N1 w - 36"}
{"sfen_before": "lnsk1gs1l/3g2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2P1PPP/LBS1KS1RL/1N1G3N1 w - 36", "action": 25, "legal_actions": [4, 6, 7, 12, 23, 25, 31, 33, 40, 50, 52, 59, 67, 76, 79, 123, 124, 178, 187, 205, 214, 286, 287, 349, 367, 368, 376, 735, 796], "sfen_after": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2P1PPP/LBS1KS1RL/1N1G3N1 b - 37"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PP2P1PPP/LBS1KS1RL/1N1G3N1 b - 37", "action": 222, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 60, 68, 77, 132, 148, 195, 211, 213, 222, 268, 295, 305, 311, 368, 449, 465, 530, 566, 593, 602, 629, 789], "sfen_after": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PPB1P1PPP/L1S1KS1RL/1N1G3N1 w - 38"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2pp/1pP2P3/2GP5/PPB1P1PPP/L1S1KS1RL/1N1G3N1 w - 38", "action": 76, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 50, 52, 59, 67, 76, 79, 114, 123, 124, 177, 205, 214, 286, 287, 350, 367, 368, 376, 735, 796], "sfen_after": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2p1/1pP2P2p/2GP5/PPB1P1PPP/L1S1KS1RL/1N1G3N1 b - 39"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2p1/1pP2P2p/2GP5/PPB1P1PPP/L1S1KS1RL/1N1G3N1 b - 39", "action": 566, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 68, 77, 132, 148, 149, 157, 195, 211, 213, 268, 295, 305, 311, 368, 449, 530, 556, 566, 593, 602, 611, 619], "sfen_after": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2p1/1pP2P2p/2GP5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 40"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p2pp1n/p2pp2p1/1pP2P2p/2GP5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 40", "action": 50, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 50, 52, 59, 67, 75, 79, 114, 123, 124, 177, 205, 214, 286, 287, 350, 367, 368, 376, 735, 796], "sfen_after": "ln1k1gs1l/2sg2rb1/2p3p1n/p2ppp1p1/1pP2P2p/2GP5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 41"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p3p1n/p2ppp1p1/1pP2P2p/2GP5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 41", "action": 311, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 60, 68, 77, 132, 148, 195, 211, 213, 222, 232, 268, 295, 305, 311, 368, 449, 465, 530, 593, 602, 789], "sfen_after": "ln1k1gs1l/2sg2rb1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 42"}
{"sfen_before": "ln1k1gs1l/2sg2rb1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 42", "action": 376, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 49, 52, 59, 67, 75, 79, 114, 123, 124, 177, 205, 214, 286, 287, 350, 367, 368, 376, 735, 796], "sfen_after": "ln1k1gs1l/2sg1r1b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 43"}
{"sfen_before": "ln1k1gs1l/2sg1r1b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 43", "action": 305, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 60, 67, 77, 132, 157, 195, 213, 222, 232, 268, 295, 305, 320, 368, 383, 449, 530, 593, 602, 789], "sfen_after": "ln1k1gs1l/2sg1r1b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BNG4N1 w - 44"}
{"sfen_before": "ln1k1gs1l/2sg1r1b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BNG4N1 w - 44", "action": 367, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 49, 51, 59, 61, 67, 75, 79, 114, 123, 124, 142, 177, 205, 286, 287, 304, 350, 367, 368, 735, 796], "sfen_after": "ln1k1gs1l/2sgr2b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BNG4N1 b - 45"}
{"sfen_before": "ln1k1gs1l/2sgr2b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BNG4N1 b - 45", "action": 377, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 57, 60, 67, 77, 132, 151, 157, 195, 213, 214, 222, 232, 268, 295, 320, 377, 383, 449, 530, 539, 593, 602, 620, 789], "sfen_after": "ln1k1gs1l/2sgr2b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 46"}
{"sfen_before": "ln1k1gs1l/2sgr2b1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 w - 46", "action": 61, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 42, 49, 52, 59, 61, 67, 75, 79, 114, 123, 142, 177, 214, 287, 295, 304, 350, 368, 449, 735, 796], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 47"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P2p/1G1P5/PP2P1PPP/L1S1KS1RL/BN1G3N1 b - 47", "action": 77, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 60, 67, 77, 132, 157, 195, 213, 222, 232, 268, 295, 305, 320, 368, 383, 449, 530, 593, 602, 789], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P2p/PG1P5/1P2P1PPP/L1S1KS1RL/BN1G3N1 w - 48"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P2p/PG1P5/1P2P1PPP/L1S1KS1RL/BN1G3N1 w - 48", "action": 75, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 42, 49, 52, 59, 67, 75, 79, 114, 123, 150, 177, 213, 287, 295, 305, 350, 368, 449, 557, 629, 735, 796], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P3/PG1P4p/1P2P1PPP/L1S1KS1RL/BN1G3N1 b - 49"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P3/PG1P4p/1P2P1PPP/L1S1KS1RL/BN1G3N1 b - 49", "action": 14, "legal_actions": [5, 14, 23, 30, 33, 41, 49, 52, 57, 60, 67, 76, 78, 132, 157, 195, 213, 222, 232, 268, 295, 305, 368, 383, 449, 530, 593, 602, 726, 789], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P3/PG1P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 w - 50"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/p2ppp1p1/1pP2P3/PG1P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 w - 50", "action": 4, "legal_actions": [4, 6, 7, 12, 23, 31, 33, 40, 42, 49, 52, 59, 67, 74, 79, 114, 123, 150, 177, 213, 287, 295, 305, 350, 368, 449, 557, 629, 735, 796, 884], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/3ppp1p1/ppP2P3/PG1P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 b - 51"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/3ppp1p1/ppP2P3/PG1P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 b - 51", "action": 157, "legal_actions": [5, 13, 15, 23, 30, 33, 41, 49, 52, 57, 60, 67, 76, 78, 132, 157, 195, 213, 222, 232, 268, 295, 305, 368, 383, 449, 530, 593, 602, 726, 789], "sfen_after": "ln1k1g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 w P 52"}
{"sfen_before": "ln1k1g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 w P 52", "action": 350, "legal_actions": [4, 5, 6, 7, 12, 23, 31, 33, 40, 42, 49, 52, 59, 67, 74, 79, 114, 123, 150, 177, 213, 287, 295, 305, 350, 368, 449, 557, 629, 735, 796, 884], "sfen_after": "lnk2g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 b P 53"}
{"sfen_before": "lnk2g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3Pp/1P2P1P1P/L1S1KS1RL/BN1G3N1 b P 53", "action": 5, "legal_actions": [5, 13, 15, 23, 30, 33, 41, 49, 52, 57, 60, 68, 75, 78, 132, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 449, 530, 593, 602, 726, 789], "sfen_after": "lnk2g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P2/L1S1KS1RL/BN1G3N1 w 2P 54"}
{"sfen_before": "lnk2g2l/2sgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P2/L1S1KS1RL/BN1G3N1 w 2P 54", "action": 178, "legal_actions": [4, 5, 6, 7, 12, 23, 31, 33, 40, 42, 49, 52, 59, 67, 79, 114, 123, 150, 177, 178, 213, 278, 295, 305, 368, 440, 449, 521, 557, 629, 735, 796], "sfen_after": "ln3g2l/1ksgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P2/L1S1KS1RL/BN1G3N1 b 2P 55"}
{"sfen_before": "ln3g2l/1ksgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P2/L1S1KS1RL/BN1G3N1 b 2P 55", "action": 735, "legal_actions": [4, 6, 13, 15, 23, 30, 33, 41, 49, 52, 57, 60, 68, 75, 78, 132, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 449, 530, 593, 602, 726, 735, 789], "sfen_after": "ln3g2l/1ksgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P1N/L1S1KS1RL/BN1G5 w 2P 56"}
{"sfen_before": "ln3g2l/1ksgr1sb1/2p3p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P1N/L1S1KS1RL/BN1G5 w 2P 56", "action": 42, "legal_actions": [4, 5, 6, 7, 12, 15, 23, 31, 33, 40, 42, 49, 52, 59, 67, 79, 114, 123, 150, 168, 177, 213, 295, 305, 331, 368, 440, 449, 512, 521, 557, 629, 735, 796], "sfen_after": "ln3g2l/1ksg2sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P1N/L1S1KS1RL/BN1G5 b 2P 57"}
{"sfen_before": "ln3g2l/1ksg2sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/1P2P1P1N/L1S1KS1RL/BN1G5 b 2P 57", "action": 726, "legal_actions": [4, 13, 15, 23, 30, 33, 41, 49, 52, 57, 60, 68, 75, 78, 132, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 422, 449, 530, 593, 602, 661, 726, 789], "sfen_after": "ln3g2l/1ksg2sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/NP2P1P1N/L1S1KS1RL/B2G5 w 2P 58"}
{"sfen_before": "ln3g2l/1ksg2sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/NP2P1P1N/L1S1KS1RL/B2G5 w 2P 58", "action": 205, "legal_actions": [4, 5, 6, 7, 12, 15, 23, 31, 33, 40, 49, 52, 59, 67, 79, 114, 150, 168, 177, 205, 213, 286, 294, 305, 331, 357, 368, 440, 448, 449, 512, 521, 557, 629, 735, 796], "sfen_after": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/NP2P1P1N/L1S1KS1RL/B2G5 b 2P 59"}
{"sfen_before": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P3PP/NP2P1P1N/L1S1KS1RL/B2G5 b 2P 59", "action": 23, "legal_actions": [4, 13, 15, 23, 30, 33, 41, 49, 52, 57, 60, 68, 75, 132, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 422, 449, 530, 557, 593, 602, 661, 796], "sfen_after": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P2PPP/NP2P3N/L1S1KS1RL/B2G5 w 2P 60"}
{"sfen_before": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp1p1/GpP2P3/P2P2PPP/NP2P3N/L1S1KS1RL/B2G5 w 2P 60", "action": 67, "legal_actions": [4, 5, 6, 7, 12, 15, 23, 31, 33, 40, 49, 59, 67, 79, 114, 132, 150, 168, 177, 195, 213, 294, 295, 331, 357, 440, 449, 512, 521, 557, 620, 629, 735, 796], "sfen_after": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp3/GpP2P1p1/P2P2PPP/NP2P3N/L1S1KS1RL/B2G5 b 2P 61"}
{"sfen_before": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp3/GpP2P1p1/P2P2PPP/NP2P3N/L1S1KS1RL/B2G5 b 2P 61", "action": 4, "legal_actions": [4, 13, 15, 22, 30, 33, 41, 49, 52, 57, 60, 68, 75, 132, 186, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 422, 449, 530, 557, 593, 602, 661, 796], "sfen_after": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp3/GpP2P1pP/P2P2PP1/NP2P3N/L1S1KS1RL/B2G5 w 2P 62"}
{"sfen_before": "ln6l/1ksgg1sb1/2p1r1p1n/3ppp3/GpP2P1pP/P2P2PP1/NP2P3N/L1S1KS1RL/B2G5 w 2P 62", "action": 23, "legal_actions": [4, 5, 6, 7, 12, 15, 23, 31, 33, 40, 49, 59, 66, 79, 114, 132, 150, 168, 177, 195, 213, 294, 295, 331, 357, 440, 449, 512, 521, 557, 620, 629, 735], "sfen_after": "ln6l/1ksgg1sb1/4r1p1n/2pppp3/GpP2P1pP/P2P2PP1/NP2P3N/L1S1KS1RL/B2G5 b 2P 63"}
{"sfen_before": "ln6l/1ksgg1sb1/4r1p1n/2pppp3/GpP2P1pP/P2P2PP1/NP2P3N/L1S1KS1RL/B2G5 b 2P 63", "action": 222, "legal_actions": [3, 13, 15, 22, 30, 33, 41, 49, 52, 57, 60, 68, 75, 132, 186, 195, 213, 222, 228, 232, 268, 295, 305, 368, 391, 422, 449, 530, 557, 593, 602, 661, 796], "sfen_after": "ln6l/1ksgg1sb1/4r1p1n/2pppp3/GpP2P1pP/P2P2PP1/NPB1P3N/L1S1KS1RL/3G5 w 2P 64"}
{"sfen_before": "ln6l/1ksgg1sb1/4r1p1n/2pppp3/GpP2P1pP/P2P2PP1/NPB1P3N/L1S1KS1RL/3G5 w 2P 64", "action": 150, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 24, 31, 33, 40, 49, 59, 66, 79, 105, 114, 132, 150, 168, 177, 186, 195, 213, 294, 295, 330, 331, 339, 348, 357, 440, 449, 512, 521, 557, 620, 629, 672, 735], "sfen_after": "ln6l/1ksgg2b1/4r1psn/2pppp3/GpP2P1pP/P2P2PP1/NPB1P3N/L1S1KS1RL/3G5 b 2P 65"}
{"sfen_before": "ln6l/1ksgg2b1/4r1psn/2pppp3/GpP2P1pP/P2P2PP1/NPB1P3N/L1S1KS1RL/3G5 b 2P 65", "action": 41, "legal_actions": [3, 13, 15, 22, 30, 33, 41, 49, 52, 57, 68, 75, 132, 149, 186, 195, 213, 228, 268, 295, 305, 368, 391, 422, 449, 530, 556, 557, 566, 593, 602, 611, 619, 661, 796], "sfen_after": "ln6l/1ksgg2b1/4r1psn/2pppp3/GpP2P1pP/P2PP1PP1/NPB5N/L1S1KS1RL/3G5 w 2P 66"}
{"sfen_before": "ln6l/1ksgg2b1/4r1psn/2pppp3/GpP2P1pP/P2PP1PP1/NPB5N/L1S1KS1RL/3G5 w 2P 66", "action": 449, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 24, 31, 33, 40, 49, 59, 66, 68, 79, 105, 114, 132, 158, 168, 177, 186, 195, 221, 294, 295, 330, 331, 339, 348, 357, 440, 449, 512, 521, 565, 628, 629, 672, 735], "sfen_after": "ln2g3l/1ksg3b1/4r1psn/2pppp3/GpP2P1pP/P2PP1PP1/NPB5N/L1S1KS1RL/3G5 b 2P 67"}
{"sfen_before": "ln2g3l/1ksg3b1/4r1psn/2pppp3/GpP2P1pP/P2PP1PP1/NPB5N/L1S1KS1RL/3G5 b 2P 67", "action": 13, "legal_actions": [3, 13, 15, 22, 30, 33, 40, 42, 49, 52, 57, 68, 75, 123, 132, 149, 186, 195, 213, 228, 268, 295, 305, 368, 391, 422, 449, 530, 556, 557, 566, 593, 602, 611, 619, 661, 796], "sfen_after": "ln2g3l/1ksg3b1/4r1psn/2pppp3/GpP2P1PP/P2PP1P2/NPB5N/L1S1KS1RL/3G5 w 3P 68"}
{"sfen_before": "ln2g3l/1ksg3b1/4r1psn/2pppp3/GpP2P1PP/P2PP1P2/NPB5N/L1S1KS1RL/3G5 w 3P 68", "action": 49, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 24, 31, 33, 40, 43, 49, 59, 68, 79, 105, 114, 133, 158, 168, 177, 186, 221, 286, 294, 296, 330, 331, 339, 348, 357, 359, 440, 448, 512, 521, 565, 628, 629, 672, 735, 796], "sfen_after": "ln2g3l/1ksg3b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S1KS1RL/3G5 b 3Pp 69"}
{"sfen_before": "ln2g3l/1ksg3b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S1KS1RL/3G5 b 3Pp 69", "action": 449, "legal_actions": [3, 12, 14, 15, 22, 33, 40, 42, 49, 52, 57, 68, 75, 123, 132, 149, 186, 195, 213, 228, 268, 295, 305, 368, 391, 422, 449, 530, 556, 557, 566, 593, 602, 611, 619, 796, 1648, 1649, 1650, 1652, 1653, 1655], "sfen_after": "ln2g3l/1ksg3b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S2S1RL/3GK4 w 3Pp 70"}
{"sfen_before": "ln2g3l/1ksg3b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S2S1RL/3GK4 w 3Pp 70", "action": 133, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 24, 31, 33, 40, 43, 48, 59, 68, 79, 105, 114, 133, 158, 168, 177, 186, 221, 286, 294, 296, 330, 331, 339, 348, 357, 359, 440, 448, 512, 521, 565, 628, 629, 672, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1691, 1695, 1697, 1699], "sfen_after": "ln6l/1ksg1g1b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S2S1RL/3GK4 b 3Pp 71"}
{"sfen_before": "ln6l/1ksg1g1b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1S2S1RL/3GK4 b 3Pp 71", "action": 133, "legal_actions": [3, 12, 14, 15, 22, 33, 40, 43, 49, 52, 57, 68, 75, 123, 133, 149, 186, 205, 213, 228, 268, 305, 359, 391, 422, 556, 557, 566, 593, 619, 796, 1649, 1650, 1652, 1653, 1655], "sfen_after": "ln6l/1ksg1g1b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1SK1S1RL/3G5 w 3Pp 72"}
{"sfen_before": "ln6l/1ksg1g1b1/4r1psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1SK1S1RL/3G5 w 3Pp 72", "action": 348, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 24, 31, 33, 40, 48, 51, 59, 68, 79, 105, 114, 158, 168, 177, 186, 221, 286, 294, 304, 330, 331, 339, 348, 357, 367, 440, 448, 449, 458, 512, 521, 565, 628, 629, 672, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1691, 1695, 1697, 1699], "sfen_after": "ln6l/1ksg1g1b1/2r3psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1SK1S1RL/3G5 b 3Pp 73"}
{"sfen_before": "ln6l/1ksg1g1b1/2r3psn/2ppp4/GpP2p1PP/P2PP1P2/NPB5N/L1SK1S1RL/3G5 b 3Pp 73", "action": 68, "legal_actions": [3, 12, 14, 15, 22, 33, 40, 49, 51, 57, 68, 75, 123, 149, 186, 204, 205, 213, 228, 268, 305, 367, 368, 391, 422, 530, 548, 556, 557, 566, 593, 611, 796, 1649, 1650, 1652, 1653, 1655], "sfen_after": "ln6l/1ksg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1B5N/L1SK1S1RL/3G5 w 3Pp 74"}
{"sfen_before": "ln6l/1ksg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1B5N/L1SK1S1RL/3G5 w 3Pp 74", "action": 512, "legal_actions": [4, 5, 6, 7, 12, 15, 22, 31, 33, 40, 48, 51, 59, 68, 79, 114, 123, 158, 168, 177, 204, 221, 276, 285, 286, 294, 304, 330, 331, 339, 367, 440, 458, 512, 521, 565, 628, 629, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1B5N/L1SK1S1RL/3G5 b 3Pp 75"}
{"sfen_before": "lnk5l/2sg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1B5N/L1SK1S1RL/3G5 b 3Pp 75", "action": 213, "legal_actions": [3, 12, 14, 15, 22, 33, 40, 49, 51, 57, 67, 75, 123, 150, 186, 204, 205, 213, 228, 268, 305, 367, 368, 391, 422, 530, 548, 556, 557, 566, 593, 611, 796, 1649, 1650, 1652, 1653, 1655], "sfen_after": "lnk5l/2sg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 w 3Pp 76"}
{"sfen_before": "lnk5l/2sg1g1b1/2r3psn/2ppp4/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 w 3Pp 76", "action": 221, "legal_actions": [4, 5, 6, 7, 12, 22, 31, 33, 40, 48, 51, 59, 68, 79, 114, 123, 158, 177, 178, 204, 221, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 565, 628, 629, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r3p1n/2ppp1s2/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 b 3Pp 77"}
{"sfen_before": "lnk5l/2sg1g1b1/2r3p1n/2ppp1s2/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 b 3Pp 77", "action": 1649, "legal_actions": [3, 12, 14, 15, 22, 33, 40, 49, 57, 67, 75, 123, 140, 142, 186, 204, 205, 228, 268, 304, 305, 367, 368, 391, 422, 530, 547, 548, 556, 566, 593, 610, 611, 796, 1649, 1650, 1652, 1653, 1655], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/2ppp1s2/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 w 2Pp 78"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/2ppp1s2/GpP2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 w 2Pp 78", "action": 22, "legal_actions": [4, 5, 6, 7, 12, 22, 31, 33, 40, 48, 51, 58, 79, 114, 123, 148, 177, 178, 204, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1689, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3pp1s2/Gpp2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 b 2P2p 79"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3pp1s2/Gpp2p1PP/PP1PP1P2/N1BS4N/L2K1S1RL/3G5 b 2P2p 79", "action": 367, "legal_actions": [3, 12, 14, 15, 22, 28, 33, 40, 49, 67, 75, 123, 140, 142, 186, 204, 205, 228, 268, 304, 305, 367, 368, 391, 422, 530, 547, 548, 556, 566, 593, 610, 611, 796, 838, 1677, 1679, 1681, 1682], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3pp1s2/Gpp2p1PP/PP1PP1P2/N1BS4N/L3KS1RL/3G5 w 2P2p 80"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3pp1s2/Gpp2p1PP/PP1PP1P2/N1BS4N/L3KS1RL/3G5 w 2P2p 80", "action": 40, "legal_actions": [4, 5, 6, 7, 12, 21, 23, 31, 33, 40, 48, 51, 58, 79, 114, 123, 148, 177, 178, 204, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1689, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1pp1PP/PP1PP1P2/N1BS4N/L3KS1RL/3G5 b 2P2p 81"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1pp1PP/PP1PP1P2/N1BS4N/L3KS1RL/3G5 b 2P2p 81", "action": 611, "legal_actions": [3, 12, 14, 15, 22, 28, 33, 40, 42, 49, 52, 67, 75, 123, 140, 142, 186, 195, 228, 268, 295, 305, 368, 391, 422, 449, 530, 547, 556, 566, 593, 602, 611, 619, 796, 838, 1677, 1679, 1681, 1682], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1pp1PP/PP1PP1P2/N2S4N/L3KS1RL/3GB4 w 2P2p 82"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1pp1PP/PP1PP1P2/N2S4N/L3KS1RL/3GB4 w 2P2p 82", "action": 48, "legal_actions": [4, 5, 6, 7, 12, 21, 23, 31, 33, 39, 48, 51, 58, 79, 114, 123, 148, 177, 178, 204, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 1625, 1626, 1627, 1685, 1686, 1688, 1689, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpP2/N2S4N/L3KS1RL/3GB4 b 2P2p 83"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpP2/N2S4N/L3KS1RL/3GB4 b 2P2p 83", "action": 1682, "legal_actions": [3, 12, 14, 15, 22, 28, 33, 40, 42, 49, 52, 67, 75, 123, 133, 140, 141, 142, 186, 228, 268, 295, 305, 391, 422, 547, 593, 602, 796, 838, 1677, 1679, 1680, 1681, 1682], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpP2/N2S4N/L3KS1RL/2PGB4 w P2p 84"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpP2/N2S4N/L3KS1RL/2PGB4 w P2p 84", "action": 1686, "legal_actions": [4, 5, 6, 7, 12, 21, 23, 31, 33, 39, 47, 51, 58, 79, 114, 123, 148, 177, 178, 204, 211, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 857, 1625, 1626, 1627, 1685, 1686, 1688, 1689, 1691, 1695, 1697, 1699], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpPp1/N2S4N/L3KS1RL/2PGB4 b Pp 85"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p2PP/PP1PPpPp1/N2S4N/L3KS1RL/2PGB4 b Pp 85", "action": 22, "legal_actions": [3, 12, 14, 15, 22, 28, 33, 40, 42, 49, 52, 61, 67, 75, 123, 133, 140, 141, 142, 186, 228, 268, 295, 391, 422, 547, 593, 602, 796, 838], "sfen_after": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S4N/L3KS1RL/2PGB4 w Pp 86"}
{"sfen_before": "lnk5l/2sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S4N/L3KS1RL/2PGB4 w Pp 86", "action": 1627, "legal_actions": [4, 5, 6, 7, 12, 21, 23, 31, 33, 39, 47, 51, 58, 65, 79, 114, 123, 148, 177, 178, 204, 211, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 857, 875, 1625, 1626, 1627, 1695, 1697, 1699], "sfen_after": "lnk5l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S4N/L3KS1RL/2PGB4 b P 87"}
{"sfen_before": "lnk5l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S4N/L3KS1RL/2PGB4 b P 87", "action": 33, "legal_actions": [3, 12, 14, 15, 21, 28, 33, 40, 42, 49, 52, 61, 67, 75, 123, 133, 140, 141, 142, 186, 228, 268, 295, 391, 422, 547, 593, 602, 796, 838], "sfen_after": "lnk5l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S1S2N/L3K2RL/2PGB4 w P 88"}
{"sfen_before": "lnk5l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S1S2N/L3K2RL/2PGB4 w P 88", "action": 278, "legal_actions": [6, 12, 21, 23, 31, 33, 39, 47, 51, 58, 65, 79, 114, 123, 148, 177, 178, 204, 211, 276, 278, 285, 286, 294, 304, 330, 339, 367, 440, 458, 521, 555, 618, 629, 735, 796, 857, 875], "sfen_after": "ln1k4l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S1S2N/L3K2RL/2PGB4 b P 89"}
{"sfen_before": "ln1k4l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N2S1S2N/L3K2RL/2PGB4 b P 89", "action": 547, "legal_actions": [3, 12, 14, 15, 21, 28, 32, 40, 42, 49, 52, 61, 67, 75, 133, 140, 141, 142, 176, 185, 186, 196, 228, 268, 277, 295, 358, 391, 422, 547, 592, 602, 796, 838], "sfen_after": "ln1k4l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1K2RL/2PGB4 w P 90"}
{"sfen_before": "ln1k4l/p1sg1g1b1/2r2Pp1n/3p2s2/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1K2RL/2PGB4 w P 90", "action": 618, "legal_actions": [6, 12, 21, 23, 31, 33, 39, 47, 51, 58, 65, 79, 114, 123, 124, 148, 177, 204, 211, 276, 285, 286, 287, 294, 304, 330, 339, 350, 367, 458, 555, 618, 629, 735, 796, 857, 875], "sfen_after": "ln1k4l/p1sg1g1b1/2r2sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1K2RL/2PGB4 b Pp 91"}
{"sfen_before": "ln1k4l/p1sg1g1b1/2r2sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1K2RL/2PGB4 b Pp 91", "action": 196, "legal_actions": [3, 12, 14, 15, 21, 32, 40, 42, 49, 52, 60, 67, 75, 132, 133, 141, 150, 176, 185, 186, 196, 213, 228, 268, 277, 295, 358, 391, 422, 557, 592, 602, 796, 1650, 1651, 1654, 1655], "sfen_after": "ln1k4l/p1sg1g1b1/2r2sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1KB1RL/2PG5 w Pp 92"}
{"sfen_before": "ln1k4l/p1sg1g1b1/2r2sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1KB1RL/2PG5 w Pp 92", "action": 33, "legal_actions": [6, 12, 21, 23, 31, 33, 39, 47, 50, 59, 65, 79, 114, 123, 124, 140, 177, 203, 204, 276, 285, 286, 287, 304, 330, 339, 350, 367, 458, 547, 610, 629, 735, 796, 857, 875, 1695, 1697, 1699], "sfen_after": "ln1k4l/p1s2g1b1/2rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1KB1RL/2PG5 b Pp 93"}
{"sfen_before": "ln1k4l/p1s2g1b1/2rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S2N/L1S1KB1RL/2PG5 b Pp 93", "action": 15, "legal_actions": [3, 12, 14, 15, 21, 32, 40, 42, 49, 52, 60, 67, 75, 123, 132, 150, 176, 185, 186, 213, 228, 268, 295, 368, 391, 422, 449, 530, 557, 592, 593, 602, 796, 1650, 1651, 1655], "sfen_after": "ln1k4l/p1s2g1b1/2rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S1RN/L1S1KB2L/2PG5 w Pp 94"}
{"sfen_before": "ln1k4l/p1s2g1b1/2rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S1RN/L1S1KB2L/2PG5 w Pp 94", "action": 735, "legal_actions": [6, 12, 21, 23, 31, 34, 39, 47, 50, 59, 65, 79, 122, 124, 140, 177, 185, 203, 204, 285, 287, 304, 330, 339, 350, 367, 439, 458, 547, 610, 629, 735, 796, 857, 875, 1695, 1697, 1699], "sfen_after": "l2k4l/p1s2g1b1/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S1RN/L1S1KB2L/2PG5 b Pp 95"}
{"sfen_before": "l2k4l/p1s2g1b1/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N4S1RN/L1S1KB2L/2PG5 b Pp 95", "action": 213, "legal_actions": [3, 12, 14, 21, 32, 40, 42, 49, 52, 60, 67, 75, 123, 132, 150, 176, 185, 186, 213, 228, 267, 295, 368, 391, 421, 422, 449, 530, 557, 592, 593, 602, 796, 1650, 1651, 1655], "sfen_after": "l2k4l/p1s2g1b1/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w Pp 96"}
{"sfen_before": "l2k4l/p1s2g1b1/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w Pp 96", "action": 629, "legal_actions": [12, 21, 23, 31, 34, 39, 47, 50, 59, 65, 79, 122, 124, 140, 177, 185, 203, 204, 285, 287, 304, 339, 350, 367, 439, 458, 547, 584, 610, 629, 796, 857, 875, 1695, 1697, 1699], "sfen_after": "l2k2b1l/p1s2g3/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 b Pp 97"}
{"sfen_before": "l2k2b1l/p1s2g3/n1rg1sp1n/3p5/Gpp1p1PPP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 b Pp 97", "action": 21, "legal_actions": [3, 12, 14, 21, 32, 40, 42, 49, 52, 61, 67, 75, 123, 140, 142, 176, 185, 186, 228, 267, 295, 368, 391, 421, 422, 449, 530, 547, 592, 593, 602, 796, 1650, 1651, 1655], "sfen_after": "l2k2b1l/p1s2g3/n1rg1sp1n/3p2P2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w Pp 98"}
{"sfen_before": "l2k2b1l/p1s2g3/n1rg1sp1n/3p2P2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w Pp 98", "action": 350, "legal_actions": [12, 21, 23, 31, 34, 39, 47, 50, 59, 65, 79, 122, 124, 140, 151, 177, 185, 203, 204, 285, 287, 304, 339, 350, 367, 439, 458, 547, 584, 610, 796, 857, 875, 1695, 1697, 1699], "sfen_after": "l1k3b1l/p1s2g3/n1rg1sp1n/3p2P2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 b Pp 99"}
{"sfen_before": "l1k3b1l/p1s2g3/n1rg1sp1n/3p2P2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 b Pp 99", "action": 1650, "legal_actions": [3, 12, 14, 20, 32, 40, 42, 49, 52, 61, 67, 75, 123, 140, 142, 176, 185, 186, 228, 267, 295, 368, 391, 421, 422, 449, 530, 547, 592, 593, 602, 796, 830, 1650, 1651, 1655], "sfen_after": "l1k3b1l/p1s2g3/n1rg1sp1n/3p1PP2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w p 100"}
{"sfen_before": "l1k3b1l/p1s2g3/n1rg1sp1n/3p1PP2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 w p 100", "action": 439, "legal_actions": [12, 21, 23, 31, 39, 47, 50, 59, 65, 79, 115, 122, 140, 151, 177, 178, 185, 203, 204, 278, 285, 304, 339, 341, 367, 439, 458, 521, 547, 584, 610, 796, 857, 875, 1695, 1697, 1699], "sfen_after": "l1k3b1l/p1sg1g3/n1r2sp1n/3p1PP2/Gpp1p2PP/PP1PPp1p1/N2S1S1RN/L3KB2L/2PG5 b p 101"}"""

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