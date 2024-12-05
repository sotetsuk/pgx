import json
import jax
import jax.numpy as jnp

from pgx.shogi import Shogi, State
from pgx._src.games.shogi import Action, HORSE, PAWN, DRAGON
from pgx.experimental.shogi import from_sfen, to_sfen

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
    state.save_svg(fname, color_theme="dark")


def update_board(state, piece_board, hand=None):
    state = state.replace(piece_board=piece_board)
    if hand is not None:
        state = state.replace(hand=hand)
    return state


def test_init():
    key = jax.random.PRNGKey(0)
    s = init(key)
    assert jnp.unique(s._x.board).shape[0] == 1 + 8 + 8
    assert s.legal_action_mask.sum() != 0
    legal_actions = jnp.int32([5, 7, 14, 23, 25, 32, 34, 41, 43, 50, 52, 59, 61, 68, 77, 79, 115, 124, 133, 142, 187, 196, 205, 214, 268, 277, 286, 295, 304, 331])
    assert (jnp.nonzero(s.legal_action_mask)[0] == legal_actions).all(), jnp.nonzero(s.legal_action_mask)[0]


def test_is_legal_drop():
    # 打ち歩詰
    # 避けられるし金でも取れる
    sfen = "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_001.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 片側に避けられるので打ち歩詰でない
    sfen = "lns1kpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_002.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 両側に避けられないので打ち歩詰
    sfen = "lnspkpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_003.svg")
    assert not state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 金で取れるので打ち歩詰でない
    sfen = "lnsgkpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_004.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]


def test_buggy_samples():
    # 歩以外の持ち駒に対しての二歩判定回避
    sfen = "9/9/9/9/9/9/PPPPPPPPP/9/9 b NLP 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_001.svg")

    # 歩は二歩になるので打てない
    assert (~state.legal_action_mask[20 * 81:21 * 81]).all()
    # 香車は2列目には打てるが、1列目と7列目（歩がいる）には打てない
    assert (state.legal_action_mask[21 * 81 + 1:22 * 81:9]).all()
    assert (~state.legal_action_mask[21 * 81:22 * 81:9]).all()
    assert (~state.legal_action_mask[21 * 81 + 6:22 * 81:9]).all()
    # 桂馬は1,2列目に打てないが3列目には打てる
    assert (~state.legal_action_mask[22 * 81:23 * 81:9]).all()
    assert (~state.legal_action_mask[22 * 81 + 1:23 * 81:9]).all()
    assert (state.legal_action_mask[21 * 81 + 2:22 * 81:9]).all()

    # 成駒のpromotion判定
    sfen = "9/2+B1G1+P2/9/9/9/9/9/9/9 b - 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_002.svg")
    # promotionは生成されてたらダメ
    assert (state.legal_action_mask[10 * 81:]).sum() == 0

    # 角は成れないはず
    sfen = "l+B6l/6k2/3pg2P1/p6p1/1pP1pB2p/2p3n2/P+r1GP3P/4KS1+s1/LNG5L b RGN2sn6p 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_003.svg")
    assert ~state.legal_action_mask[13 * 81 + 72]  # = 1125, promote + left (91角成）

    # #375
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/buggy_samples_004.svg")
    assert (s.legal_action_mask.sum() == len([43, 52, 68, 196, 222, 295, 789, 1996, 2004, 2012])).all(), jnp.nonzero(s.legal_action_mask)[0]
    assert (jnp.nonzero(s.legal_action_mask)[0] == jnp.int32([43, 52, 68, 196, 222, 295, 789, 1996, 2004, 2012])).all()

    # #602
    sfen = "9/4R4/9/9/9/9/9/9/9 b 2r2b4g3s4n4l17p 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_005.svg")
    dlshogi_action = 846
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_006.svg")
    sfen = "4+R4/9/9/9/9/9/9/9/9 w 2r2b4g3s4n4l7p 1"
    expected_state = from_sfen(sfen)
    visualize(expected_state, "tests/assets/shogi/buggy_samples_006.svg")
    assert (state._x.board == expected_state._x.board).all()

    # #603
    state = from_sfen("8k/9/9/5b3/9/3B5/9/9/K8 b 2r4g4s4n4l18p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_007.svg")
    dlshogi_action = 202
    a = Action._from_dlshogi_action(state._x, dlshogi_action)
    assert a.from_ == xy2i(6, 6)

    # #610
    state = from_sfen("+PsGg1p2+P/+B1+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 b np 1")
    visualize(state, "tests/assets/shogi/buggy_samples_008.svg")
    dlshogi_action = 225
    a = Action._from_dlshogi_action(state._x, dlshogi_action)
    assert a.from_ == xy2i(9, 2)
    assert a.piece == HORSE
    state = step(state, dlshogi_action)
    expected_state = from_sfen("+P+BGg1p2+P/2+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 w Snp 1")
    assert (state._x.board == expected_state._x.board).all()

    # #613
    state = from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1g2PL/3KPR2S/1R1G1NG2 b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_009.svg")
    dlshogi_action = 42
    a = Action._from_dlshogi_action(state._x, dlshogi_action)
    assert a.from_ == xy2i(5, 8)
    assert a.piece == PAWN
    state = step(state, dlshogi_action)
    expected_state = from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1P2PL/3K1R2S/1R1G1NG2 w GP4p 1")
    assert (state._x.board == expected_state._x.board).all()

    # #618
    state = from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg+s3L1/p2R2p2/P+n+B+p+ng1+s+p w P 1")
    visualize(state, "tests/assets/shogi/buggy_samples_010.svg")
    dlshogi_action = 28
    a = Action._from_dlshogi_action(state._x, dlshogi_action)
    assert a.from_ == xy2i(4, 3)
    state = step(state, dlshogi_action)
    expected_state = from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg4L1/p2+s2p2/P+n+B+p+ng1+s+p b Pr 1")
    assert (state._x.board == expected_state._x.board).all()

    # 629
    state = from_sfen("1ns6/+S1p+Ng1p1l/+P2pg1nNS/4k2G1/2L2R2s/p1G2+BPR1/3Pp2+p1/1+p3B1P1/1LPK2+l1+p b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_011.svg")
    dlshogi_action = 1660  # 歩打
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_012.svg")
    assert not state.terminated  # 打ち歩詰でない

    # 打ち歩詰ではないが2歩 # 639
    sfen = "+P2G1p2+P/1+N2+Pbk1p/3p3l+L/1gp3s1L/2SPG2p1/NK1SL3N/1p5RP/+p+r+p2+np+s1/b1P+p2+p2 w Gp 35"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_013.svg")
    assert not state.legal_action_mask[20 * 81 + xy2i(2, 5)]

    # Hand crafted tests #685
    # double check
    sfen = "8k/9/9/9/9/8r/8s/9/7GK w - 1"
    state = from_sfen(sfen)
    assert int(state.legal_action_mask.sum()) == 21
    dl_action = 226
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_014.svg")
    assert int(state.legal_action_mask.sum()) == 1
    # discovered check with pin
    sfen = "8k/9/9/9/9/5b3/6r2/9/7GK w - 1"
    state = from_sfen(sfen)
    assert int(state.legal_action_mask.sum()) == 49
    dl_action = 54
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_015.svg")
    assert int(state.legal_action_mask.sum()) == 1
    # discovered check
    sfen = "k8/8g/9/4r1b1K/P8/9/9/9/9 w - 1"
    state = from_sfen(sfen)
    assert int(state.legal_action_mask.sum()) == 37
    dl_action = 156
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_016.svg")
    assert int(state.legal_action_mask.sum()) == 1
    # catch pieces
    sfen = "k1b1r4/5B2g/4p4/9/9/9/8K/4L4/9 b - 1"
    state = from_sfen(sfen)
    assert int(state.legal_action_mask.sum()) == 23
    dl_action = 38
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_017.svg")
    assert int(state.legal_action_mask.sum()) == 18
    dl_action = 42
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_018.svg")
    assert int(state.legal_action_mask.sum()) == 85
    dl_action = 81 * 6 + 38
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_019.svg")
    assert int(state.legal_action_mask.sum()) == 78
    dl_action = 81 + 42
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_020.svg")
    assert int(state.legal_action_mask.sum()) == 10
    # double pin
    sfen = "8k/9/9/9/9/5b3/6r2/7P1/7GK w - 1"
    state = from_sfen(sfen)
    dl_action = 54
    state = step(state, dl_action)
    visualize(state, "tests/assets/shogi/buggy_samples_021.svg")
    assert int(state.legal_action_mask.sum()) == 2
    # drop pawn mate
    sfen = "8k/9/7L1/7N1/9/9/9/9/8K b P 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_022.svg")
    assert int(state.legal_action_mask.sum()) == 76
    # move pawn mate(legal)
    sfen = "8k/9/7LP/7N1/9/9/9/9/8K b - 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_023.svg")
    assert int(state.legal_action_mask.sum()) == 10
    # pinned same line
    sfen = "8k/9/9/9/4b4/9/6B2/9/8K b - 1"
    state = from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_024.svg")
    assert int(state.legal_action_mask.sum()) == 6


def test_step():
    with open("tests/assets/shogi/random_play.json") as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                assert False, line
            sfen = d["sfen_before"]
            state = from_sfen(sfen)
            expected_legal_actions = d["legal_actions"]
            legal_actions = jnp.nonzero(state.legal_action_mask)[0]
            ok = legal_actions.shape[0] == len(expected_legal_actions)
            if not ok:
                visualize(state, "tests/assets/shogi/failed.svg")
                for a in legal_actions:
                    if a not in expected_legal_actions:
                        print(Action._from_dlshogi_action(state._x, a))
                for a in expected_legal_actions:
                    if a not in legal_actions:
                        print(Action._from_dlshogi_action(state._x, a))
                assert False, f"{legal_actions.shape[0]} != {len(expected_legal_actions)}, {sfen}"
            action = int(d["action"])
            state = step(state, action)
            sfen = d["sfen_after"]
            assert to_sfen(state) == sfen


def test_observe():
    key = jax.random.PRNGKey(0)
    s: State = init(key)
    obs = observe(s, s.current_player)

    assert obs.shape == (9, 9, 119)

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])
    assert (obs[:, :, PAWN] == expected).all()  # 0

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])
    assert (obs[:, :, DRAGON + 1] == expected).all()  # 14

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [1.,0.,1.,0.,0.,0.,1.,1.,1.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [1.,0.,1.,1.,1.,1.,1.,1.,0.]])
    assert (obs[:, :, 28] == expected).all()  # 利きの数1以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,0.,1.,0.,0.,0.,0.,0.,1.],
                          [0.,1.,1.,1.,1.,1.,1.,0.,1.],
                          [0.,0.,1.,0.,1.,0.,0.,0.,0.]])
    assert (obs[:, :, 29] == expected).all()  # 利きの数2以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,1.,1.,1.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])

    assert (obs[:, :, 30] == expected).all()  # 利きの数3以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])

    assert (obs[:, :, 31] == expected).all()

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])

    assert (obs[:, :, 45] == expected).all()

    expected = jnp.bool_([[0.,1.,1.,1.,1.,1.,1.,0.,1.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [1.,1.,1.,0.,0.,0.,1.,0.,1.],
                          [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.]])

    assert (obs[:, :, 59] == expected).all()

    # 駒打ち
    sfen = "1ns4nl/1r4k2/2p1gp3/1p1pp3p/l8/2P2PP2/1PNPP3P/2G2S3/2S1KG2L b BGS3Prbnl2p 1"
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_001.svg")
    obs = observe(s, s.current_player)

    filled = [0, 1, 2, 16, 20, 24, 28, 29, 36, 40, 52, 54]
    for i in range(56):
        if i in filled:
            assert obs[:, :, 62 + i].all()
        else:
            assert (~obs[:, :, 62 + i]).all()

    # 王手
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"  # 先手番
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_002.svg")
    obs = observe(s, s.current_player)
    assert (~obs[:, :, -1]).all()

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"  # 後手番
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_003.svg")
    obs = observe(s, s.current_player)
    assert obs[:, :, -1].all()

    # TODO: player_id != current_player

def test_sfen():
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_001.svg")
    assert to_sfen(s) == sfen

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_002.svg")
    assert to_sfen(s) == sfen


def test_repetition():
    # without check
    sfen = "l2+B2knl/1r4g2/2n1gpsp1/p1pps1p1p/1p5P1/P1P1SPP1P/1PSPP4/2G2G3/LNK4RL b Pbn 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_001.svg")
    dlshogi_action1 = 243 + 54 # 7一馬(6一)
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_002.svg")
    dlshogi_action2 = 243 + 43 # 5二飛(8二)
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_003.svg")
    dlshogi_action3 = 324 + 45 # 6一馬(7一)
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_004.svg")
    dlshogi_action4 = 324 + 16 # 8二飛(5二)
    s = step(s, dlshogi_action4)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_005.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 3 time
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time(draw)
    #assert s.terminated
    #assert s.rewards[0] == s.rewards[1]

    # with check repetition(not continuous check)
    sfen = "ln7/1ksR5/ppp6/9/9/9/9/9/8K b Ss 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_006.svg")
    dlshogi_action1 = 1863 + 47 # 6三銀打
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_007.svg")
    dlshogi_action2 = 1863 + 35 # 6一銀打
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_008.svg")
    dlshogi_action3 = 891 + 55 # 7二銀成(6三)
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_009.svg")
    dlshogi_action4 = 162 + 25 # 7二銀(6一)
    s = step(s, dlshogi_action4)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_010.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 3 time
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time(draw)
    #assert s.terminated
    #assert s.rewards[0] == s.rewards[1]

    # with continuous check repetition
    sfen = "8l/6+P2/6+Rpk/8p/9/7S1/9/9/8K b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_011.svg")
    dlshogi_action1 = 162 + 10  # 2二龍(3一)
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_012.svg")
    dlshogi_action2 = 162 + 68  # 2四王(1三)
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_013.svg")
    dlshogi_action3 = 486 + 20  # 3一龍(2二)
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_014.svg")
    dlshogi_action4 = 486 + 78  # 1三王(2四)
    s = step(s, dlshogi_action4)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_015.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 3 time
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time(white win)
    # assert s.terminated
    #assert s.rewards[s.current_player] == -1
    #assert s.rewards[1 - s.current_player] == 1.

    # different hands
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 8G 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_016.svg")
    dlshogi_action1 = 2106 + 37  # 5二金打
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_017.svg")
    dlshogi_action2 = 243 + 43  # 52飛(82)
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_018.svg")
    dlshogi_action3 = 2106 + 64  # 82金打
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_019.svg")
    dlshogi_action4 = 324 + 16  # 82飛(52)
    s = step(s, dlshogi_action4)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_020.svg")
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_021.svg")
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_022.svg")
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_023.svg")
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_024.svg")
    # 3 time
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_025.svg")
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_026.svg")
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_027.svg")
    s = step(s, dlshogi_action4)
    # 4 time(not repetition)
    assert not s.terminated

    # different turn
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_028.svg")
    dlshogi_action1 = 243 + 34  # 48飛(28)
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_029.svg")
    dlshogi_action2 = 324 + 7  # 92飛(82)
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_030.svg")
    dlshogi_action3 = 324 + 25  # 38飛(48)
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_031.svg")
    dlshogi_action4 = 243 + 16  # 82飛(92)
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_032.svg")
    dlshogi_action5 = 324 + 16  # 38飛(28)
    s = step(s, dlshogi_action5)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_033.svg")
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_034.svg")
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_035.svg")
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_036.svg")
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_037.svg")
    s = step(s, dlshogi_action5)
    visualize(s, "tests/assets/shogi/repetition_038.svg")
    # 3 time
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_039.svg")
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_040.svg")
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_041.svg")
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_042.svg")
    s = step(s, dlshogi_action5)
    # 4 time (not repetition)
    assert not s.terminated

    sfen = "4k4/3r5/9/4p4/9/3P5/9/4R4/4K4 b Pp 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_043.svg")
    dlshogi_action1 = 1620 + 40  # 55歩
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_044.svg")
    dlshogi_action2 = 0 + 40
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_045.svg")
    dlshogi_action3 = 0 + 40
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_046.svg")
    dlshogi_action4 = 1620 + 41
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_047.svg")
    dlshogi_action5 = 405 + 43
    s = step(s, dlshogi_action5)
    visualize(s, "tests/assets/shogi/repetition_048.svg")
    # 2 time
    dlshogi_action6 = 1620 + 31
    s = step(s, dlshogi_action6)
    visualize(s, "tests/assets/shogi/repetition_049.svg")
    dlshogi_action7 = 0 + 49
    s = step(s, dlshogi_action7)
    visualize(s, "tests/assets/shogi/repetition_050.svg")
    dlshogi_action8 = 0 + 31
    s = step(s, dlshogi_action8)
    visualize(s, "tests/assets/shogi/repetition_051.svg")
    dlshogi_action9 = 1620 + 50
    s = step(s, dlshogi_action9)
    visualize(s, "tests/assets/shogi/repetition_052.svg")
    dlshogi_action10 = 405 + 34
    s = step(s, dlshogi_action10)
    visualize(s, "tests/assets/shogi/repetition_053.svg")
    # 3 time
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    s = step(s, dlshogi_action5)
    # 4 time
    assert not s.terminated
    s = step(s, dlshogi_action6)
    s = step(s, dlshogi_action7)
    s = step(s, dlshogi_action8)
    s = step(s, dlshogi_action9)
    s = step(s, dlshogi_action10)
    assert not s.terminated
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    s = step(s, dlshogi_action5)
    assert not s.terminated
    s = step(s, dlshogi_action6)
    s = step(s, dlshogi_action7)
    s = step(s, dlshogi_action8)
    s = step(s, dlshogi_action9)
    s = step(s, dlshogi_action10)
    # assert s.terminated

    sfen = "4k4/2G3+RG1/9/9/9/9/9/9/4K4 b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_054.svg")
    dlshogi_action1 = 243 + 46  # 55歩
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_055.svg")
    dlshogi_action2 = 243 + 53
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_056.svg")
    dlshogi_action3 = 324 + 19
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_057.svg")
    dlshogi_action4 = 324 + 44
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_058.svg")
    # 2 time
    assert not s.terminated
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 3 time
    assert not s.terminated
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time
    # assert s.terminated
    # assert s.rewards[s.current_player] == -1
    # assert s.rewards[1 - s.current_player] == 1.

    sfen = "3sk4/4s4/5S3/4R4/9/9/9/9/8K b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_059.svg")
    dlshogi_action1 = 891 + 37
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_060.svg")
    dlshogi_action2 = 81 + 43
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_061.svg")
    dlshogi_action3 = 1863 + 29
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_062.svg")
    dlshogi_action4 = 1863 + 35
    s = step(s, dlshogi_action4)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_063.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 3 time
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time(draw)
    # assert s.terminated
    # assert s.rewards[0] == s.rewards[1]

    sfen = "r3k4/9/9/9/9/9/9/9/4K3R b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_064.svg")
    dlshogi_action1 = 7
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_065.svg")
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_066.svg")
    dlshogi_action2 = 6
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_067.svg")
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_068.svg")
    dlshogi_action3 = 405 + 8
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_069.svg")
    s = step(s, dlshogi_action3)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_070.svg")
    assert not s.terminated
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action3)
    # 3 time
    assert not s.terminated
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action3)
    # 4 time
    # assert s.terminated

    sfen = "1r2k4/9/9/9/9/9/9/9/4K3R b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_071.svg")
    dlshogi_action1 = 810 + 1
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_072.svg")
    dlshogi_action2 = 810 + 10
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_073.svg")
    dlshogi_action3 = 405 + 8
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_074.svg")
    dlshogi_action4 = 405 + 17
    s = step(s, dlshogi_action4)
    # 2 time(駒成り含)
    visualize(s, "tests/assets/shogi/repetition_075.svg")
    dlshogi_action5 = 1
    s = step(s, dlshogi_action5)
    visualize(s, "tests/assets/shogi/repetition_076.svg")
    dlshogi_action6 = 10
    s = step(s, dlshogi_action6)
    # 2 time
    visualize(s, "tests/assets/shogi/repetition_077.svg")
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    s = step(s, dlshogi_action5)
    s = step(s, dlshogi_action6)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    assert not s.terminated
    s = step(s, dlshogi_action5)
    s = step(s, dlshogi_action6)
    # 4 time
    # assert s.terminated

    sfen = "9/9/9/9/9/9/K8/1G7/k8 b - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_078.svg")
    dlshogi_action1 = 243 + 79
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_079.svg")
    dlshogi_action2 = 243 + 9
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_080.svg")
    dlshogi_action3 = 324 + 70
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_081.svg")
    dlshogi_action4 = 324
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_082.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time
    # assert s.terminated
    # assert s.rewards[s.current_player] == -1
    # assert s.rewards[1 - s.current_player] == 1.

    sfen = "8K/7g1/8k/9/9/9/9/9/9 w - 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/repetition_083.svg")
    dlshogi_action1 = 243 + 79
    s = step(s, dlshogi_action1)
    visualize(s, "tests/assets/shogi/repetition_084.svg")
    dlshogi_action2 = 243 + 9
    s = step(s, dlshogi_action2)
    visualize(s, "tests/assets/shogi/repetition_085.svg")
    dlshogi_action3 = 324 + 70
    s = step(s, dlshogi_action3)
    visualize(s, "tests/assets/shogi/repetition_086.svg")
    dlshogi_action4 = 324
    s = step(s, dlshogi_action4)
    visualize(s, "tests/assets/shogi/repetition_087.svg")
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    s = step(s, dlshogi_action1)
    s = step(s, dlshogi_action2)
    s = step(s, dlshogi_action3)
    s = step(s, dlshogi_action4)
    # 4 time
    # assert s.terminated
    # assert s.rewards[s.current_player] == -1
    # assert s.rewards[1 - s.current_player] == 1.


def test_api():
    import pgx
    env = pgx.make("shogi")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
