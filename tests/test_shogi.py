import json
import jax
import jax.numpy as jnp

from pgx.shogi import Shogi, State, Action, HORSE, PAWN, DRAGON


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
    assert (jnp.nonzero(s.legal_action_mask)[0] == legal_actions).all(), jnp.nonzero(s.legal_action_mask)[0]


def test_is_legal_drop():
    # 打ち歩詰
    # 避けられるし金でも取れる
    sfen = "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_001.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 片側に避けられるので打ち歩詰でない
    sfen = "lns1kpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_002.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 両側に避けられないので打ち歩詰
    sfen = "lnspkpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_003.svg")
    assert not state.legal_action_mask[20 * 81 + xy2i(5, 2)]

    # 金で取れるので打ち歩詰でない
    sfen = "lnsgkpsnl/7b1/ppppGpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b P 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/legal_drops_004.svg")
    assert state.legal_action_mask[20 * 81 + xy2i(5, 2)]


def test_buggy_samples():
    # 歩以外の持ち駒に対しての二歩判定回避
    sfen = "9/9/9/9/9/9/PPPPPPPPP/9/9 b NLP 1"
    state = State._from_sfen(sfen)
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
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_002.svg")
    # promotionは生成されてたらダメ
    assert (state.legal_action_mask[10 * 81:]).sum() == 0

    # 角は成れないはず
    sfen = "l+B6l/6k2/3pg2P1/p6p1/1pP1pB2p/2p3n2/P+r1GP3P/4KS1+s1/LNG5L b RGN2sn6p 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_003.svg")
    assert ~state.legal_action_mask[13 * 81 + 72]  # = 1125, promote + left (91角成）

    # #375
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/buggy_samples_004.svg")
    assert (jnp.nonzero(s.legal_action_mask)[0] == jnp.int32([43, 52, 68, 196, 222, 295, 789, 1996, 2004, 2012])).all()

    # #602
    sfen = "9/4R4/9/9/9/9/9/9/9 b 2r2b4g3s4n4l17p 1"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_005.svg")
    dlshogi_action = 846
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_006.svg")
    sfen = "4+R4/9/9/9/9/9/9/9/9 w 2r2b4g3s4n4l7p 1"
    expected_state = State._from_sfen(sfen)
    visualize(expected_state, "tests/assets/shogi/buggy_samples_006.svg")
    assert (state.piece_board == expected_state.piece_board).all()

    # #603
    state = State._from_sfen("8k/9/9/5b3/9/3B5/9/9/K8 b 2r4g4s4n4l18p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_007.svg")
    dlshogi_action = 202
    a = Action._from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(6, 6)

    # #610
    state = State._from_sfen("+PsGg1p2+P/+B1+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 b np 1")
    visualize(state, "tests/assets/shogi/buggy_samples_008.svg")
    dlshogi_action = 225
    a = Action._from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(9, 2)
    assert a.piece == HORSE
    state = step(state, dlshogi_action)
    expected_state = State._from_sfen("+P+BGg1p2+P/2+Pgp+N1sp/1+N5l1/P3kP1pL/3P1r3/B2KP3L/4L1SP+s/+r2+p2pgP/2P2+n+p2 w Snp 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # #613
    state = State._from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1g2PL/3KPR2S/1R1G1NG2 b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_009.svg")
    dlshogi_action = 42
    a = Action._from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(5, 8)
    assert a.piece == PAWN
    state = step(state, dlshogi_action)
    expected_state = State._from_sfen("1+N3s1n1/5k2l/l+P2g1bp1/2pP1p2p/p2ppNS2/LB6P/1pS1P2PL/3K1R2S/1R1G1NG2 w GP4p 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # #618
    state = State._from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg+s3L1/p2R2p2/P+n+B+p+ng1+s+p w P 1")
    visualize(state, "tests/assets/shogi/buggy_samples_010.svg")
    dlshogi_action = 28
    a = Action._from_dlshogi_action(state, dlshogi_action)
    assert a.from_ == xy2i(4, 3)
    state = step(state, dlshogi_action)
    expected_state = State._from_sfen("2+P+P2G1+S/1P2+P+P1+Pn/+S1GK2P2/1b2PP3/1nl4PP/3k2lRL/1pg4L1/p2+s2p2/P+n+B+p+ng1+s+p b Pr 1")
    assert (state.piece_board == expected_state.piece_board).all()

    # 629
    state = State._from_sfen("1ns6/+S1p+Ng1p1l/+P2pg1nNS/4k2G1/2L2R2s/p1G2+BPR1/3Pp2+p1/1+p3B1P1/1LPK2+l1+p b P4p 1")
    visualize(state, "tests/assets/shogi/buggy_samples_011.svg")
    dlshogi_action = 1660  # 歩打
    state = step(state, dlshogi_action)
    visualize(state, "tests/assets/shogi/buggy_samples_012.svg")
    assert not state.terminated  # 打ち歩詰でない

    # 打ち歩詰ではないが2歩 # 639
    sfen = "+P2G1p2+P/1+N2+Pbk1p/3p3l+L/1gp3s1L/2SPG2p1/NK1SL3N/1p5RP/+p+r+p2+np+s1/b1P+p2+p2 w Gp 35"
    state = State._from_sfen(sfen)
    visualize(state, "tests/assets/shogi/buggy_samples_013.svg")
    assert not state.legal_action_mask[20 * 81 + xy2i(2, 5)]


def test_step():
    with open("tests/assets/shogi/random_play.json") as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                assert False, line
            sfen = d["sfen_before"]
            state = State._from_sfen(sfen)
            expected_legal_actions = d["legal_actions"]
            legal_actions = jnp.nonzero(state.legal_action_mask)[0]
            ok = legal_actions.shape[0] == len(expected_legal_actions)
            if not ok:
                visualize(state, "tests/assets/shogi/failed.svg")
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


def test_observe():
    key = jax.random.PRNGKey(0)
    s: State = init(key)
    obs = observe(s, s.current_player)

    assert obs.shape == (119, 9, 9)

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[PAWN] == expected).all()  # 0

    expected = jnp.bool_([[0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,1.,0.,0.,0.]])
    assert (obs[DRAGON + 1] == expected).all()  # 14

    expected = jnp.bool_([[0.,0.,0.,0.,0.,1.,1.,1.,0.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,1.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,1.,1.,1.,1.]])
    assert (obs[28] == expected).all()  # 利きの数1以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,1.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,1.,1.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[29] == expected).all()  # 利きの数2以上

    expected = jnp.bool_([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.,0.,1.,0.,0.]])
    assert (obs[30] == expected).all()  # 利きの数3以上

    expected = jnp.bool_([[0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.,0.,0.,0.]])
    assert (obs[31] == expected).all()

    expected = jnp.bool_([[0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.,0.,0.,0.]])
    assert (obs[45] == expected).all()

    expected = jnp.bool_([[1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [0.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,0.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [1.,1.,1.,1.,0.,0.,0.,0.,0.],
                          [0.,1.,1.,1.,0.,0.,0.,0.,0.]])
    assert (obs[59] == expected).all()

    # 駒打ち
    sfen = "1ns4nl/1r4k2/2p1gp3/1p1pp3p/l8/2P2PP2/1PNPP3P/2G2S3/2S1KG2L b BGS3Prbnl2p 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_001.svg")
    obs = observe(s, s.current_player)

    filled = [0, 1, 2, 16, 20, 24, 28, 29, 36, 40, 52, 54]
    for i in range(56):
        if i in filled:
            assert obs[62 + i].all()
        else:
            assert (~obs[62 + i]).all()

    # 王手
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"  # 先手番
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_002.svg")
    obs = observe(s, s.current_player)
    assert (~obs[-1]).all()

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"  # 後手番
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/observe_003.svg")
    obs = observe(s, s.current_player)
    assert obs[-1].all()

    # TODO: player_id != current_player

def test_sfen():
    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL b b 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_001.svg")
    assert s._to_sfen() == sfen

    sfen = "lnsgkg1nl/1r5s1/pppppp1pp/6p2/8B/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 1"
    s = State._from_sfen(sfen)
    visualize(s, "tests/assets/shogi/sfen_002.svg")
    assert s._to_sfen() == sfen


def test_api():
    import pgx
    # env = pgx.make("shogi")
    env = Shogi(max_termination_steps=50)
    pgx.api_test(env, 5)