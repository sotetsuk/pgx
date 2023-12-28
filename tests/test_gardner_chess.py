import jax
import jax.numpy as jnp
import pgx

from pgx.gardner_chess import State, Action, GardnerChess, _zobrist_hash, QUEEN, EMPTY, ROOK, PAWN, KNIGHT
from pgx.experimental.utils import act_randomly
pgx.set_visualization_config(color_theme="dark")


env = GardnerChess()
init = jax.jit(env.init)
step = jax.jit(env.step)


def p(s: str, b=False):
    """
    >>> p("e1")
    20
    >>> p("e1", b=True)
    24
    """
    x = "abcde".index(s[0])
    offset = int(s[1]) - 1 if not b else 5 - int(s[1])
    return x * 5 + offset

def test_zobrist_hash():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    state = init(subkey)
    assert (state._zobrist_hash == jax.jit(_zobrist_hash)(state)).all()
    # for i in range(5):
    prev_hash = state._zobrist_hash
    while not state.terminated:
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state.legal_action_mask)
        state = step(state, action)
        assert (state._zobrist_hash == jax.jit(_zobrist_hash)(state)).all()
        assert not (state._zobrist_hash == prev_hash).all()
        prev_hash = state._zobrist_hash


def test_action():
    state = State._from_fen("k4/5/5/1Q3/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_001.svg")
    action = Action._from_label(jnp.int32(306))
    assert action.from_ == p("b2")
    assert action.to == p("b1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(309))
    assert action.from_ == p("b2")
    assert action.to == p("b5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(314))
    assert action.from_ == p("b2")
    assert action.to == p("a2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(317))
    assert action.from_ == p("b2")
    assert action.to == p("e2")
    assert action.underpromotion == -1
    # fail
    action = Action._from_label(jnp.int32(322))
    assert action.from_ == p("b2")
    assert action.to == p("a1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(325))
    assert action.from_ == p("b2")
    assert action.to == p("e5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(330))
    assert action.from_ == p("b2")
    assert action.to == p("a3")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(331))
    assert action.from_ == p("b2")
    assert action.to == p("c1")
    assert action.underpromotion == -1
    # knight moves
    # fail
    state = State._from_fen("k4/5/2N2/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_002.svg")
    action = Action._from_label(jnp.int32(629))
    assert action.from_ == p("c3")
    assert action.to == p("a2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(630))
    assert action.from_ == p("c3")
    assert action.to == p("a4")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(631))
    assert action.from_ == p("c3")
    assert action.to == p("b1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(632))
    assert action.from_ == p("c3")
    assert action.to == p("b5")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(633))
    assert action.from_ == p("c3")
    assert action.to == p("e2")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(634))
    assert action.from_ == p("c3")
    assert action.to == p("e4")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(635))
    assert action.from_ == p("c3")
    assert action.to == p("d1")
    assert action.underpromotion == -1
    action = Action._from_label(jnp.int32(636))
    assert action.from_ == p("c3")
    assert action.to == p("d5")
    assert action.underpromotion == -1
    # underpromotion
    state = State._from_fen("r1r1k/1P3/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_003.svg")
    action = Action._from_label(jnp.int32(392))
    assert action.from_ == p("b4")
    assert action.to == p("b5")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(393))
    assert action.from_ == p("b4")
    assert action.to == p("c5")
    assert action.underpromotion == 0  # rook
    action = Action._from_label(jnp.int32(394))
    assert action.from_ == p("b4")
    assert action.to == p("a5")
    assert action.underpromotion == 0  # rook
    # black turn
    state = State._from_fen("k4/3q1/5/5/4K b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/action_004.svg")
    # 上（上下はそのまま）
    action = Action._from_label(jnp.int32(797))
    assert action.from_ == p("d4", True)
    assert action.to == p("d3", True)
    assert action.underpromotion == -1
    # 左（左右は鏡写し）
    action = Action._from_label(jnp.int32(805))
    assert action.from_ == p("d4", True)
    assert action.to == p("e4", True)
    assert action.underpromotion == -1


def test_observe():
    # position
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (5, 5, 115)
    expected_wpawn1 = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    expected_bpawn1 = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected_wpawn1).all()
    assert state.observation[4, 1, 1] == 1.
    assert state.observation[4, 2, 2] == 1.
    assert state.observation[4, 0, 3] == 1.
    assert state.observation[4, 3, 4] == 1.
    assert state.observation[4, 4, 5] == 1.
    assert (state.observation[:, :, 6] == expected_bpawn1).all()
    assert state.observation[0, 1, 7] == 1.
    assert state.observation[0, 2, 8] == 1.
    assert state.observation[0, 0, 9] == 1.
    assert state.observation[0, 3, 10] == 1.
    assert state.observation[0, 4, 11] == 1.
    assert (state.observation[:, :, 12] == 1).all()
    assert (state.observation[:, :, 13] == 0).all()
    assert state._turn == 0
    assert (state.observation[:, :, 112] == 0).all()

    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_001.svg")
    assert state._turn == 1
    assert (state.observation[:, :, 112] == 1).all()
    assert (state.observation[:, :, 0] == expected_wpawn1).all()

    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_002.svg")
    expected_wpawn2 = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    expected_wpawn3 = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected_wpawn3).all()
    assert state.observation[4, 1, 1] == 1.
    assert state.observation[4, 2, 2] == 1.
    assert state.observation[4, 0, 3] == 1.
    assert state.observation[4, 3, 4] == 1.
    assert state.observation[4, 4, 5] == 1.
    assert (state.observation[:, :, 6] == expected_bpawn1).all()
    assert state.observation[2, 0, 7] == 1.
    assert state.observation[0, 2, 8] == 1.
    assert state.observation[0, 0, 9] == 1.
    assert state.observation[0, 3, 10] == 1.
    assert state.observation[0, 4, 11] == 1.
    # history
    # 14~27
    assert (state.observation[:, :, 14] == expected_wpawn2).all()
    assert state.observation[4, 1, 15] == 1.
    assert state.observation[4, 2, 16] == 1.
    assert state.observation[4, 0, 17] == 1.
    assert state.observation[4, 3, 18] == 1.
    assert state.observation[4, 4, 19] == 1.
    assert (state.observation[:, :, 20] == expected_bpawn1).all()
    assert state.observation[0, 1, 21] == 1.
    assert state.observation[0, 2, 22] == 1.
    assert state.observation[0, 0, 23] == 1.
    assert state.observation[0, 3, 24] == 1.
    assert state.observation[0, 4, 25] == 1.
    # 28~41
    assert (state.observation[:, :, 28] == expected_wpawn1).all()
    assert state.observation[4, 1, 29] == 1.
    assert state.observation[4, 2, 30] == 1.
    assert state.observation[4, 0, 31] == 1.
    assert state.observation[4, 3, 32] == 1.
    assert state.observation[4, 4, 33] == 1.
    assert (state.observation[:, :, 34] == expected_bpawn1).all()
    assert state.observation[0, 1, 35] == 1.
    assert state.observation[0, 2, 36] == 1.
    assert state.observation[0, 0, 37] == 1.
    assert state.observation[0, 3, 38] == 1.
    assert state.observation[0, 4, 39] == 1.

    # repetition
    state = State._from_fen("k3r/5/5/5/K4 b - - 0 1")
    state = step(state, 21)
    state.save_svg("tests/assets/gardner_chess/observe_003.svg")
    state = step(state, 21)
    state.save_svg("tests/assets/gardner_chess/observe_004.svg")
    state = step(state, 265)
    state.save_svg("tests/assets/gardner_chess/observe_005.svg")
    state = step(state, 265)
    state.save_svg("tests/assets/gardner_chess/observe_006.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 1 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 5 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 6 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 6 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 7 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 7 + 13] == 0.).all()
    state = step(state, 21)
    state.save_svg("tests/assets/gardner_chess/observe_007.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 6 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 7 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 7 + 13] == 0.).all()
    state = step(state, 21)
    state.save_svg("tests/assets/gardner_chess/observe_008.svg")
    assert (state.observation[:, :, 12] == 0.).all()
    assert (state.observation[:, :, 13] == 1.).all()
    state = step(state, 265)
    state.save_svg("tests/assets/gardner_chess/observe_009.svg")
    assert (state.observation[:, :, 12] == 0.).all()
    assert (state.observation[:, :, 13] == 1.).all()
    state = step(state, 265)
    state.save_svg("tests/assets/gardner_chess/observe_010.svg")
    assert (state.observation[:, :, 12] == 0.).all()
    assert (state.observation[:, :, 13] == 1.).all()
    assert state.terminated

    # color
    state = State._from_fen("k3r/5/5/5/K3R w - - 23 20")
    state.save_svg("tests/assets/gardner_chess/observe_011.svg")
    state = step(state, 1000)
    state.save_svg("tests/assets/gardner_chess/observe_012.svg")
    assert state.observation[0, 0, 112] == 1.
    state = step(state, 1000)
    state.save_svg("tests/assets/gardner_chess/observe_013.svg")
    assert state.observation[0, 0, 112] == 0.
    state = step(state, 755)
    state.save_svg("tests/assets/gardner_chess/observe_014.svg")
    assert state.observation[0, 0, 112] == 1.
    state = step(state, 755)
    state.save_svg("tests/assets/gardner_chess/observe_015.svg")
    assert state.observation[0, 0, 112] == 0.
    state = step(state, 510)
    state.save_svg("tests/assets/gardner_chess/observe_016.svg")
    assert state.observation[0, 0, 112] == 1.
    state = step(state, 510)
    state.save_svg("tests/assets/gardner_chess/observe_017.svg")
    assert state.observation[0, 0, 112] == 0.
    state = step(state, 13)
    state.save_svg("tests/assets/gardner_chess/observe_018.svg")
    # check rook history
    assert state.observation[4, 1, 3] == 1.
    assert state.observation[4, 1, 17] == 1.
    assert state.observation[4, 2, 31] == 1.
    assert state.observation[4, 2, 45] == 1.
    assert state.observation[4, 3, 59] == 1.
    assert state.observation[4, 3, 73] == 1.
    assert state.observation[4, 4, 87] == 1.
    assert state.observation[4, 4, 101] == 1.
    # color, move_counts
    assert state.observation[0, 0, 112] == 1.
    assert state.observation[0, 0, 113] > 0.
    assert state.observation[0, 0, 114] > 0.

    # from_fen with black turn
    state = State._from_fen("rnbqk/ppppp/P4/1PPPP/RNBQK b - - 0 1")
    # same with "tests/assets/gardner_chess/observe_001.svg"
    print(state.observation[:, :, 0])
    assert (state.observation[:, :, 0] == expected_wpawn1).all()
    assert (state.observation[:, :, 112] == 1).all()
    state = step(state, 1042)
    state.save_svg("tests/assets/gardner_chess/observe_019.svg")
    assert (state.observation[:, :, 0] == expected_wpawn2).all()
    assert (state.observation[:, :, 112] == 0).all()

    # check repetition observation
    state = init(jax.random.PRNGKey(0))
    state = step(state, jnp.int32(293))
    state.save_svg("tests/assets/gardner_chess/observe_020.svg")
    state = step(state, jnp.int32(289))
    state.save_svg("tests/assets/gardner_chess/observe_021.svg")
    state = step(state, jnp.int32(631))
    state.save_svg("tests/assets/gardner_chess/observe_022.svg")
    state = step(state, jnp.int32(145))
    state.save_svg("tests/assets/gardner_chess/observe_023.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 1 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(293))
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(289))
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(631))
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(307))
    state.save_svg("tests/assets/gardner_chess/observe_024.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 0 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 3 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 6 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 7 + 13] == 1.).all()  # rep

    state = init(jax.random.PRNGKey(0))
    state = step(state, jnp.int32(293))
    # "tests/assets/gardner_chess/observe_020.svg"
    state = step(state, jnp.int32(289))
    # "tests/assets/gardner_chess/observe_021.svg"
    state = step(state, jnp.int32(631))
    # "tests/assets/gardner_chess/observe_022.svg"
    state = step(state, jnp.int32(21))
    state.save_svg("tests/assets/gardner_chess/observe_025.svg")
    state = step(state, jnp.int32(293))
    state.save_svg("tests/assets/gardner_chess/observe_026.svg")
    state = step(state, jnp.int32(265))
    state.save_svg("tests/assets/gardner_chess/observe_027.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 1 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    state = step(state, jnp.int32(631))
    state.save_svg("tests/assets/gardner_chess/observe_028.svg")
    assert (state.observation[:, :, 14 * 0 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 0 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 1 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 2 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 2 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 3 + 12] == 1.).all()
    assert (state.observation[:, :, 14 * 3 + 13] == 0.).all()
    assert (state.observation[:, :, 14 * 4 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 4 + 13] == 1.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 12] == 0.).all()  # rep
    assert (state.observation[:, :, 14 * 5 + 13] == 1.).all()  # rep

def test_step():
    # normal step
    # queen
    state = State._from_fen("k4/5/5/1Q3/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_001.svg")
    assert state._board[p("b1")] == EMPTY
    assert state._board[p("e5")] == EMPTY
    state1 = step(state, jnp.int32(306)) # b2 -> b1
    state1.save_svg("tests/assets/gardner_chess/step_002.svg")
    assert state1._board[p("b1", True)] == -QUEEN
    state2 = step(state, jnp.int32(325)) # b2 -> e5
    state2.save_svg("tests/assets/gardner_chess/step_003.svg")
    assert state2._board[p("e5", True)] == -QUEEN

    # knight
    state = State._from_fen("k1b2/5/2N2/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_004.svg")
    assert state._board[p("b1")] == EMPTY
    assert state._board[p("e4")] == EMPTY
    state1 = step(state, jnp.int32(631)) # c3 -> b1
    state1.save_svg("tests/assets/gardner_chess/step_005.svg")
    assert state1._board[p("b1", True)] == -KNIGHT
    state2 = step(state, jnp.int32(634)) # c3 -> e4
    state2.save_svg("tests/assets/gardner_chess/step_006.svg")
    assert state2._board[p("e4", True)] == -KNIGHT

    # promotion
    state = State._from_fen("r1r1k/1P3/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/step_007.svg")
    assert state._board[p("b5")] == EMPTY
    assert state._board[p("c5")] == -ROOK
    # underpromotion
    next_state = step(state, jnp.int32(392)) # b4 -> b5 (underpromotion:Rook)
    next_state.save_svg("tests/assets/gardner_chess/step_008.svg")
    assert next_state._board[p("b5", True)] == -ROOK
    # promotion to queen
    next_state = step(state, jnp.int32(421)) # b4 -> c5 (promotion:Queen)
    next_state.save_svg("tests/assets/gardner_chess/step_008.svg")
    assert next_state._board[p("c5", True)] == -QUEEN

    # steps
    state = init(jax.random.PRNGKey(0))
    steps = [1042, 552, 993, 289, 797, 1065, 1065, 771]
    for i in range(8):
        num = 9 + i
        if num == 9:
            s = "09"
        else:
            s = str(num)
        svg_name = "tests/assets/gardner_chess/step_0" + s + ".svg"
        state = step(state, steps[i])
        state.save_svg(svg_name)
        if i != 7:
            assert not state.terminated
        else:
            assert state.terminated


def test_legal_action_mask():
    # init board
    state = State()
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_001.svg")
    assert state.legal_action_mask.sum() == 7

    # pawn (blocked)
    state = State._from_fen("5/5/4k/4P/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_002.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 1

    # pawn capture
    state = State._from_fen("4k/5/3r1/4P/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_003.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 2

    # promotion (white)
    state = State._from_fen("2r1k/1P3/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_004.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 11

    # promotion (black, pin)
    state = State._from_fen("4k/5/5/1p3/BB2K b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_005.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 6

    # check
    state = State._from_fen("4k/5/2b2/5/KRR2 w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_006.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 3

    # pinned
    state = State._from_fen("4k/5/r1b2/BP3/KBr2 w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_007.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 1

    # pinned(same line)
    state = State._from_fen("k3b/5/5/1Q3/K4 w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_008.svg")
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5

    # remote check
    state = State._from_fen("5/R1B1k/1b3/5/K4 w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_009.svg")
    state = step(state, jnp.int32(673))  # c4 -> b5
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_010.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 5

    # double check
    state = State._from_fen("5/R1B1k/1b3/5/K4 w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_011.svg")
    state = step(state, jnp.int32(666))  # c4 -> d5
    state.save_svg("tests/assets/gardner_chess/legal_action_mask_012.svg")
    print(state._to_fen())
    print(jnp.nonzero(state.legal_action_mask))
    assert state.legal_action_mask.sum() == 4


def test_terminal():
    # checkmate (white win)
    state = State._from_fen("4k/4R/2N2/5/K4 b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_001.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert state.rewards[state.current_player] == -1
    assert state.rewards[1 - state.current_player] == 1.

    # stalemate
    state = State._from_fen("k4/5/1Q3/K4/5 b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_002.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # 50-move draw rule
    state = State._from_fen("2k2/p1p1p/PpPpP/1P1P1/4K b - - 99 50")
    state.save_svg("tests/assets/gardner_chess/terminal_003.svg")
    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/terminal_004.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # insufficient pieces
    # K vs K
    state = State._from_fen("k4/5/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_005.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K+B vs K
    state = State._from_fen("k4/5/5/5/3BK w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_006.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K vs K+B
    state = State._from_fen("kb3/5/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_007.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K+N vs K
    state = State._from_fen("k4/5/5/5/3NK w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_008.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K vs K+N
    state = State._from_fen("kn3/5/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_009.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in Black tile)
    state = State._from_fen("k1b1b/5/5/5/B1B1K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_010.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in White tile)
    state = State._from_fen("kb1B1/B1b2/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_011.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # insufficient cases by underpromotion
    # K+B vs K
    state = State._from_fen("k4/4P/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_012.svg")
    state = step(state, jnp.int32(1130))
    state.save_svg("tests/assets/gardner_chess/terminal_013.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+N vs K
    state = State._from_fen("k4/4P/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_014.svg")
    state = step(state, jnp.int32(1133))
    state.save_svg("tests/assets/gardner_chess/terminal_015.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B(Bishop in Black tile)
    state = State._from_fen("k1b2/4P/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_016.svg")
    state = step(state, jnp.int32(1130))
    state.save_svg("tests/assets/gardner_chess/terminal_017.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B vs K+B (Bishop in White tile)
    state = State._from_fen("kb3/3P1/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_018.svg")
    state = step(state, jnp.int32(885))
    state.save_svg("tests/assets/gardner_chess/terminal_019.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B*2 vs K(Bishop in Black tile)
    state = State._from_fen("k1B2/4P/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_020.svg")
    state = step(state, jnp.int32(1130))
    state.save_svg("tests/assets/gardner_chess/terminal_021.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # K+B*2 vs K (Bishop in White tile)
    state = State._from_fen("kB3/3P1/5/5/4K w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_022.svg")
    state = step(state, jnp.int32(885))
    state.save_svg("tests/assets/gardner_chess/terminal_023.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # stalemate with pin
    state = State._from_fen("kbR2/pn3/P1B2/5/4K b - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_024.svg")
    print(state._to_fen())
    assert state.terminated
    assert state.current_player == 0
    assert (state.rewards == 0.0).all()

    # rep termination
    state = State._from_fen("k3r/5/5/5/K3R w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/terminal_025.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(13))
    state.save_svg("tests/assets/gardner_chess/terminal_026.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    state.save_svg("tests/assets/gardner_chess/terminal_026.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    state.save_svg("tests/assets/gardner_chess/terminal_027.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    state.save_svg("tests/assets/gardner_chess/terminal_028.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    state.save_svg("tests/assets/gardner_chess/terminal_029.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1000))
    state.save_svg("tests/assets/gardner_chess/terminal_030.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1000))
    state.save_svg("tests/assets/gardner_chess/terminal_031.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/terminal_032.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/terminal_033.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()

    # rep termination2
    # 途中まで上と同じ
    state = State._from_fen("k3r/5/5/5/K3R w - - 0 1")
    print(state._to_fen())
    state = step(state, jnp.int32(13))
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    print(state._to_fen())
    state = step(state, jnp.int32(999))
    state.save_svg("tests/assets/gardner_chess/terminal_031.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(999))
    state.save_svg("tests/assets/gardner_chess/terminal_033.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(511))
    state.save_svg("tests/assets/gardner_chess/terminal_034.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(511))
    state.save_svg("tests/assets/gardner_chess/terminal_035.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/terminal_036.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/terminal_037.svg")
    print(state._to_fen())
    assert state.terminated
    assert (state.rewards == 0.0).all()


def test_buggy_samples():
    # half-movecount reset when underpromotion happens
    state = State._from_fen("5/P3k/K4/5/5 w - - 2 68")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_001.svg")
    state = step(state, 160)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_002.svg")
    assert state._to_fen() == "Q4/4k/K4/5/5 b - - 0 68"

    # pinned pawn moves
    state = State._from_fen("4k/5/r1q1n/3PK/5 b - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_003.svg")
    state = step(state, 111)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_004.svg")
    expected_legal_actions = [1041]
    assert state.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # lift pin
    state = State._from_fen("4k/3b1/5/1PP2/K4 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_005.svg")
    # self lift
    state1 = step(state, 552)
    state1.save_svg("tests/assets/gardner_chess/buggy_samples_006.svg")
    state1 = step(state1, 1000)
    state1.save_svg("tests/assets/gardner_chess/buggy_samples_007.svg")
    expected_legal_actions = [13, 21, 307, 601, 617]
    assert state1.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state1.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"
    # lift by king move
    state2 = step(state, 21)
    state2.save_svg("tests/assets/gardner_chess/buggy_samples_008.svg")
    state2 = step(state2, 1000)
    state2.save_svg("tests/assets/gardner_chess/buggy_samples_009.svg")
    expected_legal_actions = [265, 260, 281, 307, 601]
    assert state2.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state2.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"
    # lift by opponent
    state = State._from_fen("k4/2pb1/5/1P3/K4 b - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_010.svg")
    state1 = step(state, 813)
    state1.save_svg("tests/assets/gardner_chess/buggy_samples_011.svg")
    expected_legal_actions = [13, 21, 307]
    assert state1.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state1.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"
    state2 = step(state, 552)
    state2.save_svg("tests/assets/gardner_chess/buggy_samples_012.svg")
    expected_legal_actions = [13, 21, 307, 323]
    assert state2.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state2.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # pin by promotion
    state = State._from_fen("3bk/2P2/5/5/K4 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_013.svg")
    state = step(state, 650)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_014.svg")
    expected_legal_actions = [993]
    assert state.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # not the same turn repetition
    state = State._from_fen("k3r/5/5/5/K3R w - - 0 1")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_015.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(13))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_016.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_017.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(993))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_018.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_019.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1041))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_020.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(999))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_021.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(1000))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_022.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(707))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_023.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_024.svg")
    print(state._to_fen())
    state = step(state, jnp.int32(756))
    state.save_svg("tests/assets/gardner_chess/buggy_samples_025.svg")
    print(state._to_fen())
    # assert not state.terminated

    # pin
    state = State._from_fen("k3b/5/2b2/1P3/K4 b - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_026.svg")
    state = step(state, 617)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_027.svg")
    expected_legal_actions = [13, 21]
    assert state.legal_action_mask.sum() == len(
        expected_legal_actions), f"\nactual:{jnp.nonzero(state.legal_action_mask)[0]}\nexpected\n{expected_legal_actions}"

    # stalemate by promotion
    state = State._from_fen("5/2P1k/5/4P/K4 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_028.svg")
    state = step(state, 650)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_029.svg")
    assert state.terminated

    # mate by pinned piece
    state = State._from_fen("k1b1R/4Q/K4/5/5 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_030.svg")
    state = step(state, 1145)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_031.svg")
    assert state.terminated
    assert state.current_player == 1
    assert state.rewards[state.current_player] == -1
    assert state.rewards[1 - state.current_player] == 1.

    # pin(cannot promotion)
    state = State._from_fen("5/r2PK/5/5/k4 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_032.svg")
    assert state.legal_action_mask.sum() == 4

    # stalemate by promotion
    state = State._from_fen("3bk/P4/3Q1/5/K4 w - - 0 0")
    state.save_svg("tests/assets/gardner_chess/buggy_samples_033.svg")
    state = step(state, 147)
    state.save_svg("tests/assets/gardner_chess/buggy_samples_034.svg")
    assert state.terminated


def test_api():
    import pgx
    env = pgx.make("gardner_chess")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
