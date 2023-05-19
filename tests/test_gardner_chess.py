import jax
import jax.numpy as jnp
import pgx
from pgx.gardner_chess import State, Action, GardnerChess, QUEEN, EMPTY, ROOK, PAWN, KNIGHT

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
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (5, 5, 115)
    expected = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected).all()
    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_001.svg")
    state = step(state, jnp.nonzero(state.legal_action_mask, size=1)[0][0])
    state.save_svg("tests/assets/gardner_chess/observe_002.svg")
    expected = jnp.float32(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0.]]
    )
    assert (state.observation[:, :, 0] == expected).all()


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
    state.save_svg("tests/assets/chess/legal_action_mask_005.svg")
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
    # FEN is from https://www.chess.com/terms/fen-chess#halfmove-clock
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
    state.save_svg("tests/assets/chess/terminal_013.svg")
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



def test_api():
    import pgx
    env = pgx.make("gardner_chess")
    pgx.v1_api_test(env, 3)
