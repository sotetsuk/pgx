import jax
import jax.numpy as jnp
from pgx.animal_shogi import AnimalShogi, _step_move, State, Action, _can_move, _legal_action_mask, _observe


env = AnimalShogi()
init = jax.jit(env.init)
step = jax.jit(env.step)


def visualize(state, fname="tests/assets/animal_shogi/xxx.svg"):
    state.save_svg(fname, color_theme="dark")


def test_step():
    state = init(jax.random.PRNGKey(0))
    visualize(state, "tests/assets/animal_shogi/test_step_000.svg")
    assert not state.terminated
    assert state._turn == 0

    state = step(state, 3 * 12 + 6)  # Up PAWN
    visualize(state, "tests/assets/animal_shogi/test_step_001.svg")
    assert not state.terminated
    assert state._turn == 1
    assert state._board[6] == 5
    assert state._hand[0, 0] == 0
    assert state._hand[1, 0] == 1
    assert state._hand.sum() == 1

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_step_002.svg")
    assert not state.terminated
    assert state._turn == 0
    assert state._board[5] == 6
    assert state._hand[0, 0] == 1
    assert state._hand[1, 0] == 1
    assert state._hand.sum() == 2

    state = step(state, 8 * 12 + 6)  # Drop PAWN to 6
    visualize(state, "tests/assets/animal_shogi/test_step_003.svg")
    assert not state.terminated
    assert state._turn == 1
    assert state._board[5] == 5
    assert state._hand[0, 0] == 1
    assert state._hand[1, 0] == 0
    assert state._hand.sum() == 1


def test_observe():
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (4, 3, 194)

    # my pawn
    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, True,  False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 0] == expected).all()

    # my bishop
    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, False, False],
         [True , False, False]]
    )
    print(state._board_history[0])
    assert (state.observation[:, :, 1] == expected).all()

    # opp king
    expected = jnp.bool_(
        [[False, True,  False],
         [False, False, False],
         [False, False, False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 8] == expected).all()

    state = init(jax.random.PRNGKey(0))
    state.save_svg("tests/assets/animal_shogi/test_obs_000.svg")
    state = step(state, 6 + 12 * 3)
    state.save_svg("tests/assets/animal_shogi/test_obs_001.svg")
    # my pawn
    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, False,  False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 0] == expected).all()
    # opp pawn
    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, True,  False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 5] == expected).all()
    assert (state.observation[:, :, 10 + 6] == 1).all()  # opp captured pawn
    state = step(state, 7 + 12 * 3)
    state.save_svg("tests/assets/animal_shogi/test_obs_002.svg")
    # opp king
    # opp pawn in hand
    # my pawn in hand @ 1-step before
    # opp pawn in hand @ 1-step before

def test_repetition():
    state = init(jax.random.PRNGKey(0))
    # first
    visualize(state, "tests/assets/animal_shogi/test_repetition_000.svg")
    assert not state.terminated
    assert state._turn == 0
    assert (state.observation[:, :, 22] == 1).all()  # rep = 0

    state = step(state, 3 * 12 + 3)  # Up Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_002.svg")
    assert not state.terminated
    assert state._turn == 1
    assert (state.observation[:, :, 22] == 1).all()  # rep = 0

    state = step(state, 3 * 12 + 3)  # Up Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_003.svg")
    assert not state.terminated
    assert state._turn == 0
    assert (state.observation[:, :, 22] == 1).all()  # rep = 0

    state = step(state, 4 * 12 + 2)  # Down Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_004.svg")
    assert not state.terminated
    assert state._turn == 1
    assert (state.observation[:, :, 22] == 1).all()  # rep = 0

    state = step(state, 4 * 12 + 2)  # Down Rook
    # second
    visualize(state, "tests/assets/animal_shogi/test_repetition_005.svg")
    assert not state.terminated
    assert (state.observation[:, :, 22] == 0).all()  # rep = 0
    assert (state.observation[:, :, 23] == 1).all()  # rep = 1
    assert state._turn == 0

    # same repetition
    state1 = step(state, 3 * 12 + 3)  # Up Rook
    assert not state1.terminated
    state1 = step(state1, 3 * 12 + 3)  # Up Rook
    assert not state1.terminated
    state1 = step(state1, 4 * 12 + 2)  # Down Rook
    assert not state1.terminated
    # third
    state1 = step(state1, 4 * 12 + 2)  # Down Rook
    # three times
    assert state1.terminated
    assert (state1.rewards == 0).all()

    # different repetition
    state2 = step(state, 0 * 12 + 7)  # Right Up King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_006.svg")
    assert not state2.terminated
    assert state2._turn == 1
    state2 = step(state2, 0 * 12 + 7)  # Right Up King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_007.svg")
    assert not state2.terminated
    assert state2._turn == 0
    state2 = step(state2, 7 * 12 + 2)  # Left Down King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_008.svg")
    assert not state2.terminated
    assert state2._turn == 1
    state2 = step(state2, 7 * 12 + 2)  # Left Down King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_009.svg")
    # third
    assert state2.terminated
    assert (state2.rewards == 0).all()

    # hand
    state = init(jax.random.PRNGKey(0))
    visualize(state, "tests/assets/animal_shogi/test_repetition_010.svg")
    assert not state.terminated
    assert state._turn == 0
    state = step(state, 3 * 12 + 6)  # Up PAWN
    visualize(state, "tests/assets/animal_shogi/test_repetition_011.svg")
    assert not state.terminated
    assert state._turn == 1

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    # first
    visualize(state, "tests/assets/animal_shogi/test_repetition_012.svg")
    assert not state.terminated
    assert state._turn == 0

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_013.svg")
    assert not state.terminated
    assert state._turn == 1

    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_014.svg")
    assert not state.terminated
    assert state._turn == 0

    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_015.svg")
    assert not state.terminated
    assert state._turn == 1

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_016.svg")
    # second
    assert not state.terminated
    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_017.svg")
    assert not state.terminated
    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_018.svg")
    assert not state.terminated
    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    visualize(state, "tests/assets/animal_shogi/test_repetition_019.svg")
    assert not state.terminated

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    # third
    assert state.terminated
    assert (state.rewards == 0).all()


def test_api():
    import pgx
    env = pgx.make("animal_shogi")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)


def test_buggy_samples():
    # https://github.com/sotetsuk/pgx/pull/1209
    state = init(jax.random.key(0))
    state = step(state, 3 * 12 +  6) # White: Up PAWN
    state = step(state, 0 * 12 + 11) # Black: Right Up Bishop
    state = step(state, 8 * 12 +  1) # White: Drop PAWN to 1
    state = step(state, 3 * 12 +  3) # Black: Up Rook
    state = step(state, 3 * 12 +  1) # White: Up PAWN (Promote to GOLD)
    state = step(state, 1 * 12 +  7) # Black: Right King
    DOWN_GOLD      = 4 * 12 +  0
    DOWN_LEFT_GOLD = 7 * 12 +  0
    LEFT_GOLD      = 6 * 12 +  0
    mask = state.legal_action_mask
    assert mask[DOWN_GOLD]
    assert not mask[DOWN_LEFT_GOLD]
    assert mask[LEFT_GOLD]

    # https://github.com/sotetsuk/pgx/pull/1218
    state = init(jax.random.key(0))
    state = step(state, 3 * 12 +  6) # White: Up PAWN
    state = step(state, 0 * 12 + 11) # Black: Right Up Bishop
    DROP_PAWN_TO_0 = 8 * 12 +  0
    assert state.legal_action_mask[DROP_PAWN_TO_0]
