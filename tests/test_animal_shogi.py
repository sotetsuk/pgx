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
    assert state.observation.shape == (4, 3, 22)

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
    assert (state.observation[:, :, 1] == expected).all()

    # opp king
    expected = jnp.bool_(
        [[False, True,  False],
         [False, False, False],
         [False, False, False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 8] == expected).all()

    state = State(
        _board=jnp.int8([
             8, -1, -1, -1,
            -1, -1, -1,  3,
            -1, -1, -1,  0]),
        _hand=jnp.int8([[2, 0, 0], [0, 1, 0]])
    )
    state = state.replace(observation=_observe(state, state.current_player))
    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, False, False],
         [True, False, False]]
    )
    assert (state.observation[:, :, 0] == expected).all()
    # hand
    expected = jnp.bool_([True , True,
                          False, False,
                          False, False,
                          False, False,
                          True , False,
                          False, False])
    assert (state.observation[0, 0, 10:] == expected).all()


def test_repetition():
    state = init(jax.random.PRNGKey(0))
    # first
    visualize(state, "tests/assets/animal_shogi/test_repetition_000.svg")
    assert not state.terminated
    assert state._turn == 0

    state = step(state, 3 * 12 + 3)  # Up Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_002.svg")
    assert not state.terminated
    assert state._turn == 1

    state = step(state, 3 * 12 + 3)  # Up Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_003.svg")
    assert not state.terminated
    assert state._turn == 0

    state = step(state, 4 * 12 + 2)  # Down Rook
    visualize(state, "tests/assets/animal_shogi/test_repetition_004.svg")
    assert not state.terminated
    assert state._turn == 1

    state = step(state, 4 * 12 + 2)  # Down Rook
    # second
    visualize(state, "tests/assets/animal_shogi/test_repetition_005.svg")
    assert not state.terminated
    assert state._turn == 0

    # same repetition
    state1 = step(state, 3 * 12 + 3)  # Up Rook
    assert not state.terminated
    state1 = step(state1, 3 * 12 + 3)  # Up Rook
    assert not state.terminated
    state1 = step(state1, 4 * 12 + 2)  # Down Rook
    assert not state.terminated
    # third
    state1 = step(state1, 4 * 12 + 2)  # Down Rook
    # three times
    # assert state1.terminated

    # different repetition
    state2 = step(state, 0 * 12 + 7)  # Right Up King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_006.svg")
    assert not state2.terminated
    assert state2._turn == 1
    state2 = step(state2, 0 * 12 + 7)  # Right Up King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_007.svg")
    assert not state2.terminated
    assert state2._turn == 1
    state2 = step(state2, 7 * 12 + 2)  # Left Down King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_008.svg")
    assert not state2.terminated
    assert state2._turn == 1
    state2 = step(state2, 7 * 12 + 2)  # Left Down King
    visualize(state2, "tests/assets/animal_shogi/test_repetition_009.svg")
    # third
    # assert state2.terminated

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
    # second
    assert not state.terminated
    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    assert not state.terminated
    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    assert not state.terminated
    state = step(state, 7 * 12 + 6)  # Left Down Bishop
    assert not state.terminated

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    # third
    # assert state.terminated



def test_api():
    import pgx
    env = pgx.make("animal_shogi")
    pgx.v1_api_test(env, 5)
