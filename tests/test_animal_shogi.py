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
    assert state.turn == 0

    state = step(state, 3 * 12 + 6)  # Up PAWN
    visualize(state, "tests/assets/animal_shogi/test_step_001.svg")
    assert not state.terminated
    assert state.turn == 1
    assert state.board[6] == 5
    assert state.hand[0, 0] == 0
    assert state.hand[1, 0] == 1
    assert state.hand.sum() == 1

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_step_002.svg")
    assert not state.terminated
    assert state.turn == 0
    assert state.board[5] == 6
    assert state.hand[0, 0] == 1
    assert state.hand[1, 0] == 1
    assert state.hand.sum() == 2

    state = step(state, 8 * 12 + 6)  # Drop PAWN to 6
    visualize(state, "tests/assets/animal_shogi/test_step_003.svg")
    assert not state.terminated
    assert state.turn == 1
    assert state.board[5] == 5
    assert state.hand[0, 0] == 1
    assert state.hand[1, 0] == 0
    assert state.hand.sum() == 1


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
        board=jnp.int8([
             8, -1, -1, -1,
            -1, -1, -1,  3,
            -1, -1, -1,  0]),
        hand=jnp.int8([[2, 0, 0], [0, 1, 0]])
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


def test_api():
    import pgx
    env = pgx.make("animal_shogi")
    pgx.api_test(env, 5)
