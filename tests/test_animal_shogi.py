import jax
import jax.numpy as jnp
from pgx._animal_shogi import AnimalShogi, _step_move, State, Action, _can_move


env = AnimalShogi()
init = jax.jit(env.init)
step = jax.jit(env.step)


def visualize(state, fname="tests/assets/animal_shogi/xxx.svg"):
    from pgx._visualizer import Visualizer
    v = Visualizer(color_theme="dark")
    v.save_svg(state, fname)


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
    assert state.observation.shape == (4, 3, 28)

    expected = jnp.bool_(
        [[False, False, False],
         [False, False, False],
         [False, True,  False],
         [False, False, False]]
    )
    assert (state.observation[:, :, 0] == expected).all()