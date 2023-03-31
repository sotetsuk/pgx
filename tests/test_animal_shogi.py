import jax
import jax.numpy as jnp
from pgx._animal_shogi import AnimalShogi, _step_move, State, Action


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

    assert not Action._from_label(3 * 12 + 6).is_drop
    assert Action._from_label(3 * 12 + 6).from_ == 6
    assert Action._from_label(3 * 12 + 6).to == 5
    assert Action._from_label(3 * 12 + 6).drop_piece == -1
    state = step(state, 3 * 12 + 6)  # Up PAWN
    visualize(state, "tests/assets/animal_shogi/test_step_001.svg")
    assert not state.terminated
    assert state.turn == 1

    state = step(state, 0 * 12 + 11)  # Right Up Bishop
    visualize(state, "tests/assets/animal_shogi/test_step_002.svg")
    assert not state.terminated
    assert state.turn == 0

    assert Action._from_label(8 * 12 + 6).is_drop
    assert Action._from_label(8 * 12 + 6).from_ == -1
    assert Action._from_label(8 * 12 + 6).to == 6
    assert Action._from_label(8 * 12 + 6).drop_piece == 0

    state = step(state, 8 * 12 + 6)  # Drop PAWN to 6
    visualize(state, "tests/assets/animal_shogi/test_step_003.svg")
    assert not state.terminated
    assert state.turn == 1


