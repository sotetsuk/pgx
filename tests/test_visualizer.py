import os

import jax

from pgx.visualizer import Visualizer

key = jax.random.PRNGKey(0)
vis = Visualizer()


def test_animal_shogi():
    from pgx.animal_shogi import init

    state = init(key)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_backgammon():
    from pgx.backgammon import init

    state = init(key)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_chess():
    from pgx.chess import init

    state = init()
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_bridge():
    from pgx.contractbridgebidding import init

    _, state = init()
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_go():
    from pgx.go import init

    state = init(key, size=19)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_shogi():
    from pgx.shogi import init

    state = init(key)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_sparrowmahjong():
    from pgx.suzume_jong import init

    _, state = init(key)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")


def test_tic_tac_toe():
    from pgx.tic_tac_toe import init

    state = init(key)
    vis.save_svg(state, "tests/temp.svg")
    os.remove("tests/temp.svg")
