import jax

import pgx
from pgx.validator import validate_init, validate_observe, validate_step


def test_init():
    init, _, _ = pgx.make("tic_tac_toe/v0")
    validate_init(jax.jit(init))


def test_step():
    init, step, _ = pgx.make("tic_tac_toe/v0")
    validate_step(jax.jit(init), jax.jit(step))


def test_obsereve():
    init, step, observe = pgx.make("tic_tac_toe/v0")
    validate_observe(jax.jit(init), jax.jit(step), jax.jit(observe))
