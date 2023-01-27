import jax

import pgx
from pgx.validator import validate


def test_api():
    init, step, observe = pgx.make("tic_tac_toe/v0")
    validate(jax.jit(init), jax.jit(step), jax.jit(observe))
