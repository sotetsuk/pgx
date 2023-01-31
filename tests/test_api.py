import jax

import pgx
from pgx.validator import validate


def test_api():
    env = pgx.make("tic_tac_toe/v0")
    validate(env)
    env = pgx.make("minatar/asterix/v0")
    validate(env)
