import pgx
from pgx.validator import validate_init


def test_init():
    init, _, _ = pgx.make("tic_tac_toe/v0")
    validate_init(init)
