from typing import Literal, Union

import pgx.go
import pgx.minatar.breakout
import pgx.suzume_jong
import pgx.tic_tac_toe

EnvId = Literal[
    "tic_tac_toe/v0",
    "minatar/breakout/v0",
    "suzume_jong/v0",
    "go/v0",
]


State = Union[pgx.tic_tac_toe.State, pgx.suzume_jong.State]


def make(env_id: EnvId):
    if env_id == "tic_tac_toe/v0":
        return (
            pgx.tic_tac_toe.init,
            pgx.tic_tac_toe.step,
            pgx.tic_tac_toe.observe,
        )
    elif env_id == "minatar/breakout/v0":
        return (
            pgx.minatar.breakout.init,
            pgx.minatar.breakout.step,
            pgx.minatar.breakout.observe,
        )
    elif env_id == "suzume_jong/v0":
        return (
            pgx.suzume_jong.init,
            pgx.suzume_jong.step,
            pgx.suzume_jong.observe,
        )
    elif env_id == "go/v0":
        return (
            pgx.go.init,
            pgx.go.step,
            pgx.go.observe,
        )
