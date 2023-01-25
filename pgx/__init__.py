from typing import Literal

EnvId = Literal[
    "tic_tac_toe/v0",
    "minatar/breakout/v0",
    "suzume_jong/v0",
    "go/v0",
]


def make(env_id: EnvId):
    if env_id == "tic_tac_toe/v0":
        from pgx.tic_tac_toe import init, observe, step

        return init, step, observe
    elif env_id == "minatar/breakout/v0":
        from pgx.minatar.breakout import init, observe, step

        return init, step, observe
    elif env_id == "suzume_jong/v0":
        from pgx.suzume_jong import init, observe, step

        return init, step, observe
    elif env_id == "go/v0":
        from pgx.go import init, observe, step

        return init, step, observe
