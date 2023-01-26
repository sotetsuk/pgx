import pgx.go
import pgx.minatar.breakout
import pgx.suzume_jong
import pgx.tic_tac_toe
from pgx.core import EnvId, State

__all__ = [State, EnvId]


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
