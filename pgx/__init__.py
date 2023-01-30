from pgx.core import EnvId, State

__all__ = ["State", "EnvId"]


def make(env_id: EnvId):
    if env_id == "tic_tac_toe/v0":
        import tic_tac_toe

        return tic_tac_toe.TicTacToe()
    elif env_id == "go/v0":
        import go

        return go.Go()
