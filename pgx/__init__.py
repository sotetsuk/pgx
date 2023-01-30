from pgx.core import EnvId, State

__all__ = ["State", "EnvId"]


def make(env_id: EnvId):
    if env_id == "tic_tac_toe/v0":
        from pgx.tic_tac_toe import TicTacToe

        return TicTacToe()
