import jax
import pgx
from pgx.utils import act_randomly
from pgx.tic_tac_toe import TicTacToe
from pgx.shogi import Shogi
from pgx.go import Go


def test_jit():
    N = 2
    for Env in [TicTacToe, Shogi, Go]:
        env = Env()
        init = jax.jit(jax.vmap(env.init))
        step = jax.jit(jax.vmap(env.step))
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, N)

        state = init(keys)
        assert state.curr_player.shape == (N,)
        assert (state.observation).sum() != 0

        action = act_randomly(key, state)

        state: pgx.State = step(state, action)
        assert state.curr_player.shape == (N,)
        assert (state.observation).sum() != 0
