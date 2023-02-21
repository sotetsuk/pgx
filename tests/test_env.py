import jax
import pgx
from pgx.utils import act_randomly
from pgx.tic_tac_toe import TicTacToe
from pgx.shogi import Shogi


def test_jit():
    N = 2
    for Env in [TicTacToe, Shogi]:
        env = Env()
        init = jax.jit(jax.vmap(env.init))
        step = jax.jit(jax.vmap(env.step))

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, N)
        state = init(keys)
        action = act_randomly(key, state)
        state: pgx.State = step(state, action)

        print(state.legal_action_mask.shape)

        assert state.curr_player.shape == (N,)

    assert False
