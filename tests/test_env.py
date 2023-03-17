import jax
import pgx
from pgx.experimental.utils import act_randomly
from typing import get_args


def test_jit():
    N = 2
    for env_name in get_args(pgx.EnvId):
        print(f"{env_name} ...")

        env = pgx.make(env_name)

        print(env.num_players)
        print(env.observation_shape)
        print(env.action_shape)

        init = jax.jit(jax.vmap(env.init))
        step = jax.jit(jax.vmap(env.step))
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, N)

        state = init(keys)
        assert state.current_player.shape == (N,)
        # assert (state.observation).sum() != 0

        action = act_randomly(key, state)

        state: pgx.State = step(state, action)
        assert state.current_player.shape == (N,)
        assert (state.observation).sum() != 0


def test_env():
    for env_name in get_args(pgx.EnvId):
        print(f"{env_name} ...")
        env = pgx.make(env_name)
        pgx.api_test(env)
