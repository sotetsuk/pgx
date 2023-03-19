import time
import sys
import json
import jax
import pgx
from pgx.experimental.utils import act_randomly
from pgx.experimental.wrappers import auto_reset


act_randomly = jax.jit(act_randomly)


def benchmark(env_id: pgx.EnvId, batch_size):
    num_steps = batch_size * 1_000
    num_batch_step = num_steps // batch_size

    env = pgx.make(env_id)
    assert env is not None

    # warmup start
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    state = init(keys)
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state)
    state = step(state, action)
    # warmup end

    ts = time.time()
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    state = init(keys)
    for i in range(num_batch_step):
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state)
        state = step(state, action)
    te = time.time()

    return num_steps, te - ts


games = {
    "tic_tac_toe": "tic_tac_toe",
    "backgammon": "backgammon",
    "shogi": "shogi",
    "go": "go-19x19",
}


bs_list = [2 ** i for i in range(1, 11)]
d = {}
for game, env_id in games.items():
    for bs in bs_list:
        num_steps, sec = benchmark(env_id, bs)
        print(json.dumps({"game": game, "library": "pgx",
              "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
