import time
import sys
import json
import jax
import pgx
from pgx.utils import act_randomly
from tqdm import tqdm
from typing import get_args


# TODO: autoreset
def benchmark(env_id: pgx.EnvId, batch_size, num_steps=(2 ** 12) * 1000):
    assert num_steps % batch_size == 0
    num_batch_step = num_steps // batch_size

    env = pgx.make(env_id)
    assert env is not None

    # warmup start
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
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


N = int(sys.argv[1])
bs_list = [2 ** i for i in range(1, 11)]
d = {}
for game, env_id in games.items():
    for bs in bs_list:
        num_steps, sec = benchmark(env_id, bs, N)
        print(json.dumps({"game": game, "library": "pgx",
              "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
