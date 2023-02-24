import time
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


N = (2 ** 12) * 1
# print(f"Total # of steps: {N}")
bs_list = [2 ** i for i in range(5, 13)]
# print("| env_id |" + "|".join([str(bs) for bs in bs_list]) + "|")
# print("|:---:|" + "|".join([":---:" for bs in bs_list]) + "|")
d = {}
for env_id in get_args(pgx.EnvId):
    # s = f"|{env_id}|"
    for bs in bs_list:
        num_steps, sec = benchmark(env_id, bs, N)
        # s += f"{n_per_sec:.05f}"
        # s += "|"
        print(json.dumps({"game": "/".join(env_id.split("/")[:-1]), "library": "pgx",
              "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
    # print(s)


"""
Total # of steps: 409600
| env_id |32|64|128|256|512|1024|2048|4096|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|tic_tac_toe/v0|25193.54921|51203.80125|99197.43688|206175.78196|413948.81221|723250.23824|1664977.36893|3265886.46947|
|go-19x19/v0|15146.45769|29891.62027|58064.94882|108400.19704|173638.40814|286740.95368|379331.88909|449555.32632|
|shogi/v0|21047.48879|42130.05279|82988.81210|175415.01266|259940.79393|290410.20642|299800.69880|308552.26434|
|minatar/asterix/v0|13066.68075|25836.00751|52134.46018|102929.03752|205880.59846|384825.85566|806843.53100|1553951.76960|
"""
