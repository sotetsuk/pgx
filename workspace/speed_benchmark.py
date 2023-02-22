import time
import jax
import pgx
from pgx.utils import act_randomly
from typing import get_args


# TODO: autoreset
def benchmark(env_id: pgx.EnvId, batch_size, num_steps=(2 ** 10) * 1000):
    assert num_steps % batch_size == 0
    num_batch_step = num_steps // batch_size

    env = pgx.make(env_id)
    assert env is not None

    # warmup start
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
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

    return f"{num_steps / (te - ts):.05f}"


if __name__ == '__main__':
    N = (2 ** 10) * 10
    print(f"Total # of steps: {N}")
    bs_list = [2 ** i for i in range(8, 11)]
    print("| env_id |" + "|".join([str(bs) for bs in bs_list]) + "|")
    print("|:---:|" + "|".join([":---:" for bs in bs_list]) + "|")
    for env_id in get_args(pgx.EnvId):
        s = f"|{env_id}|"
        for bs in bs_list:
            s += benchmark(env_id, bs, N)
            s += "|"
        print(s)
