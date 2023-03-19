import sys
import json
import time
import jax
import pgx
from pgx.experimental.utils import act_randomly


env_id: pgx.EnvId = sys.argv[1]
env = pgx.make(env_id)

# run api test
pgx.api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

# show multi visualizations
N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)
state: pgx.State = init(keys)
i = 0
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    state.save_svg(f"{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.reward}")
    i += 1
state.save_svg(f"{i:04d}.svg")


# throughput
def benchmark(env_id: pgx.EnvId, batch_size):
    num_steps = batch_size * 1000
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

batch_size = 1024
num_steps, sec = benchmark(env_id, batch_size)

print(json.dumps({"game": env_id,
                  "library": "pgx",
                  "total_steps": num_steps,
                  "total_sec": sec,
                  "steps/sec": num_steps / sec,
                  "batch_size": batch_size}))

