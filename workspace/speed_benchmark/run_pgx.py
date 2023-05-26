import time
import sys
import json
import jax
import pgx
from pgx.experimental.utils import act_randomly
import jax.numpy as jnp

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


num_devices = jax.device_count()


def auto_reset(step_fn, init_fn):
    def wrapped_step_fn(state: pgx.State, action):
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(  # type: ignore
                _step_count=jnp.int32(0),
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action)
        state = jax.lax.cond(
            state.terminated,
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn


def benchmark(env_id: pgx.EnvId, batch_size, num_batch_steps):
    N = 100
    assert num_batch_steps % N == 0
    assert batch_size % num_devices == 0
    num_steps = batch_size * num_batch_steps
    env = pgx.make(env_id)

    def step_for(key, state, n):

        def step_fn(i, x):
            del i
            key, state = x
            action = act_randomly(key, state)
            state = auto_reset(env.step, env.init)(state, action)
            return (key, state)

        _, state = jax.lax.fori_loop(0, n, step_fn, (key, state))
        return state

    init_fn = jax.pmap(jax.vmap(env.init))
    step_for = jax.pmap(jax.vmap(step_for, in_axes=(0, 0, None)), in_axes=(0, 0, None))

    rng_key = jax.random.PRNGKey(0)

    # warmup
    # print("compiling ... ")
    ts = time.time()
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    keys = keys.reshape(num_devices, -1, 2)
    s = init_fn(keys)
    s = step_for(keys, s, N)
    te = time.time()
    # print(f"done: {te - ts:.03f} sec")
    # warmup end

    ts = time.time()

    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    keys = keys.reshape(num_devices, -1, 2)
    s = init_fn(keys)
    for i in range(num_batch_steps // N):
        rng_key, subkey = jax.random.split(rng_key)
        keys = keys.reshape(num_devices, -1, 2)
        s = step_for(keys, s, N)

    te = time.time()
    return num_steps, te - ts


game = sys.argv[1]
env_id = game
if env_id == "go":
    env_id = "go_19x19"
bs = int(sys.argv[2])
num_batch_steps = int(sys.argv[3])


num_steps, sec = benchmark(env_id, bs, num_batch_steps)
print(json.dumps({"game": game, "library": f"pgx/{num_devices}dev", "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs, "pgx.__version__": pgx.__version__}))
