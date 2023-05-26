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
    num_steps = batch_size * num_batch_steps

    env = pgx.make(env_id)
    assert env is not None

    batch_size_per_dev = batch_size // num_devices

    def run(key):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size_per_dev)
        state = jax.vmap(env.init)(keys)

        def fn(i, x):
            key, s = x
            del i
            key, subkey = jax.random.split(key)
            action = act_randomly(subkey, state)
            s = jax.vmap(env.step)(s, action)
            return key, s

        _, state = jax.lax.fori_loop(
            0, num_batch_steps, fn, (rng_key, state)
        )
        return state

    rng_key = jax.random.PRNGKey(0)

    if num_devices > 1:
        run = jax.pmap(run)
        # warmup start
        rng_key, subkey = jax.random.split(rng_key)
        pmap_keys = jax.random.split(subkey, num_devices)
        s = run(pmap_keys)
        # warmup end
        ts = time.time()
        rng_key, subkey = jax.random.split(rng_key)
        pmap_keys = jax.random.split(subkey, num_devices)
        s = run(pmap_keys)
        te = time.time()
    else:
        run = jax.jit(run)
        # warmup start
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        s = run(subkey)
        # warmup end
        ts = time.time()
        key, subkey = jax.random.split(key)
        s = run(subkey)
        te = time.time()

    return num_steps, te - ts


games = {
    "tic_tac_toe": "tic_tac_toe",
    "connect_four": "connect_four",
    "go": "go_19x19",
    "chess": "chess",
}


game = sys.argv[1]
bs = int(sys.argv[2])
num_batch_steps = int(sys.argv[3])
env_id = games[game]


num_steps, sec = benchmark(env_id, bs, bs * num_batch_steps)
print(json.dumps({"game": game, "library": f"pgx/{num_devices}dev", "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
