import time
import sys
import json
import jax
import pgx
from pgx.experimental.utils import act_randomly
import jax.numpy as jnp

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)



def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.
    There are several concerns before staging this wrapper:
    1. Final state (observation)
    When auto restting happened, the termianl (or truncated) state/observation is replaced by initial state/observation,
    This is not problematic if it's termination.
    However, when truncation happened, value of truncated state/observation might be used by agent.
    So we have to preserve the final state (observation) somewhere.
    For example, in Gymnasium,
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59
    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and terminates in reasonable # of steps.
    (NOTE: Chess, Shogi, and Go have `max_termination_steps` option following AlphaZero approach)
    So, curren implementation is enough (so far), and I feel implementing `final_state/observation` is useless and not necessary.
    2. Performance:
    Might harm the performance as it always generates new state.
    Memory usage might be doubled. Need to check.
    """

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


def benchmark(env_id: pgx.EnvId, batch_size, num_steps):
    num_batch_step = num_steps // batch_size

    env = pgx.make(env_id)
    assert env is not None

    num_devices = jax.device_count()
    batchsize_per_dev = batch_size // num_devices
    # print(num_devices)
    # print(batchsize_per_dev)

    @jax.pmap
    def run(key):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batchsize_per_dev)
        state = jax.vmap(env.init)(keys)

        def fn(i, x):
            key, s = x
            del i
            key, subkey = jax.random.split(key)
            action = act_randomly(subkey, state)
            s = jax.vmap(env.step)(s, action)
            return key, s

        _, state = jax.lax.fori_loop(
            0, num_batch_step, fn, (rng_key, state)
        )
        return state

    rng_key = jax.random.PRNGKey(0)

    # warmup start
    rng_key, subkey = jax.random.split(rng_key)
    pmap_keys = jax.random.split(subkey, num_devices)
    s = run(pmap_keys)
    # warmup end

    ts = time.time()
    rng_key, subkey = jax.random.split(rng_key)
    pmap_keys = jax.random.split(subkey, num_devices)
    run(pmap_keys)
    te = time.time()

    return num_steps, te - ts


games = {
    "tic_tac_toe": "tic_tac_toe",
    "connect_four": "connect_four",
    "backgammon": "backgammon",
    "shogi": "shogi",
    "go": "go_19x19",
    "chess": "chess",
}


game = sys.argv[1]
bs = int(sys.argv[2])
num_batch_steps = int(sys.argv[3])
env_id = games[game]

num_steps, sec = benchmark(env_id, bs, bs * num_batch_steps)
print(json.dumps({"game": game, "library": "pgx (A100 x 8)", "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
