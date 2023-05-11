import time
import sys
import json
import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.utils import act_randomly

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


def benchmark(env_id: pgx.EnvId, batch_size, num_batch_steps):

    env = pgx.make(env_id)
    assert env is not None

    @jax.jit
    def run(key):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(env.init)(keys)

        def fn(i, x):
            key, s = x
            del i
            key, subkey = jax.random.split(key)
            action = act_randomly(subkey, state)
            s = jax.vmap(env.step)(s, action)
            return key, s

        _, state = jax.lax.fori_loop(
            0, num_batch_steps, fn, (key, state)
        )
        return state

    # warmup

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    run(subkey)

    ts = time.time()
    key, subkey = jax.random.split(key)
    run(subkey)
    te = time.time()

    return te - ts


games = {
    "tic_tac_toe": "tic_tac_toe",
    "backgammon": "backgammon",
    "connect_four": "connect_four",
    "chess": "chess",
    "go": "go_19x19",
}


num_batch_steps = int(sys.argv[1])
bs_list = [2 ** i for i in range(1, 16)]
d = {}
for game, env_id in games.items():
    for bs in bs_list:
        num_steps = bs * num_batch_steps
        sec = benchmark(env_id, bs, num_batch_steps)
        print(json.dumps({"game": game, "library": "pgx (A100 x 1)",
              "total_steps": num_steps, "total_sec": sec, "steps/sec": num_steps / sec, "batch_size": bs}))
