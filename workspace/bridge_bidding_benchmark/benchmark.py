import sys
import time

import jax
import jax.numpy as jnp

from pgx.bridge_bidding import (
    _calculate_dds_tricks,
    _init_by_key,
    _key_to_hand,
    _load_sample_hash,
    _shuffle_players,
    _state_to_key,
    _to_binary,
    duplicate,
    _init_by_key,
    BridgeBidding,
    _terminated_step,
    _continue_step,
    _state_bid,
    _find_value_from_key,
    _state_pass,
    _step,
    _observe,
)

env = BridgeBidding(dds_hash_table_path="dds_hash_table.npz")
init = env.init
step = env.step
observe = env.observe
rng = jax.random.PRNGKey(0)


def test(func):
    rng = jax.random.PRNGKey(0)
    state = init(rng)
    if func.__name__ == "_init_by_key":
        HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
        time_sta = time.perf_counter()
        jax.jit(func)(HASH_TABLE_SAMPLE_KEYS[0], rng)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(HASH_TABLE_SAMPLE_KEYS[0], rng)
    elif func.__name__ == "init":
        time_sta = time.perf_counter()
        jax.jit(func)(rng)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(rng)
    elif func.__name__ == "step":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, 0)
    elif func.__name__ == "observe":
        time_sta = time.perf_counter()
        jax.jit(func)(state, state.current_player)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, state.current_player)
    elif func.__name__ == "duplicate":
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state)
    elif func.__name__ == "_shuffle_players":
        time_sta = time.perf_counter()
        jax.jit(func)(rng)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(rng)
    elif func.__name__ == "_terminated_step":
        time_sta = time.perf_counter()
        jax.jit(func)(state, env.hash_keys, env.hash_values)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, env.hash_keys, env.hash_values)
    elif func.__name__ == "_continue_step":
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state)
    elif func.__name__ == "_state_bid":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 3)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, 3)
    elif func.__name__ == "_state_to_key":
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state)
    elif func.__name__ == "_to_binary":
        x = jnp.arange(52, dtype=jnp.int8)[::-1].reshape((4, 13)) % 4
        time_sta = time.perf_counter()
        jax.jit(func)(x)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(x)
    elif func.__name__ == "_key_to_hand":
        key = env.hash_keys[0]
        time_sta = time.perf_counter()
        jax.jit(func)(key)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(key)
    elif func.__name__ == "_calculate_dds_tricks":
        time_sta = time.perf_counter()
        jax.jit(func)(state, env.hash_keys, env.hash_values)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, env.hash_keys, env.hash_values)
    elif func.__name__ == "_find_value_from_key":
        key = env.hash_keys[0]
        time_sta = time.perf_counter()
        jax.jit(func)(key, env.hash_keys, env.hash_values)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(key, env.hash_keys, env.hash_values)
    elif func.__name__ == "_step":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 0, env.hash_keys, env.hash_values)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, 0, env.hash_keys, env.hash_values)
    elif func.__name__ == "_observe":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, 0)

    n_line = len(str(exp).split("\n"))
    print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")
    return


func_name = sys.argv[1]
if func_name == "_init_by_key":
    func = _init_by_key
elif func_name == "_shuffle_players":
    func = _shuffle_players
elif func_name == "duplicate":
    func = duplicate
elif func_name == "init":
    func = init
elif func_name == "step":
    func = step
elif func_name == "_terminated_step":
    func = _terminated_step
elif func_name == "_continue_step":
    func = _continue_step
elif func_name == "observe":
    func = observe
elif func_name == "_state_bid":
    func = _state_bid
elif func_name == "_state_to_key":
    func = _state_to_key
elif func_name == "_to_binary":
    func = _to_binary
elif func_name == "_key_to_hand":
    func = _key_to_hand
elif func_name == "_calculate_dds_tricks":
    func = _calculate_dds_tricks
elif func_name == "_find_value_from_key":
    func = _find_value_from_key
elif func_name == "_step":
    func = _step
elif func_name == "_observe":
    func = _observe
else:
    print(func_name)
    assert False

test(func=func)
