import sys
import time
from functools import partial

import jax
import jax.numpy as jnp

from pgx.utils import  act_randomly
from pgx.go import (
    _count_ji,
    _get_alphazero_features,
    _get_reward,
    _merge_ren,
    _not_pass_move,
    _pass_move,
    _remove_stones,
    _set_stone_next_to_oppo_ren,
    _update_state_wo_legal_action,
    init,
    legal_actions,
    step,
)

rng = jax.random.PRNGKey(0)
size = 19


def test_vmap(n_vmap=1):
    vmap_init = jax.vmap(partial(init, size=size))
    vmap_step = jax.vmap(partial(step, size=size))

    # warmup
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, n_vmap)

    print("start warm up ...")
    _, state = vmap_init(rng=keys)
    time_sta = time.time()
    action = act_randomly(rng, state)
    _, state, _ = vmap_step(state=state, action=action)
    time_end = time.time()
    c_delta = (time_end - time_sta)
    print("end warm up ...")

    time_sta = time.time()
    _, state = vmap_init(rng=keys)
    for i in range(10):
        action = act_randomly(rng, state)
        _, state, _ = vmap_step(state=state, action=action)
    time_end = time.time()
    r_delta = (time_end - time_sta)

    print(f"| {n_vmap} | {c_delta:.3f} sec | {r_delta:.3f} sec |")


test_vmap(sys.argv[1])
