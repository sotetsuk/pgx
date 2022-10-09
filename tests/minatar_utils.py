import copy
from typing import Any, Dict

import numpy as np
from jax import numpy as jnp


INF = 99

def extract_state(env, state_keys):
    state_dict = {}
    # task-dependent attribute
    for k in state_keys:
        state_dict[k] = copy.deepcopy(getattr(env.env, k))
    return state_dict


def assert_states(state1, state2):
    keys = state1.keys()
    assert keys == state2.keys()
    for key in keys:
        if key == "entities":
            assert len(state1[key]) == len(state2[key])
            for s1, s2 in zip(state1[key], state2[key]):
                assert s1 == s2, f"{s1}, {s2}\n{state1}\n{state2}"
        else:
            assert np.allclose(
                state1[key], state2[key]
            ), f"{key}, {state1[key]}, {state2[key]}\n{state1}\n{state2}"


def pgx2minatar(state, keys) -> Dict[str, Any]:
    d = {}
    for key in keys:
        d[key] = copy.deepcopy(getattr(state, key))
        if isinstance(d[key], jnp.ndarray):
            d[key] = np.array(d[key])
        if key == "entities":
            val = [None] * 8
            for i in range(8):
                if d[key][i][0] != INF:
                    e = [d[key][i][j] for j in range(4)]
                    val[i] = e
            d[key] = val
    return d


def minatar2pgx(state_dict: Dict[str, Any], state_cls):
    d = {}
    for key in state_dict.keys():
        val = copy.deepcopy(state_dict[key])

        # Exception in Asterix
        if key == "entities":
            _val = [[INF if x is None else x[j] for j in range(4)] for i, x in enumerate(val)]
            val = jnp.array(_val, dtype=jnp.int8)
            d[key] = val
            continue

        # Exception in Seaquest
        if key in ["f_bullets", "e_bullets", "e_fish", "e_subs", "divers"]:
            N = 25 if key.startswith("e_") else 5
            M = 3 if key.endswith("bullets") else 4
            if key == "e_subs":
                M = 5
            v = - jnp.ones((N, M), dtype=jnp.int8)
            for i, x in enumerate(val):
                v = v.at[i, :].set(jnp.array(x))
            d[key] = v
            continue

        # Cast to int16
        if key in ["terminate_timer", "oxygen"]:
            val = jnp.array(val, dtype=jnp.int16)
            d[key] = val
            continue

        # Cast to bool
        if isinstance(val, np.ndarray):
            if key in (
                "brick_map",
                "alien_map",
                "f_bullet_map",
                "e_bullet_map",
                "allien_map",
            ):
                val = jnp.array(val, dtype=jnp.bool_)
            else:
                val = jnp.array(val, dtype=jnp.int8)
            d[key] = val
            continue

        if key in ["terminal", "sub_or", "surface"]:
            val = jnp.array(val, dtype=jnp.bool_)
        else:
            val = jnp.array(val, dtype=jnp.int8)
        d[key] = val

    s = state_cls(**d)
    return s
