import copy
from typing import Any, Dict

import numpy as np
from jax import numpy as jnp


def extract_state(env, state_keys):
    state_dict = {}
    # task-dependent attribute
    for k in state_keys:
        if k in ("last_action",):
            state_dict[k] = copy.deepcopy(getattr(env, k))
        else:
            state_dict[k] = copy.deepcopy(getattr(env.env, k))
    return state_dict


def assert_states(state1, state2):
    keys = state1.keys()
    assert keys == state2.keys()
    for key in keys:
        assert np.allclose(
            state1[key], state2[key]
        ), f"{key}, {state1[key]}, {state2[key]}"


def pgx2minatar(state, keys) -> Dict[str, Any]:
    d = {}
    for key in keys:
        d[key] = copy.deepcopy(getattr(state, key))
        if isinstance(d[key], jnp.ndarray):
            d[key] = np.array(d[key])
    return d


def minatar2pgx(state_dict: Dict[str, Any], state_cls):
    d = {}
    for key in state_dict.keys():
        val = copy.deepcopy(state_dict[key])
        if isinstance(val, np.ndarray):
            if key in (
                "brick_map",
                "alien_map",
                "f_bullet_map",
                "e_bullet_map",
            ):
                val = jnp.array(val, dtype=bool)
            else:
                val = jnp.array(val, dtype=int)
        d[key] = val
    s = state_cls(**d)
    return s
