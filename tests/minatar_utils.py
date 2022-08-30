import copy
from typing import Any, Dict

import numpy as np
from jax import numpy as jnp


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
            assert len(state1) == len(state2)
            for s1, s2 in zip(state1, state2):
                assert s1 == s2
        else:
            assert np.allclose(
                state1[key], state2[key]
            ), f"{key}, {state1[key]}, {state2[key]}"


def pgx2minatar(state, keys) -> Dict[str, Any]:
    d = {}
    for key in keys:
        d[key] = copy.deepcopy(getattr(state, key))
        if isinstance(d[key], jnp.ndarray):
            d[key] = np.array(d[key])
        if key == "entities":
            val = [None] * 8
            for i in range(8):
                if d[key][i][0] != 1e5:
                    e = [d[key][i][j] for j in range(4)]
                    val[i] = e
            d[key] = val
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
        if key == "entities":
            _val = jnp.ones((8, 4), dtype=int) * 1e5
            for i, x in enumerate(val):
                if x is None:
                    continue
                for j in range(4):
                    _val = _val.at[i, j].set(x[j])
            val = _val
        d[key] = val
    s = state_cls(**d)
    return s
