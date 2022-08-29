import copy
import random
from typing import Any, Dict

import numpy as np
from flax import struct
from jax import numpy as jnp
from minatar import Environment

from pgx.minatar import breakout

breakout_state_keys = {
    "ball_y",
    "ball_x",
    "ball_dir",
    "pos",
    "brick_map",
    "strike",
    "last_x",
    "last_y",
    "terminal",
    "last_action",
}


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


def pgx2minatar(state: breakout.MinAtarBreakoutState) -> Dict[str, Any]:
    d = {}
    for key in breakout_state_keys:
        d[key] = copy.deepcopy(getattr(state, key))
        if isinstance(d[key], jnp.ndarray):
            d[key] = np.array(d[key])
    return d


def minatar2pgx(state_dict: Dict[str, Any]) -> breakout.MinAtarBreakoutState:
    d = {}
    for key in breakout_state_keys:
        val = copy.deepcopy(state_dict[key])
        if isinstance(val, np.ndarray):
            val = jnp.array(val)
        d[key] = val
    s = breakout.MinAtarBreakoutState(**d)
    return s


def test_step_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, breakout_state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            s_next = extract_state(env, breakout_state_keys)
            s_next_pgx, _, _ = breakout._step_det(minatar2pgx(s), a)
            assert_states(s_next, pgx2minatar(s_next_pgx))

        # check terminal state
        s = extract_state(env, breakout_state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        s_next = extract_state(env, breakout_state_keys)
        s_next_pgx, _, _ = breakout._step_det(minatar2pgx(s), a)
        assert_states(s_next, pgx2minatar(s_next_pgx))


def test_reset_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    N = 100
    for _ in range(N):
        env.reset()
        ball_start = 0 if env.env.ball_x == 0 else 1
        s = extract_state(env, breakout_state_keys)
        s_pgx = breakout._reset_det(ball_start)
        assert_states(s, pgx2minatar(s_pgx))
