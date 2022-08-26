import copy
import random

import numpy as np
from minatar import Environment

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
}


def extract_state(env, state_keys):
    state_dict = {}
    # task-dependent attribute
    for k in state_keys:
        state_dict[k] = copy.deepcopy(getattr(env.env, k))
    # last_action and random state
    for k in ("last_action",):
        state_dict[k] = copy.deepcopy(getattr(env, k))
    return state_dict


def assert_states(state1, state2):
    keys = state1.keys()
    assert keys == state2.keys()
    for key in keys:
        assert np.allclose(
            state1[key], state2[key]
        ), f"{key}, {state1[key]}, {state2[key]}"


def test_step():
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    env.reset()
    s = extract_state(env, breakout_state_keys)
    a = random.randrange(num_actions)
    env.act(a)
    s_next = extract_state(env, breakout_state_keys)

    assert_states(s, s_next)
