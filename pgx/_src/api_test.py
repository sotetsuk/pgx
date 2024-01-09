# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import get_args

import jax
import jax.numpy as jnp

from pgx.core import Env, EnvId, State
from pgx.experimental.utils import act_randomly

act_randomly = jax.jit(act_randomly)


def api_test(env: Env, num: int = 100, use_key=True):
    api_test_single(env, num, use_key)
    api_test_batch(env, num, use_key)


def api_test_single(env: Env, num: int = 100, use_key=True):
    """validate checks these items:

    - init
      - reward is zero array
      - legal_action_mask is not empty
    - step
      - state.current_player is positive
      - (TODO) taking illegal actions terminates the episode with a negative reward
      - legal_action_mask is empty when terminated (TODO: or all True?)
      - taking actions at terminal states returns the same state (with zero reward)
      - (TODO) player change at the last step before terminal
    - observe
      - Returns different observations when player_ids are different (except the initial state)
    - TODO: reward must be zero when step is called after terminated
    - TODO: observation type (bool, int32 or int32) for efficiency; https://jax.readthedocs.io/en/latest/type_promotion.html
    """

    init = jax.jit(env.init)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(849020)
    for _ in range(num):
        rng, subkey = jax.random.split(rng)
        state = init(subkey)
        assert state.env_id == env.id
        assert state.legal_action_mask.sum() != 0, "legal_action_mask at init state cannot be zero."

        assert state._step_count == 0
        curr_steps = state._step_count
        _validate_state(state)
        _validate_init_reward(state)
        _validate_current_player(state)
        _validate_legal_actions(state)

        while True:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state.legal_action_mask)
            rng, subkey = jax.random.split(rng)
            if not use_key:
                subkey = None
            state = step(state, action, subkey)
            assert state._step_count == curr_steps + 1, f"{state._step_count}, {curr_steps}"
            curr_steps += 1

            _validate_state(state)
            _validate_current_player(state)
            _validate_legal_actions(state)

            if state.terminated:
                break

    # check visualization
    filename = "/tmp/tmp.svg"
    state.save_svg(filename)
    os.remove(filename)


def api_test_batch(env: Env, num: int = 100, use_key=True):
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))

    # random play
    # TODO: add tests
    batch_size = 4
    rng = jax.random.PRNGKey(9999)

    for _ in range(num):
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, batch_size)
        state = init(keys)
        while not state.terminated.all():
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state.legal_action_mask)
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, batch_size)
            if not use_key:
                subkey = None
            state = step(state, action, keys)

    # check visualization
    filename = "/tmp/tmp.svg"
    state.save_svg(filename)
    os.remove(filename)


def _validate_init_reward(state: State):
    assert (state.rewards == jnp.zeros_like(state.rewards)).all()


def _validate_state(state: State):
    """validate_state checks these items:

    - current_player is int32
    - terminated is bool_
    - reward is float
    - legal_action_mask is bool_
    - TODO: observation is bool_ or int32 (can promote to any other types)
    """
    assert state.env_id in get_args(EnvId)
    assert state.current_player.dtype == jnp.int32, state.current_player.dtype
    assert state.terminated.dtype == jnp.bool_, state.terminated.dtype
    assert state.rewards.dtype == jnp.float32, state.rewards.dtype
    assert state.legal_action_mask.dtype == jnp.bool_, state.legal_action_mask.dtype

    # check public attributes
    public_attributes = [
        "current_player",
        "observation",
        "rewards",
        "terminated",
        "truncated",
        "legal_action_mask",
    ]
    for k, v in state.__dict__.items():
        if k.startswith("_"):  # internal
            continue
        assert k in public_attributes


def _validate_legal_actions(state: State):
    if state.terminated:
        # Agent can take any action at terminal state (but give no effect to the next state)
        # This is to avoid zero-division error by normalizing action probability by legal actions
        assert (state.legal_action_mask == jnp.ones_like(state.legal_action_mask)).all(), state.legal_action_mask
    else:
        ...


def _validate_current_player(state: State):
    assert (
        state.current_player >= 0
    ), f"current_player must be positive before terminated but got : {state.current_player}"
