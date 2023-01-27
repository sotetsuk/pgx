from dataclasses import fields

import jax
import jax.numpy as jnp

import pgx
from pgx.utils import act_randomly


def validate(init_fn, step_fn, observe_fn, N=100):
    """validate checks these items:

    - init
      - rng is set non-default value
      - reward is zero array
    - step
      - state.curr_player is positive when not terminated
      - state.curr_player = -1 when terminated
      - rng changes after each step
      - (TODO) taking illegal actions terminates the episode with a negative reward
      - legal_action_mask is empty when terminated
      - taking actions at terminal states returns the same state (with zero reward)
    - observe
      - Returns different observations when player_ids are different (except the initial state)
      - Returns zero observations when player_id=-1 (curr_player is set -1 when terminated)
    """
    rng = jax.random.PRNGKey(849020)
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        state = init_fn(subkey)

        _validate_state(state)
        _validate_init_reward(state)
        _validate_curr_player(state)
        _validate_legal_actions(state)

        while True:
            prev_rng = state.rng
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
            state = step_fn(state, action)

            _validate_state(state)
            _validate_curr_player(state)
            _validate_obs(observe_fn, state)
            _validate_rng_changes(state, prev_rng)
            _validate_legal_actions(state)

            if state.terminated:
                break

        _validate_taking_action_after_terminal(state, step_fn)


def _validate_taking_action_after_terminal(state: pgx.State, step_fn):
    prev_state = state
    if not state.terminated:
        return
    action = 0
    state = step_fn(state, action)
    assert (state.reward == 0).all()
    for field in fields(state):
        if field.name == "reward" or field.name == "rng":
            continue
        assert (
            getattr(state, field.name) == getattr(prev_state, field.name)
        ).all(), f"{field.name} : \n{getattr(state, field.name)}\n{getattr(prev_state, field.name)}"


def _validate_rng_changes(state, prev_rng):
    assert not (state.rng == prev_rng).all()


def _validate_init_reward(state: pgx.State):
    assert (state.reward == jnp.zeros_like(state.reward)).all()


def _validate_state(state: pgx.State):
    """validate_state checks these items:

    - rng is uint32 and different from zero key
    - curr_player is int8
    - terminated is bool_
    - reward is float
    - legal_action_mask is bool_
    """
    assert state.rng.dtype == jnp.uint32, state.rng.dtype
    assert (state.rng != 0).all(), state.rng  # type: ignore
    assert state.curr_player.dtype == jnp.int8, state.curr_player.dtype
    assert state.terminated.dtype == jnp.bool_, state.terminated.dtype
    assert state.reward.dtype == jnp.float32, state.reward.dtype
    assert (
        state.legal_action_mask.dtype == jnp.bool_
    ), state.legal_action_mask.dtype


def _validate_obs(observe_fn, state: pgx.State):
    # basic usage
    obs = observe_fn(state, state.curr_player)
    if state.terminated:
        assert (obs == 0).all(), f"Got non-zero obs at terminal state : {obs}"
    else:
        assert not (
            obs == 0
        ).all(), f"Got zero obs at non terminal state : {obs}"

    # when terminal
    obs = observe_fn(state, -1)
    assert (
        obs == 0
    ).all(), f"player_id = -1 must return zero obs but got : {obs}"

    # when player_id is different from state.curr_player
    obs_default = observe_fn(state, player_id=state.curr_player)
    obs_player_0 = observe_fn(state, player_id=0)
    if state.curr_player == 0:
        assert (
            obs_default == obs_player_0
        ).all(), f"Got different obs : \n{obs_default}\n{obs_player_0}"
    else:
        assert not (
            obs_default == obs_player_0
        ).all(), f"Got same obs : \n{obs_default}\n{obs_player_0}"


def _validate_legal_actions(state: pgx.State):
    if state.terminated:
        assert (
            state.legal_action_mask == jnp.zeros_like(state.legal_action_mask)
        ).all(), state.legal_action_mask
    else:
        ...


def _validate_curr_player(state: pgx.State):
    if state.terminated:
        assert (
            state.curr_player == -1
        ), f"curr_player must be -1 when terminated but got : {state.curr_player}"
    else:
        assert (
            state.curr_player >= 0
        ), f"curr_player must be positivie before terminated but got : {state.curr_player}"
