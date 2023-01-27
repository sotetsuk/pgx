import jax

import pgx
from pgx.utils import act_randomly


def validate_state(state):
    """validate_state checks these items:

    - rng is set and different from zero key
    - curr_player is int8
    - terminated is bool_
    - reward is float
    - legal_action_mask is bool_
    """
    ...


def validate_init(init_fn):
    """validate_init checks these items:

    - rng is set non-default value

    - reward is zero array
    """
    rng = jax.random.PRNGKey(38292)
    state = init_fn(rng)

    assert not (
        state.rng == jax.random.PRNGKey(0)  # default
    ).all(), f"Non-default rng must be set to state.rng: got {state.rng}"


def validate_step(init_fn, step_fn, N=100):
    """validate_step checks these items:

    - state.curr_player is positive when not terminated
    - state.curr_player = -1 when terminated

    - rng changes after each step
    - taking illegal actions terminates the episode with a negative reward
    - legal_action_mask is empty when terminated
    - taking actions at terminal states returns the same state (with zero reward)
    """
    rng = jax.random.PRNGKey(38201)
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        state = init_fn(rng)
        while True:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
            state = step_fn(state, action)

            validate_curr_player(state)

            if state.terminated:
                break


def validate_observe(init_fn, step_fn, observe_fn, N=100):
    """validate_observe checks these items:

    - Returns different observations when player_ids are different
    - Returns zero observations when player_id=-1 (curr_player is set 0 when terminated)
    """
    rng = jax.random.PRNGKey(849020)
    for _ in range(N):
        rng, subkey = jax.random.split(rng)
        state = init_fn(rng)
        while True:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
            state = step_fn(state, action)

            _validate_zero_obs(observe_fn, state)

            if state.terminated:
                break


def _validate_zero_obs(observe_fn, state: pgx.State):
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


def validate_illegal_actions(state: pgx.State):
    ...


def validate_curr_player(state: pgx.State):
    if state.terminated:
        assert (
            state.curr_player == -1
        ), f"curr_player must be -1 when terminated but got : {state.curr_player}"
    else:
        assert (
            state.curr_player >= 0
        ), f"curr_player must be positivie before terminated but got : {state.curr_player}"
