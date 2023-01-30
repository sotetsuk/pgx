import random
import jax

from minatar import Environment

from pgx.minatar import freeway

from .minatar_utils import *

state_keys = [
    "cars",
    "pos",
    "move_timer",
    "terminate_timer",
    "terminal",
    "last_action",
]

_step_det = jax.jit(freeway._step_det)
_init_det = jax.jit(freeway._init_det)
_to_obs = jax.jit(freeway._to_obs)

def test_step_det():
    env = Environment("freeway", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 3
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            # extract random variables
            speeds, directions = jnp.array(env.env.speeds), jnp.array(
                env.env.directions
            )
            s_next = extract_state(env, state_keys)
            s_next_pgx, _, _ = _step_det(
                minatar2pgx(s, freeway.State),
                a,
                speeds,
                directions,
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        # extract random variables
        speeds, directions = jnp.array(env.env.speeds), jnp.array(
            env.env.directions
        )
        s_next = extract_state(env, state_keys)
        s_next_pgx, _, _ = _step_det(
            minatar2pgx(s, freeway.State), a, speeds, directions
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))


def test_init_det():
    env = Environment("freeway", sticky_action_prob=0.0)
    N = 10
    for _ in range(N):
        env.reset()
        s = extract_state(env, state_keys)
        # extract random variables
        speeds = jnp.array(env.env.speeds)
        directions = jnp.array(env.env.directions)
        s_pgx = _init_det(speeds, directions)
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_observe():
    env = Environment("freeway", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 1  # TODO: increase N
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, freeway.State)
            obs_pgx = _to_obs(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, freeway.State)
        obs_pgx = _to_obs(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )
