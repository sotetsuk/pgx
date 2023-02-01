import random
import jax

from minatar import Environment

from pgx.minatar import breakout

from .minatar_utils import *

state_keys = {
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
_step_det = jax.jit(breakout._step_det)
_init_det = jax.jit(breakout._init_det)
observe = jax.jit(breakout.observe)

def test_step_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            s_next = extract_state(env, state_keys)
            s_next_pgx, _, _ = _step_det(
                minatar2pgx(s, breakout.State), a
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        s_next = extract_state(env, state_keys)
        s_next_pgx, _, _ = _step_det(
            minatar2pgx(s, breakout.State), a
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))


def test_init_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    N = 1
    for _ in range(N):
        env.reset()
        ball_start = 0 if env.env.ball_x == 0 else 1
        s = extract_state(env, state_keys)
        s_pgx = _init_det(ball_start)
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_observe():
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, breakout.State)
            obs_pgx = observe(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, breakout.State)
        obs_pgx = observe(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )