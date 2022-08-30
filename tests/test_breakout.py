import random

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
            s_next_pgx, _, _ = breakout._step_det(
                minatar2pgx(s, breakout.MinAtarBreakoutState), a
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        s_next = extract_state(env, state_keys)
        s_next_pgx, _, _ = breakout._step_det(
            minatar2pgx(s, breakout.MinAtarBreakoutState), a
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))


def test_reset_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    N = 100
    for _ in range(N):
        env.reset()
        ball_start = 0 if env.env.ball_x == 0 else 1
        s = extract_state(env, state_keys)
        s_pgx = breakout._reset_det(ball_start)
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_to_obs():
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            assert jnp.allclose(
                env.state(),
                breakout._to_obs(
                    minatar2pgx(s, breakout.MinAtarBreakoutState)
                ),
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        assert jnp.allclose(
            env.state(),
            breakout._to_obs(minatar2pgx(s, breakout.MinAtarBreakoutState)),
        )
