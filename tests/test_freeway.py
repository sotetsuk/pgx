import random

from minatar import Environment

from pgx.minatar import freeway

from .minatar_utils import *

state_keys = {}


def test_spawn_entity():
    entities = jnp.ones((8, 4), dtype=int) * 1e5
    entities = entities.at[:, :].set(
        freeway._spawn_entity(entities, True, True, 1)
    )
    assert entities[1][0] == 0, entities
    assert entities[1][1] == 2, entities
    assert entities[1][2] == 1, entities
    assert entities[1][3] == 1, entities


def test_step_det():
    env = Environment("freeway", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            lr, is_gold, slot = env.env.lr, env.env.is_gold, env.env.slot
            s_next = extract_state(env, state_keys)
            s_next_pgx, _, _ = freeway._step_det(
                minatar2pgx(s, freeway.MinAtarFreewayState),
                a,
                lr,
                is_gold,
                slot,
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        lr, is_gold, slot = env.env.lr, env.env.is_gold, env.env.slot
        s_next = extract_state(env, state_keys)
        s_next_pgx, _, _ = freeway._step_det(
            minatar2pgx(s, freeway.MinAtarFreewayState), a, lr, is_gold, slot
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))


def test_reset_det():
    env = Environment("freeway", sticky_action_prob=0.0)
    N = 100
    for _ in range(N):
        env.reset()
        s = extract_state(env, state_keys)
        s_pgx = freeway._reset_det()
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_to_obs():
    env = Environment("freeway", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, freeway.MinAtarFreewayState)
            obs_pgx = freeway._to_obs(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, freeway.MinAtarFreewayState)
        obs_pgx = freeway._to_obs(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )
