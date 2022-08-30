import random

from minatar import Environment

from pgx.minatar import asterix

from .minatar_utils import *

state_keys = {
    "player_x",
    "player_y",
    "entities",
    "shot_timer",
    "spawn_speed",
    "spawn_timer",
    "move_speed",
    "move_timer",
    "ramp_timer",
    "ramp_index",
    "terminal",
    "last_action",
}


def test_spawn_entity():
    entities = jnp.ones((8, 4), dtype=int) * 1e5
    entities = entities.at[:, :].set(
        asterix._spawn_entity(entities, True, True, 1)
    )
    assert entities[1][0] == 0, entities
    assert entities[1][1] == 2, entities
    assert entities[1][2] == 1, entities
    assert entities[1][3] == 1, entities


def test_step_det():
    env = Environment("asterix", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 1000
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            lr, is_gold, slot = env.env.lr, env.env.is_gold, env.env.slot
            s_next = extract_state(env, state_keys)
            s_next_pgx, _, _ = asterix._step_det(
                minatar2pgx(s, asterix.MinAtarAsterixState),
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
        s_next_pgx, _, _ = asterix._step_det(
            minatar2pgx(s, asterix.MinAtarAsterixState), a, lr, is_gold, slot
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))
