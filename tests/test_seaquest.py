import random
import jax
from dataclasses import fields

from minatar import Environment

from pgx.minatar import seaquest

from .minatar_utils import *

state_keys = {
    "oxygen",
    "diver_count",
    "sub_x",
    "sub_y",
    "sub_or",
    "f_bullets",
    "e_bullets",
    "e_fish",
    "e_subs",
    "divers",
    "e_spawn_speed",
    "e_spawn_timer",
    "d_spawn_timer",
    "move_speed",
    "ramp_index",
    "shot_timer",
    "surface",
    "terminal",
    "last_action",
}

_step_det = jax.jit(seaquest._step_det)
_init_det = jax.jit(seaquest._init_det)
observe = jax.jit(seaquest._observe)


def test_step_det():
    env = Environment("seaquest", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            enemy_lr, is_sub, enemy_y, diver_lr, diver_y = env.env.enemy_lr, env.env.is_sub, env.env.enemy_y, env.env.diver_lr, env.env.diver_y
            s_next_pgx = _step_det(
                minatar2pgx(s, seaquest.State),
                a,
                enemy_lr,
                is_sub,
                enemy_y,
                diver_lr,
                diver_y
            )
            assert jnp.allclose(
                env.state(),
                observe(s_next_pgx),
            )
            # if not jnp.allclose(env.state(), observe(s_next_pgx)):
            #     for field in fields(s_next_pgx):
            #         print(str(field.name) + "\n" + str(getattr(s_next_pgx, field.name)) + "\n"  + str(getattr(minatar2pgx(extract_state(env, state_keys), seaquest.MinAtarSeaquestState), field.name)))
            #     assert False

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        enemy_lr, is_sub, enemy_y, diver_lr, diver_y = env.env.enemy_lr, env.env.is_sub, env.env.enemy_y, env.env.diver_lr, env.env.diver_y
        s_next_pgx = _step_det(
            minatar2pgx(s, seaquest.State), a,
            enemy_lr,
            is_sub,
            enemy_y,
            diver_lr,
            diver_y
        )
        assert jnp.allclose(
            env.state(),
            observe(s_next_pgx),
        )
        # if not jnp.allclose(env.state(), observe(s_next_pgx)):
        #     for field in fields(s_next_pgx):
        #         print(str(field.name) + "\n" + str(getattr(s_next_pgx, field.name)) + "\n"  + str(getattr(minatar2pgx(extract_state(env, state_keys), seaquest.MinAtarSeaquestState), field.name)))
        #     assert False


def test_init_det():
    env = Environment("seaquest", sticky_action_prob=0.0)
    N = 100
    for _ in range(N):
        env.reset()
        s = extract_state(env, state_keys)
        s_pgx = _init_det()
        s_pgx2 = minatar2pgx(s, seaquest.State)
        for field in fields(s_pgx):
            assert jnp.allclose(getattr(s_pgx, field.name), getattr(s_pgx2, field.name))


def test_observe():
    env = Environment("seaquest", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 100
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, seaquest.State)
            obs_pgx = observe(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, seaquest.State)
        obs_pgx = observe(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )


def test_api():
    import pgx
    env = pgx.make("minatar/seaquest")
    pgx.api_test(env)


def test_buggy_sample():
    state = seaquest.State(
        oxygen=jnp.int32(187),
        diver_count=jnp.int32(0),
        sub_x=jnp.int32(0),
        sub_y=jnp.int32(3),
        sub_or=jnp.bool_(False),
        f_bullets= -jnp.ones(
            (5, 3), dtype=jnp.int32
        ),  #.at[0, :].set(jnp.int32([6, 6, 0])),
        e_bullets=-jnp.ones(
            (25, 3), dtype=jnp.int32
        ),
        e_fish = (-jnp.ones(
            (25, 4), dtype=jnp.int32
        )).at[:2, :].set(
            [[3, 8, 0, 2],
             [0, 4, 1, 4]]
        ),
        e_subs = (-jnp.ones(
            (25, 5), dtype=jnp.int32
        )).at[:2, :].set(
            jnp.int32(
                [[9, 1, 1, 0, 6],
                 [6, 6, 0, 3, 1]]
            )
        ),
        divers = (-jnp.ones(
            (5, 4), dtype=jnp.int32
        )).at[:2, :].set(
            [[3, 3, 0, 2],
             [1, 7, 1, 2]]
        ),
        e_spawn_speed=jnp.int32(19),
        e_spawn_timer=jnp.int32(18),
        d_spawn_timer=jnp.int32(21),
        move_speed=jnp.int32(5),
        ramp_index=jnp.int32(1),
        shot_timer=jnp.int32(0),
        surface=jnp.bool_(False),
        terminal=jnp.bool_(False),
        last_action=jnp.int32(4)
    )
    state = state.replace(observation=observe(state))
    # state.save_svg("tmp.svg")
    state = _step_det(state, 0,
                      enemy_lr=True,
                      is_sub=False,
                      enemy_y=4,
                      diver_lr=True,
                      diver_y=7
                      )
    # print(state.f_bullets)
    print("e_bullets")
    print(state.e_bullets)
    # print(state.e_fish)
    print("e_subs")
    print(state.e_subs)
    assert (state.e_bullets[0] == jnp.int32([-1, -1, -1])).all()
    assert (state.e_subs[0] == jnp.int32([6, 6, 0, 2, 0])).all()


def test_api():
    import pgx
    env = pgx.make("minatar/seaquest")
    pgx.api_test(env, 10)
