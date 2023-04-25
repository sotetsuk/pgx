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
            assert r == s_next_pgx.reward[0]
            assert done == s_next_pgx.terminated


def test_init_det():
    env = Environment("seaquest", sticky_action_prob=0.0)
    N = 100
    for _ in range(N):
        env.reset()
        s = extract_state(env, state_keys)
        s_pgx = seaquest.State()
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
        _oxygen=jnp.int32(187),
        _diver_count=jnp.int32(0),
        _sub_x=jnp.int32(0),
        _sub_y=jnp.int32(3),
        _sub_or=jnp.bool_(False),
        _f_bullets=-jnp.ones(
            (5, 3), dtype=jnp.int32
        ),  #.at[0, :].set(jnp.int32([6, 6, 0])),
        _e_bullets=-jnp.ones(
            (25, 3), dtype=jnp.int32
        ),
        _e_fish=(-jnp.ones(
            (25, 4), dtype=jnp.int32
        )).at[:2, :].set(
            [[3, 8, 0, 2],
             [0, 4, 1, 4]]
        ),
        _e_subs=(-jnp.ones(
            (25, 5), dtype=jnp.int32
        )).at[:2, :].set(
            jnp.int32(
                [[9, 1, 1, 0, 6],
                 [6, 6, 0, 3, 1]]
            )
        ),
        _divers=(-jnp.ones(
            (5, 4), dtype=jnp.int32
        )).at[:2, :].set(
            [[3, 3, 0, 2],
             [1, 7, 1, 2]]
        ),
        _e_spawn_speed=jnp.int32(19),
        _e_spawn_timer=jnp.int32(18),
        _d_spawn_timer=jnp.int32(21),
        _move_speed=jnp.int32(5),
        _ramp_index=jnp.int32(1),
        _shot_timer=jnp.int32(0),
        _surface=jnp.bool_(False),
        _terminal=jnp.bool_(False),
        _last_action=jnp.int32(4)
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
    print(state._e_bullets)
    # print(state.e_fish)
    print("e_subs")
    print(state._e_subs)
    assert (state._e_bullets[0] == jnp.int32([-1, -1, -1])).all()
    assert (state._e_subs[0] == jnp.int32([6, 6, 0, 2, 0])).all()


def test_minimal_action_set():
    import pgx
    env = pgx.make("minatar/seaquest")
    assert env.num_actions == 6
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (6,)
    state = jax.jit(env.step)(state, 0)
    assert state.legal_action_mask.shape == (6,)


def test_api():
    import pgx
    env = pgx.make("minatar/seaquest")
    pgx.api_test(env, 10)
