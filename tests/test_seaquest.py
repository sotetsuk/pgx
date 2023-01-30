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
observe = jax.jit(seaquest.observe)


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
            s_next_pgx, _, _ = _step_det(
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
            # if not jnp.allclose(env.state(), seaquest.observe(s_next_pgx)):
            #     for field in fields(s_next_pgx):
            #         print(str(field.name) + "\n" + str(getattr(s_next_pgx, field.name)) + "\n"  + str(getattr(minatar2pgx(extract_state(env, state_keys), seaquest.MinAtarSeaquestState), field.name)))
            #     assert False

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        enemy_lr, is_sub, enemy_y, diver_lr, diver_y = env.env.enemy_lr, env.env.is_sub, env.env.enemy_y, env.env.diver_lr, env.env.diver_y
        s_next_pgx, _, _ = _step_det(
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
        # if not jnp.allclose(env.state(), seaquest.observe(s_next_pgx)):
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
