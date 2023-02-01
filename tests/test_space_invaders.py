import jax
import random

from minatar import Environment

from pgx.minatar import space_invaders

from .minatar_utils import *

state_keys = [
    "pos",
    "f_bullet_map",
    "e_bullet_map",
    "alien_map",
    "alien_dir",
    "enemy_move_interval",
    "alien_move_timer",
    "alien_shot_timer",
    "ramp_index",
    "shot_timer",
    "terminal",
    "last_action",
]

_step_det = jax.jit(space_invaders._step_det)
_nearest_alien = jax.jit(space_invaders._nearest_alien)
_init_det = jax.jit(space_invaders._init_det)
observe = jax.jit(space_invaders.observe)


def test_neareset_alien():
    pos: jnp.ndarray = jnp.int8(3)
    alien_map: jnp.ndarray = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[0:4, 2:8].set(True)
    )
    alien_map = alien_map.at[3, 3].set(False)
    assert _nearest_alien(pos, alien_map) == (jnp.int8(2), jnp.int8(3))

def test_step_det():
    env = Environment("space_invaders", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = random.randrange(num_actions)
            r, done = env.act(a)
            s_next = extract_state(env, state_keys)
            s_next_pgx, _, _ = _step_det(
                minatar2pgx(s, space_invaders.State),
                a,
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))

        # check terminal state
        s = extract_state(env, state_keys)
        a = random.randrange(num_actions)
        r, done = env.act(a)
        s_next = extract_state(env, state_keys)
        s_next_pgx, _, _ = _step_det(
            minatar2pgx(s, space_invaders.State), a
        )
        assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))


def test_init_det():
    env = Environment("space_invaders", sticky_action_prob=0.0)
    N = 10
    for _ in range(N):
        env.reset()
        s = extract_state(env, state_keys)
        s_pgx = _init_det()
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_observe():
    env = Environment("space_invaders", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10  # TODO: increase N
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, space_invaders.State)
            obs_pgx = observe(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = random.randrange(num_actions)
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, space_invaders.State)
        obs_pgx = observe(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )
