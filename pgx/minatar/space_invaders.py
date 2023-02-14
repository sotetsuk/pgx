"""MinAtar/SpaceInvaders: A fork of github.com/kenjyoung/MinAtar

https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Tuple

import jax
import jax.lax as lax
from jax import numpy as jnp

from pgx.flax.struct import dataclass

SHOT_COOL_DOWN = jnp.int8(5)
ENEMY_MOVE_INTERVAL = jnp.int8(12)
ENEMY_SHOT_INTERVAL = jnp.int8(10)

ZERO = jnp.int8(0)
NINE = jnp.int8(9)


@dataclass
class State:
    pos: jnp.ndarray = jnp.int8(5)
    f_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_)
    e_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_)
    alien_map: jnp.ndarray = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[0:4, 2:8].set(True)
    )
    alien_dir: jnp.ndarray = jnp.int8(-1)
    enemy_move_interval: jnp.ndarray = ENEMY_MOVE_INTERVAL
    alien_move_timer: jnp.ndarray = ENEMY_MOVE_INTERVAL
    alien_shot_timer: jnp.ndarray = ENEMY_SHOT_INTERVAL
    ramp_index: jnp.ndarray = jnp.int8(0)
    shot_timer: jnp.ndarray = jnp.int8(0)
    terminal: jnp.ndarray = jnp.bool_(False)
    last_action: jnp.ndarray = jnp.int8(0)


def step(
    state: State,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    action = jnp.int8(action)
    action = jax.lax.cond(
        jax.random.uniform(rng) < sticky_action_prob,
        lambda: state.last_action,
        lambda: action,
    )
    return _step_det(state, action)


def init(rng: jnp.ndarray) -> State:
    return _init_det()


def observe(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
    obs = obs.at[9, state.pos, 0].set(1)
    obs = obs.at[:, :, 1].set(state.alien_map)
    obs = obs.at[:, :, 2].set(
        lax.cond(
            state.alien_dir < 0,
            lambda: state.alien_map,
            lambda: jnp.zeros_like(state.alien_map),
        )
    )
    obs = obs.at[:, :, 3].set(
        lax.cond(
            state.alien_dir < 0,
            lambda: jnp.zeros_like(state.alien_map),
            lambda: state.alien_map,
        )
    )
    obs = obs.at[:, :, 4].set(state.f_bullet_map)
    obs = obs.at[:, :, 5].set(state.e_bullet_map)
    return obs


def _step_det(
    state: State,
    action: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    return lax.cond(
        state.terminal,
        lambda: (state.replace(last_action=action), jnp.int16(0), state.terminal),  # type: ignore
        lambda: _step_det_at_non_terminal(state, action),
    )


def _step_det_at_non_terminal(
    state: State,
    action: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    r = jnp.int16(0)

    pos = state.pos
    f_bullet_map = state.f_bullet_map
    e_bullet_map = state.e_bullet_map
    alien_map = state.alien_map
    alien_dir = state.alien_dir
    enemy_move_interval = state.enemy_move_interval
    alien_move_timer = state.alien_move_timer
    alien_shot_timer = state.alien_shot_timer
    ramp_index = state.ramp_index
    shot_timer = state.shot_timer
    terminal = state.terminal

    # Resolve player action
    # action_map = ['n','l','u','r','d','f']
    pos, f_bullet_map, shot_timer = _resole_action(
        pos, f_bullet_map, shot_timer, action
    )

    # Update Friendly Bullets
    f_bullet_map = jnp.roll(f_bullet_map, -1, axis=0)
    f_bullet_map = f_bullet_map.at[9, :].set(0)

    # Update Enemy Bullets
    e_bullet_map = jnp.roll(e_bullet_map, 1, axis=0)
    e_bullet_map = e_bullet_map.at[0, :].set(0)
    terminal = lax.cond(
        e_bullet_map[9, pos], lambda: jnp.bool_(True), lambda: terminal
    )

    # Update aliens
    terminal = lax.cond(
        alien_map[9, pos], lambda: jnp.bool_(True), lambda: terminal
    )
    alien_move_timer, alien_map, alien_dir, terminal = lax.cond(
        alien_move_timer == 0,
        lambda: _update_alien_by_move_timer(
            alien_map, alien_dir, enemy_move_interval, pos, terminal
        ),
        lambda: (alien_move_timer, alien_map, alien_dir, terminal),
    )
    timer_zero = alien_shot_timer == 0
    alien_shot_timer = lax.cond(
        timer_zero, lambda: ENEMY_SHOT_INTERVAL, lambda: alien_shot_timer
    )
    e_bullet_map = lax.cond(
        timer_zero,
        lambda: e_bullet_map.at[_nearest_alien(pos, alien_map)].set(1),
        lambda: e_bullet_map,
    )

    kill_locations = alien_map & (alien_map == f_bullet_map)

    r += jnp.sum(kill_locations, dtype=jnp.int16)
    alien_map = alien_map & (~kill_locations)
    f_bullet_map = f_bullet_map & (~kill_locations)

    # Update various timers
    shot_timer -= shot_timer > 0
    alien_move_timer -= 1
    alien_shot_timer -= 1
    ramping = True
    is_enemy_zero = jnp.count_nonzero(alien_map) == 0
    enemy_move_interval = lax.cond(
        is_enemy_zero & (enemy_move_interval > 6) & ramping,
        lambda: enemy_move_interval - 1,
        lambda: enemy_move_interval,
    )
    ramp_index = lax.cond(
        is_enemy_zero & (enemy_move_interval > 6) & ramping,
        lambda: ramp_index + 1,
        lambda: ramp_index,
    )
    alien_map = lax.cond(
        is_enemy_zero, lambda: alien_map.at[0:4, 2:8].set(1), lambda: alien_map
    )

    return (
        State(
            pos=pos,
            f_bullet_map=f_bullet_map,
            e_bullet_map=e_bullet_map,
            alien_map=alien_map,
            alien_dir=alien_dir,
            enemy_move_interval=enemy_move_interval,
            alien_move_timer=alien_move_timer,
            alien_shot_timer=alien_shot_timer,
            ramp_index=ramp_index,
            shot_timer=shot_timer,
            terminal=terminal,
            last_action=action,
        ),  # type: ignore
        r,
        terminal,
    )


def _resole_action(pos, f_bullet_map, shot_timer, action):
    f_bullet_map = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: f_bullet_map.at[9, pos].set(1),
        lambda: f_bullet_map,
    )
    shot_timer = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: SHOT_COOL_DOWN,
        lambda: shot_timer,
    )
    pos = lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - 1), lambda: pos
    )
    pos = lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + 1), lambda: pos
    )
    return pos, f_bullet_map, shot_timer


# TODO: avoid loop
def _nearest_alien(pos, alien_map):
    search_order = jnp.argsort(jnp.abs(jnp.arange(10, dtype=jnp.int8) - pos))
    ix = lax.while_loop(
        lambda i: jnp.sum(alien_map[:, search_order[i]]) <= 0,
        lambda i: i + 1,
        0,
    )
    ix = search_order[ix]
    j = lax.while_loop(lambda i: alien_map[i, ix] == 0, lambda i: i - 1, 9)
    return (j, ix)


def _update_alien_by_move_timer(
    alien_map, alien_dir, enemy_move_interval, pos, terminal
):
    alien_move_timer = lax.min(
        jnp.sum(alien_map, dtype=jnp.int8), enemy_move_interval
    )
    cond = ((jnp.sum(alien_map[:, 0]) > 0) & (alien_dir < 0)) | (
        (jnp.sum(alien_map[:, 9]) > 0) & (alien_dir > 0)
    )
    terminal = lax.cond(
        cond & (jnp.sum(alien_map[9, :]) > 0),
        lambda: jnp.bool_(True),
        lambda: terminal,
    )
    alien_dir = lax.cond(cond, lambda: -alien_dir, lambda: alien_dir)
    alien_map = lax.cond(
        cond,
        lambda: jnp.roll(alien_map, 1, axis=0),
        lambda: jnp.roll(alien_map, alien_dir, axis=1),
    )
    terminal = lax.cond(
        alien_map[9, pos], lambda: jnp.bool_(True), lambda: terminal
    )
    return alien_move_timer, alien_map, alien_dir, terminal


def _init_det() -> State:
    return State()
