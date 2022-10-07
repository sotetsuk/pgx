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
from flax import struct
from jax import numpy as jnp

SHOT_COOL_DOWN = jnp.int8(5)
ENEMY_MOVE_INTERVAL = jnp.int8(12)
ENEMY_SHOT_INTERVAL = jnp.int8(10)


@struct.dataclass
class MinAtarSpaceInvadersState:
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


# @jax.jit
def step(
    state: MinAtarSpaceInvadersState,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[MinAtarSpaceInvadersState, jnp.ndarray, jnp.ndarray]:
    action = jax.lax.cond(
        jax.random.uniform(rng) < sticky_action_prob,
        lambda: state.last_action,
        lambda: action,
    )
    return _step_det(state, action)


@jax.jit
def init(rng: jnp.ndarray) -> MinAtarSpaceInvadersState:
    return _init_det()


@jax.jit
def observe(state: MinAtarSpaceInvadersState) -> jnp.ndarray:
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


# @jax.jit
def _step_det(
    state: MinAtarSpaceInvadersState,
    action: jnp.ndarray,
) -> Tuple[MinAtarSpaceInvadersState, jnp.ndarray, jnp.ndarray]:
    if state.terminal:
        return state.replace(last_action=action), jnp.int16(0), state.terminal  # type: ignore
    else:
        return _step_det_at_non_terminal(state, action)


# @jax.jit
def _step_det_at_non_terminal(
    state: MinAtarSpaceInvadersState,
    action: jnp.ndarray,
) -> Tuple[MinAtarSpaceInvadersState, jnp.ndarray, jnp.ndarray]:
    r = 0

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
    pos, f_bullet_map, shot_timer = _resole_action(pos, f_bullet_map, shot_timer, action)

    # Update Friendly Bullets
    f_bullet_map = jnp.roll(f_bullet_map, -1, axis=0)
    f_bullet_map = f_bullet_map.at[9, :].set(0)

    # Update Enemy Bullets
    e_bullet_map = jnp.roll(e_bullet_map, 1, axis=0)
    e_bullet_map = e_bullet_map.at[0, :].set(0)
    if e_bullet_map[9, pos]:
        terminal = jnp.bool_(True)

    # Update aliens
    if alien_map[9, pos]:
        terminal = jnp.bool_(True)
    if alien_move_timer == 0:
        alien_move_timer = min(
            jnp.count_nonzero(alien_map), enemy_move_interval
        )
        if (jnp.sum(alien_map[:, 0]) > 0 and alien_dir < 0) or ( jnp.sum(alien_map[:, 9]) > 0 and alien_dir > 0 ):
            alien_dir = -alien_dir
            if jnp.sum(alien_map[9, :]) > 0:
                terminal = jnp.bool_(True)
            alien_map = jnp.roll(alien_map, 1, axis=0)
        else:
            alien_map = jnp.roll(alien_map, alien_dir, axis=1)
        if alien_map[9, pos]:
            terminal = jnp.bool_(True)
    if alien_shot_timer == 0:
        alien_shot_timer = ENEMY_SHOT_INTERVAL
        nearest_alien = _nearest_alien(pos, alien_map)
        e_bullet_map = e_bullet_map.at[nearest_alien[0], nearest_alien[1]].set(
            1
        )

    kill_locations = jnp.logical_and(alien_map, alien_map == f_bullet_map)

    r += jnp.sum(kill_locations)
    alien_map = alien_map.at[kill_locations].set(0)
    f_bullet_map = f_bullet_map.at[kill_locations].set(0)

    # Update various timers
    shot_timer -= shot_timer > 0
    alien_move_timer -= 1
    alien_shot_timer -= 1
    ramping = True
    if jnp.count_nonzero(alien_map) == 0:
        if enemy_move_interval > 6 and ramping:
            enemy_move_interval -= 1
            ramp_index += 1
        alien_map = alien_map.at[0:4, 2:8].set(1)

    return (
        MinAtarSpaceInvadersState(
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
    if action == 5 and shot_timer == 0:
        f_bullet_map = f_bullet_map.at[9, pos].set(1)
        shot_timer = SHOT_COOL_DOWN
    elif action == 1:
        pos = max(0, pos - 1)
    elif action == 3:
        pos = min(9, pos + 1)
    return pos, f_bullet_map, shot_timer


def _nearest_alien(pos, alien_map):
    search_order = [i for i in range(10)]
    search_order.sort(key=lambda x: abs(x - pos))
    for i in search_order:
        if jnp.sum(alien_map[:, i]) > 0:
            return [jnp.max(jnp.arange(10)[alien_map[:, i]]), i]


@jax.jit
def _init_det() -> MinAtarSpaceInvadersState:
    return MinAtarSpaceInvadersState()
