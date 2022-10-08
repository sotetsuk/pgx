"""MinAtar/Seaquest: A fork of github.com/kenjyoung/MinAtar

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


RAMP_INTERVAL: jnp.ndarray = jnp.int8(100)
MAX_OXYGEN: jnp.ndarray = jnp.int16(200)
INIT_SPAWN_SPEED: jnp.ndarray = jnp.int8(20)
DIVER_SPAWN_SPEED: jnp.ndarray = jnp.int8(30)
INIT_MOVE_INTERVAL: jnp.ndarray = jnp.int8(5)
SHOT_COOL_DOWN: jnp.ndarray = jnp.int8(5)
ENEMY_SHOT_INTERVAL: jnp.ndarray = jnp.int8(10)
ENEMY_MOVE_INTERVAL: jnp.ndarray = jnp.int8(5)
DIVER_MORE_INTERVAL: jnp.ndarray = jnp.int8(5)


ZERO: jnp.ndarray = jnp.int8(0)
TRUE: jnp.ndarray = jnp.bool_(True)
FALSE: jnp.ndarray = jnp.bool_(False)


@struct.dataclass
class MinAtarSeaquestState:
    oxygen: jnp.ndarray = MAX_OXYGEN
    diver_count: jnp.ndarray = ZERO
    sub_x: jnp.ndarray = jnp.int8(5)
    sub_y: jnp.ndarray = ZERO
    sub_or: jnp.ndarray = FALSE
    f_bullets: jnp.ndarray = - jnp.ones((5, 3), dtype=jnp.int8)  # <= 2  TODO: confirm
    e_bullets: jnp.ndarray = - jnp.ones((25, 3), dtype=jnp.int8)  # <= 1 per each sub  TODO: confirm
    e_fish: jnp.ndarray = - jnp.ones((25, 4), dtype=jnp.int8)  # <= 19  TODO: confirm
    e_subs: jnp.ndarray = - jnp.ones((25, 4), dtype=jnp.int8)  # <= 19  TODO: confirm
    divers: jnp.ndarray = - jnp.ones((5, 4), dtype=jnp.int8)  # <= 2  TODO: confirm
    e_spawn_speed: jnp.ndarray = INIT_SPAWN_SPEED
    e_spawn_timer: jnp.ndarray = INIT_SPAWN_SPEED
    d_spawn_timer: jnp.ndarray = DIVER_SPAWN_SPEED
    move_speed: jnp.ndarray = INIT_MOVE_INTERVAL
    ramp_index: jnp.ndarray = ZERO  # TODO: require int16?
    shot_timer: jnp.ndarray = ZERO
    surface: jnp.ndarray = TRUE
    terminal: jnp.ndarray = FALSE
    last_action: jnp.ndarray = ZERO


@jax.jit
def init(rng: jnp.ndarray) -> MinAtarSeaquestState:
    return _init_det()


def observe(state: MinAtarSeaquestState) -> jnp.ndarray:
    """
            self.channels ={
            'sub_front':0,
            'sub_back':1,
            'friendly_bullet':2,
            'trail':3,
            'enemy_bullet':4,
            'enemy_fish':5,
            'enemy_sub':6,
            'oxygen_guage':7,
            'diver_guage':8,
            'diver':9
        }
    """
    obs = jnp.zeros((10, 10, 10), dtype=jnp.bool_)
    obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
    back_x = lax.cond(state.sub_or, lambda: state.sub_x - 1, lambda: state.sub_x + 1)
    obs = obs.at[state.sub_y, back_x, 1].set(1)
    obs = obs.at[9, 0:state.oxygen * 10 // MAX_OXYGEN, 7].set(1)
    obs = obs.at[9, 9 - state.diver_count:9, 8].set(1)
    obs = lax.fori_loop(
        0, 5, lambda i, _obs : lax.cond(
            state.f_bullets[i][0] >= 0,
            lambda: _obs.at[state.f_bullets[i][1], state.f_bullets[i][0], 2].set(1),
            lambda: _obs
        ),
        obs
    )
    # for bullet in state.f_bullets:
    #     if bullet[0] == -1:
    #         continue
    #     obs = obs.at[bullet[1], bullet[0], 2].set(1)
    obs = lax.fori_loop(
        0, 25, lambda i, _obs: lax.cond(
            state.e_bullets[i][0] >= 0,
            lambda: _obs.at[state.e_bullets[i][1], state.e_bullets[i][0], 4].set(1),
            lambda: _obs
        ), obs
    )
    # for bullet in state.e_bullets:
    #     if bullet[0] == -1:
    #         continue
    #     obs = obs.at[bullet[1], bullet[0], 4].set(1)

    def set_e_fish(_obs, fish):
        _obs = _obs.at[fish[1], fish[0], 5].set(1)
        back_x = fish[0] + jnp.array([1, -1])[fish[2]]
        # back_x = fish[0] - 1 if fish[2] else fish[0] + 1
        _obs = lax.cond(
            (0 <= back_x) & (back_x <= 9),
            lambda: _obs.at[fish[1], back_x, 3].set(1),
            lambda: _obs,
        )
        # if (back_x >= 0 and back_x <= 9):
        #     _obs = _obs.at[fish[1], back_x, 3].set(1)
        return _obs

    obs = lax.fori_loop(
        0, 25, lambda i, _obs : lax.cond(
            state.e_fish[i][0] >= 0,
            lambda: set_e_fish(_obs, state.e_fish[i]),
            lambda: _obs
        ), obs
    )
    # for fish in state.e_fish:
    #     if fish[0] == -1:
    #         continue
    #     obs = obs.at[fish[1], fish[0], 5].set(1)
    #     back_x = fish[0] - 1 if fish[2] else fish[0] + 1
    #     if (back_x >= 0 and back_x <= 9):
    #         obs = obs.at[fish[1], back_x, 3].set(1)
    for sub in state.e_subs:
        if sub[0] == -1:
            continue
        obs = obs.at[sub[1], sub[0], 6].set(1)
        back_x = sub[0] - 1 if sub[2] else sub[0] + 1
        if (back_x >= 0 and back_x <= 9):
            obs = obs.at[sub[1], back_x, 3].set(1)
    for diver in state.divers:
        if diver[0] == -1:
            continue
        obs = obs.at[diver[1], diver[0], 9].set(1)
        back_x = diver[0] - 1 if diver[2] else diver[0] + 1
        if (back_x >= 0 and back_x <= 9):
            obs = obs.at[diver[1], back_x, 3].set(1)

    return obs


@jax.jit
def _init_det() -> MinAtarSeaquestState:
    return MinAtarSeaquestState()