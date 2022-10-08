"""MinAtar/Seaquest: A fork of github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Tuple

import jax
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


@jax.jit
def _init_det() -> MinAtarSeaquestState:
    return MinAtarSeaquestState()