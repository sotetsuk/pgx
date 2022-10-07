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
from flax import struct
from jax import numpy as jnp

shot_cool_down = jnp.int8(5)
enemy_move_interval = jnp.int8(12)
enemy_shot_interval = jnp.int8(10)

@struct.dataclass
class MinAtarSpaceInvadersState:
    pos: jnp.ndarray = jnp.int8(5),
    f_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_),
    e_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_),
    alien_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_).at[0:4,2:8].set(True),
    alien_dir: jnp.ndarray = jnp.int8(-1),
    enemy_move_interval: jnp.ndarray = enemy_move_interval,
    alien_move_timer: jnp.ndarray = enemy_move_interval,
    alien_shot_timer: jnp.ndarray = enemy_shot_interval,
    ramp_index: jnp.ndarray = jnp.int8(0),
    shot_timer: jnp.ndarray = jnp.int8(0),
    terminal: jnp.ndarray = jnp.bool_(False),
    last_action: jnp.ndarray = jnp.int8(0),


@jax.jit
def step(
    state: MinAtarSpaceInvadersState,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[MinAtarSpaceInvadersState, jnp.ndarray, jnp.ndarray]:
    ...


@jax.jit
def init(rng: jnp.ndarray) -> MinAtarSpaceInvadersState:
    return _init_det()


@jax.jit
def observe(state: MinAtarSpaceInvadersState) -> jnp.ndarray:
    ...

@jax.jit
def _init_det() -> MinAtarSpaceInvadersState:
    return MinAtarSpaceInvadersState()
