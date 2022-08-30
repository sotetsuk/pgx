"""MinAtar/Asterix: A form of github.com/kenjyoung/MinAtar

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

ramp_interval = 100
init_spawn_speed = 10
init_move_interval = 5
shot_cool_down = 5
INF = 1e5


@struct.dataclass
class MinAtarAsterixState:
    player_x: int = 5
    player_y: int = 5
    entities: jnp.ndarray = jnp.ones((8, 4), dtype=int) * INF
    shot_timer: int = 0
    spawn_speed: int = init_spawn_speed
    spawn_timer: int = init_spawn_speed
    move_speed: int = init_move_interval
    move_timer: int = init_move_interval
    ramp_timer: int = ramp_interval
    ramp_index: int = 0
    terminal: bool = False
    last_action: int = 0


def _step_det(
    state: MinAtarAsterixState,
    action: int,
    lr: False,
    is_gold: False,
    slot: int,
    ramping: bool = True,
) -> Tuple[MinAtarAsterixState, int, bool]:
    player_x = state.player_x
    player_y = state.player_y
    entities = state.entities
    shot_timer = state.shot_timer
    spawn_speed = state.spawn_speed
    spawn_timer = state.spawn_timer
    move_speed = state.move_speed
    move_timer = state.move_timer
    ramp_timer = state.ramp_timer
    ramp_index = state.ramp_index
    terminal = state.terminal
    last_action = action
    r = 0
    if terminal:
        return state, r, terminal

    # Spawn enemy if timer is up
    if spawn_timer == 0:
        entities = _spawn_entity(entities, lr, is_gold, slot)
        spawn_timer = spawn_speed

    # Resolve player action
    if action == 1:
        player_x = max(0, player_x - 1)
    elif action == 3:
        player_x = min(9, player_x + 1)
    elif action == 2:
        player_y = max(1, player_y - 1)
    elif action == 4:
        player_y = min(8, player_y + 1)

    # Update entities
    for i in range(len(entities)):
        x = entities[i]
        if x[0] != INF:
            if x[0:2] == [player_x, player_y]:
                if entities[i][3]:
                    entities[i, :].set(INF)
                    r += 1
                else:
                    terminal = True
    if move_timer == 0:
        move_timer = move_speed
        for i in range(len(entities)):
            x = entities[i]
            if x[0] != INF:
                x[0] += 1 if x[2] else -1
                if x[0] < 0 or x[0] > 9:
                    entities[i, :].set(INF)
                if x[0:2] == [player_x, player_y]:
                    if entities[i][3]:
                        entities[i, :].set(INF)
                        r += 1
                    else:
                        terminal = True

    # Update various timers
    spawn_timer -= 1
    move_timer -= 1

    # Ramp difficulty if interval has elapsed
    if ramping and (spawn_speed > 1 or move_speed > 1):
        if ramp_timer >= 0:
            ramp_timer -= 1
        else:
            if move_speed > 1 and ramp_index % 2:
                move_speed -= 1
            if spawn_speed > 1:
                spawn_speed -= 1
            ramp_index += 1
            ramp_timer = ramp_interval

    next_state = MinAtarAsterixState(
        player_x,
        player_y,
        entities,
        shot_timer,
        spawn_speed,
        spawn_timer,
        move_speed,
        move_timer,
        ramp_timer,
        ramp_index,
        terminal,
        last_action,
    )

    return next_state, r, terminal


# Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
def _spawn_entity(entities, lr, is_gold, slot):
    # lr = random.choice([True, False])
    # is_gold = random.choice([True, False], p=[1 / 3, 2 / 3])
    x = 0 if lr else 9
    slot_options = [i for i in range(len(entities)) if entities[i][0] == INF]
    if not slot_options:
        return
    # slot = random.choice(slot_options)
    entities[slot][0] = x
    entities[slot][1] = slot + 1
    entities[slot][2] = lr
    entities[slot][3] = is_gold
    return entities
