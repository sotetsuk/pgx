"""MinAtar/Asterix: A form of github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Tuple

import random
import jax
from flax import struct
from jax import numpy as jnp

player_speed = 3
time_limit = 2500


@struct.dataclass
class MinAtarFreewayState:
    cars: jnp.ndarray = jnp.ones((8, 4), dtype=int)
    pos: int = 9
    move_timer: int = player_speed
    terminate_timer: int = time_limit
    terminal: bool = False
    last_action: int = 0


# TODO: make me jit
def _det_step_det(state: MinAtarFreewayState, action: jnp.ndarray, cars: jnp.ndarray) -> MinAtarFreewayState:
    cars = state.cars
    pos = state.pos
    move_timer = state.move_timer
    terminate_timer = state.terminate_timer
    terminal = state.terminal
    last_action = action

    r = 0
    if terminal:
        return r, terminal

    a = action_map[a]

    if a == "u" and move_timer == 0:
        move_timer = player_speed
        pos = max(0, pos - 1)
    elif a == "d" and move_timer == 0:
        move_timer = player_speed
        pos = min(9, pos + 1)

    # Win condition
    if pos == 0:
        r += 1
        _randomize_cars(initialize=False)
        pos = 9

    # Update cars
    for car in cars:
        if car[0:2] == [4, pos]:
            pos = 9
        if car[2] == 0:
            car[2] = abs(car[3])
            car[0] += 1 if car[3] > 0 else -1
            if car[0] < 0:
                car[0] = 9
            elif car[0] > 9:
                car[0] = 0
            if car[0:2] == [4, pos]:
                pos = 9
        else:
            car[2] -= 1

    # Update various timers
    move_timer -= move_timer > 0
    terminate_timer -= 1
    if terminate_timer < 0:
        terminal = True
    return r, terminal

# TODO: make me jit
def _to_obs(
    state = np.zeros((10, 10, len(channels)), dtype=bool)
    state[pos, 4, channels["chicken"]] = 1
    for car in cars:
        state[car[1], car[0], channels["car"]] = 1
        back_x = car[0] - 1 if car[3] > 0 else car[0] + 1
        if back_x < 0:
            back_x = 9
        elif back_x > 9:
            back_x = 0
        if abs(car[3]) == 1:
            trail = channels["speed1"]
        elif abs(car[3]) == 2:
            trail = channels["speed2"]
        elif abs(car[3]) == 3:
            trail = channels["speed3"]
        elif abs(car[3]) == 4:
            trail = channels["speed4"]
        elif abs(car[3]) == 5:
            trail = channels["speed5"]
        state[car[1], back_x, trail] = 1
    return state

# TODO: make me jit
def _randomize_cars(cars: jnp.ndarray, initialize=False) -> jnp.ndarray:
    speeds = random.randint(1, 6, 8)
    directions = random.choice([-1, 1], 8)
    speeds *= directions
    if initialize:
        cars = []
        for i in range(8):
            cars += [[0, i + 1, abs(speeds[i]), speeds[i]]]
        cars = jnp.array(cars)
    else:
        for i in range(8):
            cars = cars.at[i, 2].set(abs(speeds[i]))
            cars = cars.at[i, 3].set(speeds[i])
            # self.cars[i][2:4]=[abs(speeds[i]),speeds[i]]
    return cars
