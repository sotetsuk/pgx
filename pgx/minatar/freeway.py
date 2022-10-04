"""MinAtar/Asterix: A form of github.com/kenjyoung/MinAtar

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

player_speed = 3
time_limit = 2500


@struct.dataclass
class MinAtarFreewayState:
    cars: jnp.ndarray = jnp.zeros((8, 4), dtype=int)
    pos: int = 9
    move_timer: int = player_speed
    terminate_timer: int = time_limit
    terminal: bool = False
    last_action: int = 0


# TODO: make me @jax.jit
def step(
    state: MinAtarFreewayState,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[MinAtarFreewayState, int, bool]:
    if jax.random.uniform(rng) < sticky_action_prob:
        action = state.last_action  # type: ignore
    speeds, directions = _random_speed_directions(rng)
    return _step_det(state, action, speeds=speeds, directions=directions)


# TODO: make me  @jax.jit
def reset(rng: jnp.ndarray) -> MinAtarFreewayState:
    speeds, directions = _random_speed_directions(rng)
    return _reset_det(speeds=speeds, directions=directions)


# TODO: make me  @jax.jit
def to_obs(state: MinAtarFreewayState) -> jnp.ndarray:
    return _to_obs(state)


# TODO: make me  @jax.jit
def _step_det(
    state: MinAtarFreewayState,
    action: jnp.ndarray,
    speeds: jnp.ndarray,
    directions: jnp.ndarray,
) -> Tuple[MinAtarFreewayState, int, bool]:

    cars = state.cars
    pos = state.pos
    move_timer = state.move_timer
    terminate_timer = state.terminate_timer
    terminal = state.terminal
    last_action = action

    r = 0
    if terminal:
        next_state = MinAtarFreewayState(
            cars,
            pos,
            move_timer,
            terminate_timer,
            terminal,
            last_action,
        )  # type: ignore

        return next_state, r, terminal

    # self.action_map = ['n','l','u','r','d','f']
    if action == 2 and move_timer == 0:
        move_timer = player_speed
        pos = max(0, pos - 1)
    elif action == 4 and move_timer == 0:
        move_timer = player_speed
        pos = min(9, pos + 1)

    # Win condition
    if pos == 0:
        r += 1
        cars = _randomize_cars(speeds, directions, cars, initialize=False)
        pos = 9

    # Update cars
    # TDOO: remove me
    cars = [[cars[i, j] for j in range(4)] for i in range(8)]  # type: ignore
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
    cars = jnp.array(cars)  # TDOO: remove me

    # Update various timers
    move_timer -= move_timer > 0
    terminate_timer -= 1
    if terminate_timer < 0:
        terminal = True

    next_state = MinAtarFreewayState(
        cars,
        pos,
        move_timer,
        terminate_timer,
        terminal,
        last_action,
    )  # type: ignore

    return next_state, r, terminal


@jax.jit
def _reset_det(
    speeds: jnp.ndarray, directions: jnp.ndarray
) -> MinAtarFreewayState:
    cars = _randomize_cars(speeds, directions, initialize=True)
    return MinAtarFreewayState(cars=cars)  # type: ignore


@jax.jit
def _randomize_cars(
    speeds: jnp.ndarray,
    directions: jnp.ndarray,
    cars: jnp.ndarray = jnp.zeros((8, 4), dtype=int),
    initialize: bool = False
) -> jnp.ndarray:
    assert isinstance(cars, jnp.ndarray), cars
    speeds *= directions

    def _init(_cars):
        _cars = _cars.at[:, 1].set(jnp.arange(1, 9))
        _cars = _cars.at[:, 2].set(jax.lax.abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    def _update(_cars):
        _cars = _cars.at[:, 2].set(abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    return jax.lax.cond(initialize, _init, _update, cars)


# TODO: make me  @jax.jit
def _random_speed_directions(rng):
    # TDOO: use jnp instead
    # TDOO: rng must be splitted
    import numpy as np

    speeds = np.random.randint(1, 6, 8)
    directions = np.random.choice([-1, 1], 8)
    return speeds, directions


def _to_obs(state: MinAtarFreewayState) -> jnp.ndarray:
    import numpy as np  # TODO: remove me

    obs = np.zeros((10, 10, 7), dtype=bool)
    obs[state.pos, 4, 0] = 1
    for car in state.cars:
        obs[car[1], car[0], 1] = 1
        back_x = car[0] - 1 if car[3] > 0 else car[0] + 1
        if back_x < 0:
            back_x = 9
        elif back_x > 9:
            back_x = 0
        if abs(car[3]) == 1:
            trail = 2
        elif abs(car[3]) == 2:
            trail = 3
        elif abs(car[3]) == 3:
            trail = 4
        elif abs(car[3]) == 4:
            trail = 5
        elif abs(car[3]) == 5:
            trail = 6
        obs[car[1], back_x, trail] = 1

    return jnp.array(obs)
