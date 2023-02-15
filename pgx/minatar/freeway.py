"""MinAtar/Asterix: A fork of github.com/kenjyoung/MinAtar

https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Tuple

import jax
from jax import numpy as jnp

from pgx.flax.struct import dataclass

player_speed = jnp.array(3, dtype=jnp.int8)
time_limit = jnp.array(2500, dtype=jnp.int16)

ZERO = jnp.array(0, dtype=jnp.int8)
ONE = jnp.array(1, dtype=jnp.int8)
NINE = jnp.array(9, dtype=jnp.int8)


@dataclass
class State:
    cars: jnp.ndarray = jnp.zeros((8, 4), dtype=jnp.int8)
    pos: jnp.ndarray = jnp.array(9, dtype=jnp.int8)
    move_timer: jnp.ndarray = jnp.array(player_speed, dtype=jnp.int8)
    terminate_timer: jnp.ndarray = jnp.array(time_limit, dtype=jnp.int16)
    terminal: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    last_action: jnp.ndarray = jnp.array(0, dtype=jnp.int8)


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
    speeds, directions = _random_speed_directions(rng)
    return _step_det(state, action, speeds=speeds, directions=directions)


def init(rng: jnp.ndarray) -> State:
    speeds, directions = _random_speed_directions(rng)
    return _init_det(speeds=speeds, directions=directions)


def observe(state: State) -> jnp.ndarray:
    return _to_obs(state)


def _step_det(
    state: State,
    action: jnp.ndarray,
    speeds: jnp.ndarray,
    directions: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    return jax.lax.cond(
        state.terminal,
        lambda: (state.replace(last_action=action), jnp.array(0, dtype=jnp.int16), True),  # type: ignore
        lambda: _step_det_at_non_terminal(state, action, speeds, directions),
    )


def _step_det_at_non_terminal(
    state: State,
    action: jnp.ndarray,
    speeds: jnp.ndarray,
    directions: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:

    cars = state.cars
    pos = state.pos
    move_timer = state.move_timer
    terminate_timer = state.terminate_timer
    terminal = state.terminal
    last_action = action

    r = jnp.array(0, dtype=jnp.int16)

    move_timer, pos = jax.lax.cond(
        (action == 2) & (move_timer == 0),
        lambda: (player_speed, jax.lax.max(ZERO, pos - ONE)),
        lambda: (move_timer, pos),
    )
    move_timer, pos = jax.lax.cond(
        (action == 4) & (move_timer == 0),
        lambda: (player_speed, jax.lax.min(NINE, pos + ONE)),
        lambda: (move_timer, pos),
    )

    # Win condition
    cars, r, pos = jax.lax.cond(
        pos == 0,
        lambda: (
            _randomize_cars(speeds, directions, cars, initialize=False),
            r + 1,
            NINE,
        ),
        lambda: (cars, r, pos),
    )

    pos, cars = _update_cars(pos, cars)

    # Update various timers
    move_timer = jax.lax.cond(
        move_timer > 0, lambda: move_timer - 1, lambda: move_timer
    )
    terminate_timer -= ONE
    terminal = terminate_timer < 0

    next_state = State(
        cars,
        pos,
        move_timer,
        terminate_timer,
        terminal,
        last_action,
    )  # type: ignore

    return next_state, r, terminal


def _update_cars(pos, cars):
    def _update_stopped_car(pos, car):
        car = car.at[2].set(jax.lax.abs(car[3]))
        car = jax.lax.cond(
            car[3] > 0, lambda: car.at[0].add(1), lambda: car.at[0].add(-1)
        )
        car = jax.lax.cond(car[0] < 0, lambda: car.at[0].set(9), lambda: car)
        car = jax.lax.cond(car[0] > 9, lambda: car.at[0].set(0), lambda: car)
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        return pos, car

    def _update_car(pos, car):
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        pos, car = jax.lax.cond(
            car[2] == 0,
            lambda: _update_stopped_car(pos, car),
            lambda: (pos, car.at[2].add(-1)),
        )
        return pos, car

    pos, cars = jax.lax.scan(_update_car, pos, cars)

    return pos, cars


def _init_det(speeds: jnp.ndarray, directions: jnp.ndarray) -> State:
    cars = _randomize_cars(speeds, directions, initialize=True)
    return State(cars=cars)  # type: ignore


def _randomize_cars(
    speeds: jnp.ndarray,
    directions: jnp.ndarray,
    cars: jnp.ndarray = jnp.zeros((8, 4), dtype=int),
    initialize: bool = False,
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


def _random_speed_directions(rng):
    _, rng1, rng2 = jax.random.split(rng, 3)
    speeds = jax.random.randint(rng1, [8], 1, 6, dtype=jnp.int8)
    directions = jax.random.choice(
        rng2, jnp.array([-1, 1], dtype=jnp.int8), [8]
    )
    return speeds, directions


def _to_obs(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 7), dtype=jnp.bool_)
    obs = obs.at[state.pos, 4, 0].set(1)

    def _update_obs(i, _obs):
        car = state.cars[i]
        _obs = _obs.at[car[1], car[0], 1].set(1)
        back_x = jax.lax.cond(
            car[3] > 0, lambda: car[0] - 1, lambda: car[0] + 1
        )
        back_x = jax.lax.cond(back_x < 0, lambda: NINE, lambda: back_x)
        back_x = jax.lax.cond(back_x > 9, lambda: ZERO, lambda: back_x)
        trail = jax.lax.abs(car[3]) + 1
        _obs = _obs.at[car[1], back_x, trail].set(1)
        return _obs

    obs = jax.lax.fori_loop(0, 8, _update_obs, obs)
    return obs
