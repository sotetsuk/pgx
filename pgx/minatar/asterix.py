"""MinAtar/Asterix: A fork of github.com/kenjyoung/MinAtar

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

ramp_interval: jnp.ndarray = jnp.array(100, dtype=jnp.int8)
init_spawn_speed: jnp.ndarray = jnp.array(10, dtype=jnp.int8)
init_move_interval: jnp.ndarray = jnp.array(5, dtype=jnp.int8)
shot_cool_down: jnp.ndarray = jnp.array(5, dtype=jnp.int8)
INF: jnp.ndarray = jnp.array(99, dtype=jnp.int8)

ZERO = jnp.array(0, dtype=jnp.int8)
ONE = jnp.array(1, dtype=jnp.int8)
EIGHT = jnp.array(8, dtype=jnp.int8)
NINE = jnp.array(9, dtype=jnp.int8)


@struct.dataclass
class MinAtarAsterixState:
    player_x: jnp.ndarray = jnp.array(5, dtype=jnp.int8)
    player_y: jnp.ndarray = jnp.array(5, dtype=jnp.int8)
    entities: jnp.ndarray = jnp.ones((8, 4), dtype=jnp.int8) * INF
    shot_timer: jnp.ndarray = jnp.ones(0, dtype=jnp.int8)
    spawn_speed: jnp.ndarray = init_spawn_speed
    spawn_timer: jnp.ndarray = init_spawn_speed
    move_speed: jnp.ndarray = init_move_interval
    move_timer: jnp.ndarray = init_move_interval
    ramp_timer: jnp.ndarray = ramp_interval
    ramp_index: jnp.ndarray = jnp.array(0, dtype=jnp.int8)
    terminal: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    last_action: jnp.ndarray = jnp.array(0, dtype=jnp.int8)


@jax.jit
def step(
    state: MinAtarAsterixState,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[MinAtarAsterixState, jnp.ndarray, jnp.ndarray]:
    action = jnp.int8(action)
    rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
    # sticky action
    action = jax.lax.cond(
        jax.random.uniform(rng0) < sticky_action_prob,
        lambda: state.last_action,
        lambda: action,
    )

    lr = jax.random.choice(rng1, jnp.array([True, False]))
    is_gold = jax.random.choice(
        rng2, jnp.array([True, False]), p=jnp.array([1 / 3, 2 / 3])
    )
    slots = jnp.zeros((8))
    slots = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            state.entities[i, 0] == INF,
            lambda: x.at[i].set(1),
            lambda: x,
        ),
        slots,
    )
    slots = jax.lax.cond(
        slots.sum() == 0, lambda: slots.at[0].set(1), lambda: slots
    )
    p = slots / slots.sum()
    slot = jax.random.choice(rng3, jnp.arange(8), p=p)
    return _step_det(
        state,
        action,
        lr=lr,
        is_gold=is_gold,
        slot=slot,
    )


@jax.jit
def init(rng: jnp.ndarray) -> MinAtarAsterixState:
    return _init_det()


@jax.jit
def observe(state: MinAtarAsterixState) -> jnp.ndarray:
    return _to_obs(state)


@jax.jit
def _step_det(
    state: MinAtarAsterixState,
    action: jnp.ndarray,
    lr: bool,
    is_gold: bool,
    slot: int,
) -> Tuple[MinAtarAsterixState, jnp.ndarray, jnp.ndarray]:
    return jax.lax.cond(
        state.terminal,
        lambda: (state.replace(last_action=action), jnp.array(0, dtype=jnp.int16), True),  # type: ignore
        lambda: _step_det_at_non_terminal(state, action, lr, is_gold, slot),
    )


def _step_det_at_non_terminal(
    state: MinAtarAsterixState,
    action: jnp.ndarray,
    lr: bool,
    is_gold: bool,
    slot: int,
) -> Tuple[MinAtarAsterixState, jnp.ndarray, jnp.ndarray]:
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

    ramping: bool = True
    r = jnp.array(0, dtype=jnp.int16)

    # Spawn enemy if timer is up
    entities, spawn_timer = jax.lax.cond(
        spawn_timer == 0,
        lambda _entities, _spawn_timer: (
            entities.at[:, :].set(_spawn_entity(entities, lr, is_gold, slot)),
            spawn_speed,
        ),
        lambda _entities, _spawn_timer: (_entities, _spawn_timer),
        entities,
        spawn_timer,
    )

    # Resolve player action
    player_x, player_y = jax.lax.switch(
        action,
        [
            lambda: (player_x, player_y),  # 0
            lambda: (jax.lax.max(ZERO, player_x - 1), player_y),  # 1
            lambda: (player_x, jax.lax.max(ONE, player_y - 1)),  # 2
            lambda: (jax.lax.min(NINE, player_x + 1), player_y),  # 3
            lambda: (player_x, jax.lax.min(EIGHT, player_y + 1)),  # 4
            lambda: (player_x, player_y),  # 5
        ],
    )

    # Update entities
    entities, player_x, player_y, r, terminal = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            entities[i, 0] == INF,
            lambda: x,
            lambda: _update_entities(x[0], x[1], x[2], x[3], x[4], i),
        ),
        (entities, player_x, player_y, r, terminal),
    )

    entities, r, terminal = jax.lax.cond(
        move_timer == 0,
        lambda: _update_entities_by_timer(
            entities, r, terminal, player_x, player_y
        ),
        lambda: (entities, r, terminal),
    )
    move_timer = jax.lax.cond(
        move_timer == 0, lambda: move_speed, lambda: move_timer
    )

    # Update various timers
    spawn_timer -= 1
    move_timer -= 1

    # Ramp difficulty if interval has elapsed
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        ramping,
        lambda: _update_ramp(spawn_speed, move_speed, ramp_timer, ramp_index),
        lambda: (spawn_speed, move_speed, ramp_timer, ramp_index),
    )

    next_state = MinAtarAsterixState(
        player_x=player_x,
        player_y=player_y,
        entities=entities,
        shot_timer=shot_timer,
        spawn_speed=spawn_speed,
        spawn_timer=spawn_timer,
        move_speed=move_speed,
        move_timer=move_timer,
        ramp_timer=ramp_timer,
        ramp_index=ramp_index,
        terminal=terminal,
        last_action=action,
    )  # type: ignore
    return next_state, r, terminal


# Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
@jax.jit
def _spawn_entity(entities, lr, is_gold, slot):
    x = jax.lax.cond(lr == 1, lambda: 0, lambda: 9)
    new_entities = entities
    new_entities = new_entities.at[slot, 0].set(x)
    new_entities = new_entities.at[slot, 1].set(slot + 1)
    new_entities = new_entities.at[slot, 2].set(lr)
    new_entities = new_entities.at[slot, 3].set(is_gold)

    has_empty_slot = jnp.any(entities[:, 0] == INF)
    new_entities = jax.lax.cond(
        has_empty_slot,
        lambda: new_entities,
        lambda: entities,
    )
    return new_entities


@jax.jit
def _update_entities(entities, player_x, player_y, r, terminal, i):
    entities, r, terminal = jax.lax.cond(
        (entities[i, 0] == player_x) & (entities[i, 1] == player_y),
        lambda: jax.lax.cond(
            entities[i, 3] == 1,
            lambda: (entities.at[i, :].set(INF), r + 1, terminal),
            lambda: (entities, r, True),
        ),
        lambda: (entities, r, terminal),
    )
    return entities, player_x, player_y, r, terminal


@jax.jit
def _update_entities_by_timer(entities, r, terminal, player_x, player_y):
    entities, r, terminal = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            entities[i, 0] != INF,
            lambda: __update_entities_by_timer(
                x[0], x[1], x[2], player_x, player_y, i
            ),
            lambda: x,
        ),
        (entities, r, terminal),
    )
    return entities, r, terminal


@jax.jit
def __update_entities_by_timer(entities, r, terminal, player_x, player_y, i):
    entities = entities.at[i, 0].add(
        jax.lax.cond(entities[i, 2] == 1, lambda: 1, lambda: -1)
    )
    entities = jax.lax.cond(
        (entities[i, 0] < 0) | (entities[i, 0] > 9),
        lambda: entities.at[i, :].set(INF),
        lambda: entities,
    )
    entities, player_x, player_y, r, terminal = _update_entities(
        entities, player_x, player_y, r, terminal, i
    )
    return entities, r, terminal


@jax.jit
def _update_ramp(spawn_speed, move_speed, ramp_timer, ramp_index):
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        (spawn_speed > 1) | (move_speed > 1),
        lambda: jax.lax.cond(
            ramp_timer >= 0,
            lambda: (spawn_speed, move_speed, ramp_timer - 1, ramp_index),
            lambda: __update_ramp(spawn_speed, move_speed, ramp_index),
        ),
        lambda: (spawn_speed, move_speed, ramp_timer, ramp_index),
    )
    return spawn_speed, move_speed, ramp_timer, ramp_index


@jax.jit
def __update_ramp(spawn_speed, move_speed, ramp_index):
    move_speed = jax.lax.cond(
        (move_speed > 1) & (ramp_index % 2),
        lambda: move_speed - 1,
        lambda: move_speed,
    )
    spawn_speed = jax.lax.cond(
        spawn_speed > 1,
        lambda: spawn_speed - 1,
        lambda: spawn_speed,
    )
    ramp_index += 1
    ramp_timer = ramp_interval
    return spawn_speed, move_speed, ramp_timer, ramp_index


@jax.jit
def _init_det() -> MinAtarAsterixState:
    return MinAtarAsterixState()


@jax.jit
def _to_obs(state: MinAtarAsterixState) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state.player_y, state.player_x, 0].set(True)
    obs = jax.lax.fori_loop(
        0,
        8,
        lambda i, _obs: jax.lax.cond(
            state.entities[i, 0] != INF,
            lambda: _update_obs_by_entity(_obs, state, i),
            lambda: _obs,
        ),
        obs,
    )
    return obs


@jax.jit
def _update_obs_by_entity(obs, state, i):
    c = jax.lax.cond(state.entities[i, 3], lambda: 3, lambda: 1)
    obs = obs.at[state.entities[i, 1], state.entities[i, 0], c].set(True)
    back_x = jax.lax.cond(
        state.entities[i, 2],
        lambda: state.entities[i, 0] - 1,
        lambda: state.entities[i, 0] + 1,
    )
    obs = jax.lax.cond(
        (0 <= back_x) & (back_x <= 9),
        lambda: obs.at[state.entities[i, 1], back_x, 2].set(True),
        lambda: obs,
    )
    return obs
