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


@struct.dataclass
class MinAtarAsterixState:
    player_x: int = 5
    player_y: int = 5
    entities: jnp.ndarray = jnp.ones((8, 4), dtype=int) * 1e5
    shot_timer: int = 0
    spawn_speed: int = init_spawn_speed
    spawn_timer: int = init_spawn_speed
    move_speed: int = init_move_interval
    move_timer: int = init_move_interval
    ramp_timer: int = ramp_interval
    ramp_index: int = 0
    terminal: bool = False
    last_action: int = 0


@jax.jit
def _step_det(
    state: MinAtarAsterixState,
    action: int,
    lr: bool,
    is_gold: bool,
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

    terminal_state = MinAtarAsterixState(
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
    )  # type: ignore

    r = 0
    # if terminal:
    #     return state, r, terminal

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
    # if spawn_timer == 0:
    #     entities = _spawn_entity()
    #     spawn_timer = spawn_speed

    # Resolve player action
    player_x, player_y = jax.lax.switch(
        action,
        [
            lambda x, y: (x, y),  # 0
            lambda x, y: (x - 1, y),  # 1
            lambda x, y: (x, y - 1),  # 2
            lambda x, y: (x + 1, y),  # 3
            lambda x, y: (x, y + 1),  # 4
            lambda x, y: (x, y),  # 5
        ],
        player_x,
        player_y,
    )
    player_x = jax.lax.max(0, player_x)
    player_x = jax.lax.min(9, player_x)
    player_y = jax.lax.max(1, player_y)
    player_y = jax.lax.min(8, player_y)
    # if action == 1:
    #     player_x = max(0, player_x - 1)
    # elif action == 3:
    #     player_x = min(9, player_x + 1)
    # elif action == 2:
    #     player_y = max(1, player_y - 1)
    # elif action == 4:
    #     player_y = min(8, player_y + 1)

    # Update entities
    for i in range(8):
        entities, player_x, player_y, r, terminal = jax.lax.cond(
            entities[i, 0] == 1e5,
            lambda _entities, _player_x, _player_y, _r, _terminal: (
                _entities,
                _player_x,
                _player_y,
                _r,
                _terminal,
            ),
            lambda _entities, _player_x, _player_y, _r, _terminal: _update_entities(
                _entities, _player_x, _player_y, _r, _terminal, i
            ),
            entities,
            player_x,
            player_y,
            r,
            terminal,
        )
    # for i in range(len(entities)):
    #     x = entities[i]
    #     if x[0] != 1e5:
    #         if x[0] == player_x and x[1] == player_y:
    #             if entities[i, 3] == 1:
    #                 entities = entities.at[i, :].set(1e5)
    #                 r += 1
    #             else:
    #                 terminal = True
    move_timer, entities, r, terminal = jax.lax.cond(
        move_timer == 0,
        lambda _move_timer, _entities, _r, _terminal: (
            move_speed,
            *_update_entities_by_timer(_entities, _r, _terminal),
        ),
        lambda _move_timer, _entities, _r, _terminal: (
            _move_timer,
            _entities,
            _r,
            _terminal,
        ),
        move_timer,
        entities,
        r,
        terminal,
    )
    # if move_timer == 0:
    #     move_timer = move_speed
    #     entities, r, terminal = _update_entities_by_timer(
    #         entities, r, terminal, player_x, player_y
    #     )

    # Update various timers
    spawn_timer -= 1
    move_timer -= 1

    # Ramp difficulty if interval has elapsed
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        ramping,
        _update_ramp,
        lambda _x: _x,
        spawn_speed,
        move_speed,
        ramp_timer,
        ramp_index,
    )
    # if ramping:
    #     spawn_speed, move_speed, ramp_timer, ramp_index = _update_ramp(
    #         spawn_speed, move_speed, ramp_timer, ramp_index
    #     )

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
    )  # type: ignore

    next_state, r, terminal = jax.lax.cond(
        state.terminal,
        lambda _next_state, _r, _terminal: (terminal_state, 0, True),
        lambda _next_state, _r, _terminal: (next_state, r, terminal),
        next_state,
        r,
        terminal,
    )

    return next_state, r, terminal


# Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
@jax.jit
def _spawn_entity(entities, lr, is_gold, slot):
    # lr = random.choice([True, False])
    # is_gold = random.choice([True, False], p=[1 / 3, 2 / 3])
    x = 0
    x = jax.lax.cond(lr == 1, lambda _: 0, lambda _: 9, x)
    # x = 0 if lr else 9
    # slot_options = [i for i in range(len(entities)) if entities[i][0] == 1e5]
    # if not slot_options:
    #     return entities
    # slot = random.choice(slot_options)
    new_entities = entities
    new_entities = new_entities.at[slot, 0].set(x)
    new_entities = new_entities.at[slot, 1].set(slot + 1)
    new_entities = new_entities.at[slot, 2].set(lr)
    new_entities = new_entities.at[slot, 3].set(is_gold)

    has_empty_slot = False
    for i in range(8):
        has_empty_slot = jax.lax.cond(
            entities[i][0] == 1e5,
            lambda z: True,
            lambda z: z,
            has_empty_slot,
        )
    new_entities = jax.lax.cond(
        has_empty_slot,
        lambda _: new_entities,
        lambda _: entities,
        new_entities,
    )

    return new_entities


@jax.jit
def _update_entities(entities, player_x, player_y, r, terminal, i):
    entities, r, terminal = jax.lax.cond(
        entities[i, 0] == player_x,
        lambda _entities, _r, _terminal: jax.lax.cond(
            entities[i, 1] == player_y,
            lambda __entities, __r, __terminal: jax.lax.cond(
                entities[i, 3] == 1,
                lambda ___entities, ___r, ___terminal: (
                    ___entities.at[i, :].set(1e5),
                    ___r + 1,
                    ___terminal,
                ),
                lambda ___entities, ___r, ___terminal: (
                    ___entities,
                    ___r,
                    True,
                ),
                __entities,
                __r,
                __terminal,
            ),
            lambda __entities, __r, __terminal: (__entities, __r, __terminal),
            _entities,
            _r,
            _terminal,
        ),
        lambda _entities, _r, _terminal: (_entities, _r, _terminal),
        entities,
        r,
        terminal,
    )
    # if entities[i, 0] == player_x and entities[i, 1] == player_y:
    #     if entities[i, 3] == 1:
    #         entities = entities.at[i, :].set(1e5)
    #         r += 1
    #     else:
    #         terminal = True
    return entities, player_x, player_y, r, terminal


@jax.jit
def _update_entities_by_timer(entities, r, terminal, player_x, player_y):
    for i in range(8):
        entities, r, terminal = jax.lax.cond(
            entities[i, 0] != 1e5,
            __update_entities_by_timer,
            lambda _entities, _r, _terminal: (_entities, _r, _terminal),
            entities,
            r,
            terminal,
        )
        # if entities[i, 0] != 1e5:
        #     entities, r, terminal = __update_entities_by_timer(
        #         entities, r, terminal, player_x, player_y, i
        #     )
    return entities, r, terminal


@jax.jit
def __update_entities_by_timer(entities, r, terminal, player_x, player_y, i):
    entities = jax.lax.cond(
        entities[i, 2] == 1,
        lambda _entities: _entities.at[i, 0].set(_entities[i, 0] + 1),
        lambda _entities: _entities.at[i, 0].set(_entities[i, 0] - 1),
        entities,
    )
    # x[0]+=1 if x[2] else -1
    entities = jax.lax.cond(
        entities[i, 0] < 0,
        lambda _entities: entities.at[i, :].set(1e5),
        lambda _entities: _entities,
        entities,
    )
    entities = jax.lax.cond(
        entities[i, 0] > 9,
        lambda _entities: entities.at[i, :].set(1e5),
        lambda _entities: _entities,
        entities,
    )
    # if entities[i, 0] < 0 or entities[i, 0] > 9:
    #     entities = entities.at[i, :].set(1e5)
    entities, r, terminal = jax.lax.cond(
        entities[i, 0] == player_x,
        lambda _entities, _r, _terminal: jax.lax.cond(
            entities[i, 1] == player_y,
            lambda __entities, __r, __terminal: jax.lax.cond(
                entities[i, 3] == 1,
                lambda ___entities, ___r, ___terminal: (
                    entities.at[i, :].set(1e5),
                    ___r + 1,
                    ___terminal,
                ),
                lambda ___entities, ___r, ___terminal: (
                    ___entities,
                    ___r,
                    True,
                ),
                __entities,
                __r,
                __terminal,
            ),
            lambda __entities, __r, __terminal: (__entities, __r, __terminal),
            _entities,
            _r,
            _terminal,
        ),
        lambda _entities, _r, _terminal: (_entities, _r, _terminal),
        entities,
        r,
        terminal,
    )
    # if entities[i, 0] == player_x and entities[i, 1] == player_y:
    #     if entities[i, 3] == 1:
    #         entities = entities.at[i, :].set(1e5)
    #         r += 1
    #     else:
    #         terminal = True
    return entities, r, terminal


@jax.jit
def _update_ramp(spawn_speed, move_speed, ramp_timer, ramp_index):
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        spawn_speed > 1,
        lambda _spawn_speed, _move_speed, _ramp_timer, _ramp_index: jax.lax.cond(
            _ramp_timer >= 0,
            lambda __spawn_speed, __move_speed, __ramp_timer, __ramp_index: (
                __spawn_speed,
                __move_speed,
                __ramp_timer - 1,
                __ramp_index,
            ),
            __update_ramp,
            _spawn_speed,
            _move_speed,
            _ramp_timer,
            _ramp_index,
        ),
        lambda _spawn_speed, _move_speed, _ramp_timer, _ramp_index: jax.lax.cond(
            move_speed > 1,
            lambda __spawn_speed, __move_speed, __ramp_timer, __ramp_index: jax.lax.cond(
                __ramp_timer >= 0,
                lambda ___spawn_speed, ___move_speed, ___ramp_timer, ___ramp_index: (
                    ___spawn_speed,
                    ___move_speed,
                    ___ramp_timer - 1,
                    ___ramp_index,
                ),
                __update_ramp,
                __spawn_speed,
                __move_speed,
                __ramp_timer,
                __ramp_index,
            ),
            lambda __spawn_speed, __move_speed, __ramp_timer, __ramp_index: (
                __spawn_speed,
                __move_speed,
                __ramp_timer,
                __ramp_index,
            ),
            _spawn_speed,
            _move_speed,
            _ramp_timer,
            _ramp_index,
        ),
        spawn_speed,
        move_speed,
        ramp_timer,
        ramp_index,
    )
    # if spawn_speed > 1 or move_speed > 1:
    #     if ramp_timer >= 0:
    #         ramp_timer -= 1
    #     else:
    #         spawn_speed, move_speed, ramp_timer, ramp_index = __update_ramp(
    #             spawn_speed, move_speed, ramp_timer, ramp_index
    #         )

    return spawn_speed, move_speed, ramp_timer, ramp_index


@jax.jit
def __update_ramp(spawn_speed, move_speed, ramp_timer, ramp_index):
    move_speed = jax.lax.cond(
        move_speed > 1,
        lambda _move_speed: jax.lax.cond(
            ramp_index % 2,
            lambda __move_speed: __move_speed - 1,
            lambda __move_speed: __move_speed,
            _move_speed,
        ),
        lambda _move_speed: _move_speed,
        move_speed,
    )
    # if move_speed > 1 and ramp_index % 2:
    #     move_speed -= 1

    jax.lax.cond(
        spawn_speed > 1,
        lambda _spawn_speed: spawn_speed - 1,
        lambda _spawn_speed: spawn_speed,
        spawn_speed,
    )
    # if spawn_speed > 1:
    #     spawn_speed -= 1

    ramp_index += 1
    ramp_timer = ramp_interval

    return spawn_speed, move_speed, ramp_timer, ramp_index


@jax.jit
def _reset_det() -> MinAtarAsterixState:
    return MinAtarAsterixState()


def _to_obs(state: MinAtarAsterixState) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=bool)
    obs = obs.at[state.player_y, state.player_x, 0].set(True)
    # state[self.player_y, self.player_x, self.channels["player"]] = 1
    for i in range(8):
        if state.entities[i, 0] != 1e5:
            if state.entities[i, 3]:
                c = 3
            else:
                c = 1
            obs = obs.at[state.entities[i, 1], state.entities[i, 0], c].set(
                True
            )
            if state.entities[i, 2]:
                back_x = state.entities[i, 0] - 1
            else:
                back_x = state.entities[i, 0] + 1

            if back_x >= 0 and back_x <= 9:
                obs = obs.at[state.entities[i, 1], back_x, 2].set(True)

    # for x in self.entities:
    #     if x is not None:
    #         c = self.channels["gold"] if x[3] else self.channels["enemy"]
    #         state[x[1], x[0], c] = 1
    #         back_x = x[0] - 1 if x[2] else x[0] + 1
    #         if back_x >= 0 and back_x <= 9:
    #             state[x[1], back_x, self.channels["trail"]] = 1
    return obs
