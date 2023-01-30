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
DIVER_MOVE_INTERVAL: jnp.ndarray = jnp.int8(5)


ZERO: jnp.ndarray = jnp.int8(0)
NINE: jnp.ndarray = jnp.int8(9)
TRUE: jnp.ndarray = jnp.bool_(True)
FALSE: jnp.ndarray = jnp.bool_(False)


@struct.dataclass
class State:
    oxygen: jnp.ndarray = MAX_OXYGEN
    diver_count: jnp.ndarray = ZERO
    sub_x: jnp.ndarray = jnp.int8(5)
    sub_y: jnp.ndarray = ZERO
    sub_or: jnp.ndarray = FALSE
    f_bullets: jnp.ndarray = -jnp.ones(
        (5, 3), dtype=jnp.int8
    )  # <= 2  TODO: confirm
    e_bullets: jnp.ndarray = -jnp.ones(
        (25, 3), dtype=jnp.int8
    )  # <= 1 per each sub  TODO: confirm
    e_fish: jnp.ndarray = -jnp.ones(
        (25, 4), dtype=jnp.int8
    )  # <= 19  TODO: confirm
    e_subs: jnp.ndarray = -jnp.ones(
        (25, 5), dtype=jnp.int8
    )  # <= 19  TODO: confirm
    divers: jnp.ndarray = -jnp.ones(
        (5, 4), dtype=jnp.int8
    )  # <= 2  TODO: confirm
    e_spawn_speed: jnp.ndarray = INIT_SPAWN_SPEED
    e_spawn_timer: jnp.ndarray = INIT_SPAWN_SPEED
    d_spawn_timer: jnp.ndarray = DIVER_SPAWN_SPEED
    move_speed: jnp.ndarray = INIT_MOVE_INTERVAL
    ramp_index: jnp.ndarray = ZERO  # TODO: require int16?
    shot_timer: jnp.ndarray = ZERO
    surface: jnp.ndarray = TRUE
    terminal: jnp.ndarray = FALSE
    last_action: jnp.ndarray = ZERO


def step(
    state: State,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    rngs = jax.random.split(rng, 6)
    action = jnp.int8(action)
    # sticky action
    action = jax.lax.cond(
        jax.random.uniform(rngs[0]) < sticky_action_prob,
        lambda: state.last_action,
        lambda: action,
    )
    enemy_lr = jax.random.choice(rngs[1], jnp.array([True, False]))
    is_sub = jax.random.choice(
        rngs[2], jnp.array([True, False]), p=jnp.array([1 / 3, 2 / 3])
    )
    enemy_y = jax.random.choice(rngs[3], jnp.arange(1, 9))
    diver_lr = jax.random.choice(rngs[4], jnp.array([True, False]))
    diver_y = jax.random.choice(rngs[5], jnp.arange(1, 9))
    return _step_det(
        state, action, enemy_lr, is_sub, enemy_y, diver_lr, diver_y
    )


def _step_det(
    state: State,
    action: jnp.ndarray,
    enemy_lr,
    is_sub,
    enemy_y,
    diver_lr,
    diver_y,
):
    return lax.cond(
        state.terminal,
        lambda: (
            state.replace(last_action=action),  # type: ignore
            jnp.int16(0),
            state.terminal,
        ),
        lambda: _step_det_at_non_terminal(
            state, action, enemy_lr, is_sub, enemy_y, diver_lr, diver_y
        ),
    )


def _step_det_at_non_terminal(
    state: State,
    action: jnp.ndarray,
    enemy_lr,
    is_sub,
    enemy_y,
    diver_lr,
    diver_y,
):
    ramping = TRUE

    oxygen = state.oxygen
    diver_count = state.diver_count
    sub_x = state.sub_x
    sub_y = state.sub_y
    sub_or = state.sub_or
    f_bullets = state.f_bullets
    e_bullets = state.e_bullets
    e_fish = state.e_fish
    e_subs = state.e_subs
    divers = state.divers
    e_spawn_speed = state.e_spawn_speed
    e_spawn_timer = state.e_spawn_timer
    d_spawn_timer = state.d_spawn_timer
    move_speed = state.move_speed
    ramp_index = state.ramp_index
    shot_timer = state.shot_timer
    surface = state.surface
    terminal = state.terminal

    r = jnp.int16(0)

    # Spawn enemy if timer is up
    e_subs, e_fish = lax.cond(
        e_spawn_timer == 0,
        lambda: _spawn_enemy(
            e_subs, e_fish, move_speed, enemy_lr, is_sub, enemy_y
        ),
        lambda: (e_subs, e_fish),
    )
    e_spawn_timer = lax.cond(
        e_spawn_timer == 0, lambda: e_spawn_speed, lambda: e_spawn_timer
    )

    # Spawn diver if timer is up
    divers, d_spawn_timer = lax.cond(
        d_spawn_timer == 0,
        lambda: (_spawn_diver(divers, diver_lr, diver_y), DIVER_SPAWN_SPEED),
        lambda: (divers, d_spawn_timer),
    )

    # Resolve player action
    f_bullets, shot_timer, sub_x, sub_y, sub_or = _resolve_action(
        action, shot_timer, f_bullets, sub_x, sub_y, sub_or
    )

    # Update friendly Bullets
    f_bullets, e_subs, e_fish, r = _update_friendly_bullets(
        f_bullets, e_subs, e_fish, r
    )

    # Update divers
    divers, diver_count = _update_divers(divers, diver_count, sub_x, sub_y)

    # Update enemy subs
    f_bullets, e_subs, e_bullets, terminal, r = _update_enemy_subs(
        f_bullets, e_subs, e_bullets, sub_x, sub_y, move_speed, terminal, r
    )

    # Update enemy bullets
    e_bullets, terminal = _update_enemy_bullets(
        e_bullets, sub_x, sub_y, terminal
    )

    # Update enemy fish
    f_bullets, e_fish, terminal, r = _update_enemy_fish(
        f_bullets, e_fish, sub_x, sub_y, move_speed, terminal, r
    )

    # Update various timers
    e_spawn_timer = lax.cond(
        e_spawn_timer > 0, lambda: e_spawn_timer - 1, lambda: e_spawn_timer
    )
    d_spawn_timer = lax.cond(
        d_spawn_timer, lambda: d_spawn_timer - 1, lambda: d_spawn_timer
    )
    shot_timer = lax.cond(
        shot_timer > 0, lambda: shot_timer - 1, lambda: shot_timer
    )
    terminal |= oxygen < 0
    tmp = surface
    oxygen = lax.cond(sub_y > 0, lambda: oxygen - 1, lambda: oxygen)
    surface = lax.cond(sub_y > 0, lambda: FALSE, lambda: surface)
    terminal = lax.cond(
        (sub_y <= 0) & ~tmp & (diver_count == 0),
        lambda: TRUE,
        lambda: terminal,
    )
    surface |= (sub_y <= 0) & ~tmp & (diver_count != 0)
    _r, oxygen, diver_count, move_speed, e_spawn_speed, ramp_index = lax.cond(
        (sub_y <= 0) & ~tmp & (diver_count != 0),
        lambda: _surface(
            diver_count, oxygen, e_spawn_speed, move_speed, ramping, ramp_index
        ),
        lambda: (
            jnp.int16(0),
            oxygen,
            diver_count,
            move_speed,
            e_spawn_speed,
            ramp_index,
        ),
    )
    r += _r

    state = State(
        oxygen=oxygen,
        diver_count=diver_count,
        sub_x=sub_x,
        sub_y=sub_y,
        sub_or=sub_or,
        f_bullets=f_bullets,
        e_bullets=e_bullets,
        e_fish=e_fish,
        e_subs=e_subs,
        divers=divers,
        e_spawn_speed=e_spawn_speed,
        e_spawn_timer=e_spawn_timer,
        d_spawn_timer=d_spawn_timer,
        move_speed=move_speed,
        ramp_index=ramp_index,
        shot_timer=shot_timer,
        surface=surface,
        terminal=terminal,
        last_action=action,
    )  # type: ignore
    return state, r, terminal


def find_ix(arr):
    ix = lax.while_loop(lambda i: arr[i][0] != -1, lambda i: i + 1, 0)
    return ix


def _resolve_action(action, shot_timer, f_bullets, sub_x, sub_y, sub_or):
    f_bullets, shot_timer = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: (
            f_bullets.at[find_ix(f_bullets)].set(
                jnp.int8([sub_x, sub_y, sub_or])
            ),
            SHOT_COOL_DOWN,
        ),
        lambda: (f_bullets, shot_timer),
    )
    sub_x, sub_or = lax.cond(
        action == 1,
        lambda: (lax.max(ZERO, sub_x - 1), FALSE),
        lambda: (sub_x, sub_or),
    )
    sub_x, sub_or = lax.cond(
        action == 3,
        lambda: (lax.min(NINE, sub_x + 1), TRUE),
        lambda: (sub_x, sub_or),
    )
    sub_y = lax.cond(
        action == 2, lambda: lax.max(ZERO, sub_y - 1), lambda: sub_y
    )
    sub_y = lax.cond(
        action == 4, lambda: lax.min(jnp.int8(8), sub_y + 1), lambda: sub_y
    )
    return f_bullets, shot_timer, sub_x, sub_y, sub_or


def _update_by_f_bullets_hit(j, _f_bullets, e):
    k = lax.while_loop(
        lambda i: ~_is_hit(_f_bullets[j], e[i, 0], e[i, 1]) & (i < 25),
        lambda i: i + 1,
        0,
    )
    _f_bullets, e, removed = lax.cond(
        k < 25,
        lambda: (_remove_i(_f_bullets, j), _remove_i(e, k), TRUE),
        lambda: (_f_bullets, e, FALSE),
    )
    return _f_bullets, e, removed


def _update_friendly_bullets(f_bullets, e_subs, e_fish, r):
    def _remove(j, _f_bullets, _e_subs, _e_fish, _r):
        _f_bullets, _e_fish, removed = _update_by_f_bullets_hit(
            j, _f_bullets, _e_fish
        )
        _r += removed
        _f_bullets, _e_subs, removed = lax.cond(
            removed,
            lambda: (_f_bullets, _e_subs, removed),
            lambda: _update_by_f_bullets_hit(j, _f_bullets, _e_subs),
        )
        _r += removed
        return _f_bullets, _e_subs, _e_fish, _r

    def _update_each(i, x):
        _f_bullets, _e_subs, _e_fish, _r = x
        j = 5 - i - 1
        is_filled = _is_filled(_f_bullets[j])
        _f_bullets = lax.cond(
            is_filled,
            lambda: _f_bullets.at[j, 0].add(
                lax.cond(_f_bullets[j, 2], lambda: 1, lambda: -1)
            ),
            lambda: _f_bullets,
        )
        _f_bullets, _e_subs, _e_fish, _r = lax.cond(
            is_filled,
            lambda: lax.cond(
                _is_out(_f_bullets[j]),
                lambda: (_remove_i(_f_bullets, j), _e_subs, _e_fish, _r),
                lambda: _remove(j, _f_bullets, _e_subs, _e_fish, _r),
            ),
            lambda: (_f_bullets, _e_subs, _e_fish, _r),
        )
        return _f_bullets, _e_subs, _e_fish, _r

    f_bullets, e_subs, e_fish, r = lax.fori_loop(
        0, 5, _update_each, (f_bullets, e_subs, e_fish, r)
    )
    return f_bullets, e_subs, e_fish, r


def _is_hit(row, x, y):
    return (row[0] == x) & (row[1] == y)


def _is_out(row):
    return (row[0] < 0) | (row[0] > 9)


def _is_filled(row):
    return jnp.any(row != -1)


def _update_divers(divers, diver_count, sub_x, sub_y):
    def _update_by_move(_divers, _diver_count, j):
        _divers = _divers.at[j, 3].set(DIVER_MOVE_INTERVAL)
        _divers = _divers.at[j, 0].add(
            lax.cond(_divers[j, 2], lambda: 1, lambda: -1)
        )
        _divers, _diver_count = lax.cond(
            _is_out(_divers[j]),
            lambda: (_remove_i(_divers, j), _diver_count),
            lambda: lax.cond(
                _is_hit(_divers[j], sub_x, sub_y) & (_diver_count < 6),
                lambda: (_remove_i(_divers, j), _diver_count + 1),
                lambda: (_divers, _diver_count),
            ),
        )
        return _divers, _diver_count

    def _update_each(i, x):
        _divers, _diver_count = x
        j = 5 - i - 1
        return lax.cond(
            _is_filled(_divers[j]),
            lambda: lax.cond(
                _is_hit(_divers[j], sub_x, sub_y) & (_diver_count < 6),
                lambda: (_remove_i(_divers, j), _diver_count + 1),
                lambda: lax.cond(
                    _divers[j, 3] == 0,
                    lambda: _update_by_move(_divers, _diver_count, j),
                    lambda: (_divers.at[j, 3].add(-1), _diver_count),
                ),
            ),
            lambda: (_divers, _diver_count),
        )

    divers, diver_count = lax.fori_loop(
        0, 5, _update_each, (divers, diver_count)
    )

    return divers, diver_count


def _update_enemy_subs(
    f_bullets, e_subs, e_bullets, sub_x, sub_y, move_speed, terminal, r
):
    def _update_sub(j, _f_bullets, _e_subs, _terminal, _r):
        _e_subs = _e_subs.at[j, 3].set(move_speed)
        _e_subs = _e_subs.at[j, 0].add(
            lax.cond(_e_subs[j, 2], lambda: 1, lambda: -1)
        )
        is_out = _is_out(_e_subs[j])
        is_hit = _is_hit(_e_subs[j], sub_x, sub_y)
        _e_subs = lax.cond(
            is_out, lambda: _remove_i(_e_subs, j), lambda: _e_subs
        )
        _terminal = lax.cond(~is_out & is_hit, lambda: TRUE, lambda: _terminal)
        _f_bullets, _e_subs, removed = lax.cond(
            ~is_out & ~is_hit,
            lambda: _update_by_hit(j, _f_bullets, _e_subs),
            lambda: (_f_bullets, _e_subs, FALSE),
        )
        _r += removed
        return _f_bullets, _e_subs, _terminal, _r

    def _update_each_filled(j, x):
        _f_bullets, _e_subs, _e_bullets, _terminal, _r = x
        _terminal |= _is_hit(_e_subs[j], sub_x, sub_y)
        _f_bullets, _e_subs, _terminal, _r = lax.cond(
            _e_subs[j, 3] == 0,
            lambda: _update_sub(j, _f_bullets, _e_subs, _terminal, _r),
            lambda: (_f_bullets, _e_subs.at[j, 3].add(-1), _terminal, _r),
        )
        timer_zero = _e_subs[j, 4] == 0
        _e_subs = lax.cond(
            timer_zero,
            lambda: _e_subs.at[j, 4].set(ENEMY_SHOT_INTERVAL),
            lambda: _e_subs.at[j, 4].add(-1),
        )
        _e_bullets = lax.cond(
            timer_zero,
            lambda: _e_bullets.at[find_ix(_e_bullets)].set(
                jnp.int8(
                    [
                        lax.cond(
                            _e_subs[j, 2],
                            lambda: _e_subs[j, 0],
                            lambda: _e_subs[j, 0],
                        ),
                        _e_subs[j, 1],
                        _e_subs[j, 2],
                    ]
                )
            ),
            lambda: _e_bullets,
        )
        return _f_bullets, _e_subs, _e_bullets, _terminal, _r

    def _update_each(i, x):
        j = 25 - i - 1
        return lax.cond(
            _is_filled(x[1][j]), lambda: (_update_each_filled(j, x)), lambda: x
        )

    f_bullets, e_subs, e_bullets, terminal, r = lax.fori_loop(
        0, 25, _update_each, (f_bullets, e_subs, e_bullets, terminal, r)
    )

    return f_bullets, e_subs, e_bullets, terminal, r


def _remove_i(arr, i):
    N = arr.shape[0]
    arr = lax.fori_loop(
        i, N - 1, lambda j, _arr: _arr.at[j].set(arr[j + 1]), arr
    )
    return arr


def _remove_out_of_bound(arr, ix):
    arr = lax.fori_loop(
        0,
        ix,
        lambda i, a: lax.cond(
            (a[i][0] < 0) & (a[i][1] > 9), lambda: _remove_i(a, i), lambda: a
        ),
        arr,
    )
    return arr


def _remove_hit(arr, ix, x, y):
    arr = lax.fori_loop(
        0,
        ix,
        lambda i, a: lax.cond(
            (a[i][0] == x) & (a[i][1] == y), lambda: _remove_i(a, i), lambda: a
        ),
        arr,
    )
    return arr


def _step_obj(arr, ix):
    arr = lax.fori_loop(
        0,
        ix,
        lambda i, a: a.at[i, 0].add(lax.cond(a[i, 2], lambda: 1, lambda: -1)),
        arr,
    )
    return arr


def _hit(arr, ix, x, y):
    return lax.fori_loop(
        0,
        ix,
        lambda i, t: lax.cond(
            (arr[i][0] == x) & (arr[i][1] == y), lambda: TRUE, lambda: t
        ),
        FALSE,
    )


def _update_enemy_bullets(e_bullets, sub_x, sub_y, terminal):
    ix = find_ix(e_bullets)
    terminal |= _hit(e_bullets, ix, sub_x, sub_y)
    e_bullets = _step_obj(e_bullets, ix)
    e_bullets = _remove_out_of_bound(e_bullets, ix)
    terminal |= _hit(e_bullets, ix, sub_x, sub_y)
    return e_bullets, terminal


def _update_by_hit(j, _f_bullets, e):
    k = lax.while_loop(
        lambda i: ~_is_hit(e[j], _f_bullets[i, 0], _f_bullets[i, 1]) & (i < 5),
        lambda i: i + 1,
        0,
    )
    _f_bullets, e, removed = lax.cond(
        k < 5,
        lambda: (_remove_i(_f_bullets, k), _remove_i(e, j), TRUE),
        lambda: (_f_bullets, e, FALSE),
    )
    return _f_bullets, e, removed


def _update_enemy_fish(
    f_bullets, e_fish, sub_x, sub_y, move_speed, terminal, r
):
    def _update_by_hit_fish(j, _f_bullets, e, _terminal, _r):
        _f_bullets, e, removed = _update_by_hit(j, _f_bullets, e)
        return _f_bullets, e, _terminal, _r + removed

    def _update_fish(j, _f_bullets, _e_fish, _terminal, _r):
        _e_fish = _e_fish.at[j, 3].set(move_speed)
        _e_fish = _e_fish.at[j, 0].add(
            lax.cond(_e_fish[j, 2], lambda: 1, lambda: -1)
        )
        _f_bullets, _e_fish, _terminal, _r = lax.cond(
            _is_out(_e_fish[j]),
            lambda: (_f_bullets, _remove_i(_e_fish, j), _terminal, _r),
            lambda: lax.cond(
                _is_hit(_e_fish[j], sub_x, sub_y),
                lambda: (_f_bullets, _e_fish, TRUE, _r),
                lambda: _update_by_hit_fish(
                    j, _f_bullets, _e_fish, _terminal, _r
                ),
            ),
        )
        return _f_bullets, _e_fish, _terminal, _r

    def _update_each(i, x):
        j = 25 - i - 1
        _f_bullets, _e_fish, _terminal, _r = x
        _terminal |= _is_hit(_e_fish[j], sub_x, sub_y)
        _f_bullets, _e_fish, _terminal, _r = lax.cond(
            _is_filled(_e_fish[j]),
            lambda: lax.cond(
                _e_fish[j, 3] == 0,
                lambda: _update_fish(j, _f_bullets, _e_fish, _terminal, _r),
                lambda: (_f_bullets, _e_fish.at[j, 3].add(-1), _terminal, _r),
            ),
            lambda: (_f_bullets, _e_fish, _terminal, _r),
        )
        return _f_bullets, _e_fish, _terminal, _r

    f_bullets, e_fish, terminal, r = lax.fori_loop(
        0, 25, _update_each, (f_bullets, e_fish, terminal, r)
    )

    return f_bullets, e_fish, terminal, r


# Called when player hits surface (top row) if they have no divers, this ends the game,
# if they have 6 divers this gives reward proportional to the remaining oxygen and restores full oxygen
# otherwise this reduces the number of divers and restores full oxygen
def _surface(
    diver_count, oxygen, e_spawn_speed, move_speed, ramping, ramp_index
):
    diver_count, r = lax.cond(
        diver_count == 6,
        lambda: (ZERO, oxygen * 10 // MAX_OXYGEN),
        lambda: (diver_count, jnp.int16(0)),
    )
    oxygen = MAX_OXYGEN
    diver_count -= 1
    ramp_update = ramping & ((e_spawn_speed > 1) | (move_speed > 2))
    ramp_index = lax.cond(
        ramp_update, lambda: ramp_index + 1, lambda: ramp_index
    )
    move_speed = lax.cond(
        ramp_update & ((move_speed > 2) & (ramp_index % 2)),
        lambda: move_speed - 1,
        lambda: move_speed,
    )
    e_spawn_speed = lax.cond(
        ramp_update & (e_spawn_speed > 1),
        lambda: e_spawn_speed - 1,
        lambda: e_spawn_speed,
    )
    return r, oxygen, diver_count, move_speed, e_spawn_speed, ramp_index


# Spawn an enemy fish or submarine in random row and random direction,
# if the resulting row and direction would lead to a collision, do nothing instead
def _spawn_enemy(e_subs, e_fish, move_speed, enemy_lr, is_sub, enemy_y):
    x = lax.cond(enemy_lr, lambda: ZERO, lambda: NINE)
    has_collision = (
        (e_subs[:, 1] == enemy_y) & (e_subs[:, 2] != enemy_lr)
    ).sum() > 0
    has_collision |= (
        (e_fish[:, 1] == enemy_y) & (e_fish[:, 2] != enemy_lr)
    ).sum() > 0
    return lax.cond(
        has_collision,
        lambda: (e_subs, e_fish),
        lambda: lax.cond(
            is_sub,
            lambda: (
                e_subs.at[find_ix(e_subs)].set(
                    jnp.int8(
                        [
                            x,
                            enemy_y,
                            enemy_lr,
                            move_speed,
                            ENEMY_SHOT_INTERVAL,
                        ]
                    )
                ),
                e_fish,
            ),
            lambda: (
                e_subs,
                e_fish.at[find_ix(e_fish)].set(
                    jnp.int8([x, enemy_y, enemy_lr, move_speed])
                ),
            ),
        ),
    )


# Spawn a diver in random row with random direction
def _spawn_diver(divers, diver_lr, diver_y):
    x = lax.cond(diver_lr, lambda: ZERO, lambda: NINE)
    ix = find_ix(divers)
    divers = divers.at[ix].set(
        jnp.array([x, diver_y, diver_lr, DIVER_MOVE_INTERVAL], dtype=jnp.int8)
    )
    return divers


def init(rng: jnp.ndarray) -> State:
    return _init_det()


def observe(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 10), dtype=jnp.bool_)
    obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
    back_x = lax.cond(
        state.sub_or, lambda: state.sub_x - 1, lambda: state.sub_x + 1
    )
    obs = obs.at[state.sub_y, back_x, 1].set(1)
    oxygen_guage = state.oxygen * 10 // MAX_OXYGEN
    # hotfix to align to original minatar
    oxygen_guage = lax.cond(
        state.oxygen < 0, lambda: jnp.int16(9), lambda: oxygen_guage
    )
    obs = lax.fori_loop(
        jnp.int16(0),
        oxygen_guage,
        lambda i, _obs: _obs.at[9, i, 7].set(1),
        obs,
    )
    obs = lax.fori_loop(
        9 - state.diver_count,
        jnp.int8(9),
        lambda i, _obs: _obs.at[9, i, 8].set(1),
        obs,
    )
    obs = lax.fori_loop(
        0,
        5,
        lambda i, _obs: lax.cond(
            state.f_bullets[i][0] >= 0,
            lambda: _obs.at[
                state.f_bullets[i][1], state.f_bullets[i][0], 2
            ].set(1),
            lambda: _obs,
        ),
        obs,
    )
    obs = lax.fori_loop(
        0,
        25,
        lambda i, _obs: lax.cond(
            state.e_bullets[i][0] >= 0,
            lambda: _obs.at[
                state.e_bullets[i][1], state.e_bullets[i][0], 4
            ].set(1),
            lambda: _obs,
        ),
        obs,
    )

    def set_e_fish(_obs, fish):
        _obs = _obs.at[fish[1], fish[0], 5].set(1)
        back_x = fish[0] + jnp.array([1, -1])[fish[2]]
        _obs = lax.cond(
            (0 <= back_x) & (back_x <= 9),
            lambda: _obs.at[fish[1], back_x, 3].set(1),
            lambda: _obs,
        )
        return _obs

    obs = lax.fori_loop(
        0,
        25,
        lambda i, _obs: lax.cond(
            state.e_fish[i][0] >= 0,
            lambda: set_e_fish(_obs, state.e_fish[i]),
            lambda: _obs,
        ),
        obs,
    )

    def set_e_subs(_obs, sub):
        _obs = _obs.at[sub[1], sub[0], 6].set(1)
        back_x = sub[0] + jnp.array([1, -1], dtype=jnp.int8)[sub[2]]
        _obs = lax.cond(
            (0 <= back_x) & (back_x <= 9),
            lambda: _obs.at[sub[1], back_x, 3].set(1),
            lambda: _obs,
        )
        return _obs

    obs = lax.fori_loop(
        0,
        25,
        lambda i, _obs: lax.cond(
            state.e_subs[i][0] >= 0,
            lambda: set_e_subs(_obs, state.e_subs[i]),
            lambda: _obs,
        ),
        obs,
    )

    def set_divers(_obs, diver):
        _obs = _obs.at[diver[1], diver[0], 9].set(1)
        back_x = diver[0] + jnp.array([1, -1], dtype=jnp.int8)[diver[2]]
        _obs = lax.cond(
            (back_x >= 0) & (back_x <= 9),
            lambda: _obs.at[diver[1], back_x, 3].set(1),
            lambda: _obs,
        )
        return _obs

    obs = lax.fori_loop(
        0,
        5,
        lambda i, _obs: lax.cond(
            state.divers[i][0] >= 0,
            lambda: set_divers(_obs, state.divers[i]),
            lambda: _obs,
        ),
        obs,
    )

    return obs


def _init_det() -> State:
    return State()
