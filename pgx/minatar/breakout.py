"""MinAtar/Breakout: A fork of github.com/kenjyoung/MinAtar

The player controls a paddle on the bottom of the screen and must bounce a ball to break 3 rows of bricks along the
top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.

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

ZERO = jnp.array(0, dtype=jnp.int8)
ONE = jnp.array(1, dtype=jnp.int8)
TWO = jnp.array(2, dtype=jnp.int8)
THREE = jnp.array(3, dtype=jnp.int8)
FOUR = jnp.array(4, dtype=jnp.int8)
NINE = jnp.array(9, dtype=jnp.int8)


@struct.dataclass
class State:
    ball_y: jnp.ndarray = THREE
    ball_x: jnp.ndarray = ZERO
    ball_dir: jnp.ndarray = TWO
    pos: jnp.ndarray = FOUR
    brick_map: jnp.ndarray = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[1:4, :].set(True)
    )
    strike: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    last_x: jnp.ndarray = ZERO
    last_y: jnp.ndarray = THREE
    terminal: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    last_action: jnp.ndarray = ZERO


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
    return _step_det(state, action)


def init(rng: jnp.ndarray) -> State:
    ball_start = jax.random.choice(rng, 2)
    return _init_det(ball_start=ball_start)


def observe(state: State) -> jnp.ndarray:
    return _to_obs(state)


def _step_det(
    state: State, action: jnp.ndarray
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    return jax.lax.cond(
        state.terminal,
        lambda: (
            state.replace(last_action=action),  # type: ignore
            jnp.array(0, dtype=jnp.int16),
            jnp.array(True, dtype=jnp.bool_),
        ),
        lambda: _step_det_at_non_terminal(state, action),
    )


def _step_det_at_non_terminal(
    state: State, action: jnp.ndarray
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    ball_y = state.ball_y
    ball_x = state.ball_x
    ball_dir = state.ball_dir
    pos = state.pos
    brick_map = state.brick_map
    strike = state.strike
    terminal = state.terminal

    r = jnp.array(0, dtype=jnp.int16)

    pos = apply_action(pos, action)

    # Update ball position
    last_x = ball_x
    last_y = ball_y
    new_x, new_y = update_ball_pos(ball_x, ball_y, ball_dir)

    new_x, ball_dir = jax.lax.cond(
        (new_x < 0) | (new_x > 9),
        lambda: update_ball_pos_x(new_x, ball_dir),
        lambda: (new_x, ball_dir),
    )

    is_new_y_negative = new_y < 0
    is_strike = brick_map[new_y, new_x] == 1
    is_bottom = new_y == 9
    new_y, ball_dir = jax.lax.cond(
        is_new_y_negative,
        lambda: update_ball_pos_y(ball_dir),
        lambda: (new_y, ball_dir),
    )
    strike_toggle = ~is_new_y_negative & is_strike
    r, strike, brick_map, new_y, ball_dir = jax.lax.cond(
        ~is_new_y_negative & is_strike & ~strike,
        lambda: update_by_strike(
            r, brick_map, new_x, new_y, last_y, ball_dir, strike
        ),
        lambda: (r, strike, brick_map, new_y, ball_dir),
    )
    brick_map, new_y, ball_dir, terminal = jax.lax.cond(
        ~is_new_y_negative & ~is_strike & is_bottom,
        lambda: update_by_bottom(
            brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
        ),
        lambda: (brick_map, new_y, ball_dir, terminal),
    )

    strike = jax.lax.cond(
        ~strike_toggle, lambda: jnp.zeros_like(strike), lambda: strike
    )

    state = State(
        ball_y=new_y,
        ball_x=new_x,
        ball_dir=ball_dir,
        pos=pos,
        brick_map=brick_map,
        strike=strike,
        last_x=last_x,
        last_y=last_y,
        terminal=terminal,
        last_action=action,
    )  # type: ignore
    return state, r, terminal


def apply_action(pos, action):
    pos = jax.lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - ONE), lambda: pos
    )
    pos = jax.lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + ONE), lambda: pos
    )
    return pos


def update_ball_pos(ball_x, ball_y, ball_dir):
    return jax.lax.switch(
        ball_dir,
        [
            lambda: (ball_x - ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y + ONE),
            lambda: (ball_x - ONE, ball_y + ONE),
        ],
    )


def update_ball_pos_x(new_x, ball_dir):
    new_x = jax.lax.max(ZERO, new_x)
    new_x = jax.lax.min(NINE, new_x)
    ball_dir = jnp.array([1, 0, 3, 2], dtype=jnp.int8)[ball_dir]
    return new_x, ball_dir


def update_ball_pos_y(ball_dir):
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int8)[ball_dir]
    return ZERO, ball_dir


def update_by_strike(r, brick_map, new_x, new_y, last_y, ball_dir, strike):
    brick_map = brick_map.at[new_y, new_x].set(False)
    new_y = last_y
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int8)[ball_dir]
    return r + 1, jnp.ones_like(strike), brick_map, new_y, ball_dir


def update_by_bottom(
    brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
):
    brick_map = jax.lax.cond(
        brick_map.sum() == 0,
        lambda: brick_map.at[1:4, :].set(True),
        lambda: brick_map,
    )
    new_y, ball_dir, terminal = jax.lax.cond(
        ball_x == pos,
        lambda: (
            last_y,
            jnp.array([3, 2, 1, 0], dtype=jnp.int8)[ball_dir],
            terminal,
        ),
        lambda: jax.lax.cond(
            new_x == pos,
            lambda: (
                last_y,
                jnp.array([2, 3, 0, 1], dtype=jnp.int8)[ball_dir],
                terminal,
            ),
            lambda: (new_y, ball_dir, jnp.array(True, dtype=jnp.bool_)),
        ),
    )
    return brick_map, new_y, ball_dir, terminal


def _init_det(ball_start: jnp.ndarray) -> State:
    ball_x, ball_dir = jax.lax.switch(
        ball_start,
        [lambda: (ZERO, TWO), lambda: (NINE, THREE)],
    )
    last_x = ball_x
    return State(
        ball_x=ball_x, ball_dir=ball_dir, last_x=last_x
    )  # type: ignore


def _to_obs(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state.ball_y, state.ball_x, 1].set(True)
    obs = obs.at[9, state.pos, 0].set(True)
    obs = obs.at[state.last_y, state.last_x, 2].set(True)
    obs = obs.at[:, :, 3].set(state.brick_map)
    return obs
