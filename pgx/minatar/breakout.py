"""MinAtar/Breakout: A form of github.com/kenjyoung/MinAtar

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
class MinAtarBreakoutState:
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


@jax.jit
def step(
    state: MinAtarBreakoutState,
    action: jnp.ndarray,
    rng: jnp.ndarray,
    sticky_action_prob: jnp.ndarray,
) -> Tuple[MinAtarBreakoutState, int, bool]:
    action = jax.lax.cond(
        jax.random.uniform(rng) < sticky_action_prob,
        lambda _: state.last_action,
        lambda _: action,
        0,
    )
    return _step_det(state, action)


@jax.jit
def reset(rng: jnp.ndarray) -> MinAtarBreakoutState:
    ball_start = jax.random.choice(rng, 2)
    return _reset_det(ball_start=ball_start)


@jax.jit
def to_obs(state: MinAtarBreakoutState) -> jnp.ndarray:
    return _to_obs(state)


def _step_det(
    state: MinAtarBreakoutState, action: jnp.ndarray
) -> Tuple[MinAtarBreakoutState, int, bool]:
    if state.terminal:
        return (
            state.replace(last_action=action),
            0,
            jnp.array(True, dtype=jnp.bool_),
        )
    else:
        return _step_det_at_non_terminal(state, action)


def _step_det_at_non_terminal(
    state: MinAtarBreakoutState, action: jnp.ndarray
) -> Tuple[MinAtarBreakoutState, int, bool]:
    ball_y = state.ball_y
    ball_x = state.ball_x
    ball_dir = state.ball_dir
    pos = state.pos
    brick_map = state.brick_map
    strike = state.strike
    terminal = state.terminal

    r = 0

    pos = apply_action(pos, action)

    # Update ball position
    last_x = ball_x
    last_y = ball_y
    new_x, new_y = update_ball_pos(ball_x, ball_y, ball_dir)

    strike_toggle = False
    if new_x < 0 or new_x > 9:
        new_x, ball_dir = update_ball_pos_x(new_x, ball_dir)
    if new_y < 0:
        new_y, ball_dir = update_ball_pos_y(ball_dir)
    elif brick_map[new_y, new_x] == 1:
        strike_toggle = True
        if not strike:
            r, strike, brick_map, new_y, ball_dir = update_by_strike(
                r, brick_map, new_x, new_y, last_y, ball_dir
            )
    elif new_y == 9:
        brick_map, new_y, ball_dir, terminal = update_by_botom(
            brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
        )

    if not strike_toggle:
        strike = jnp.array(False, dtype=jnp.bool_)

    ball_x = new_x
    ball_y = new_y
    state = MinAtarBreakoutState(
        ball_y=ball_y,
        ball_x=ball_x,
        ball_dir=ball_dir,
        pos=pos,
        brick_map=brick_map,
        strike=strike,
        last_x=last_x,
        last_y=last_y,
        terminal=terminal,
        last_action=action,
    )
    return state, r, terminal


def apply_action(pos, action):
    # self.action_map = ['n','l','u','r','d','f']
    if action == 1:
        pos = max(0, pos - 1)
    elif action == 3:
        pos = min(9, pos + 1)
    return pos


def update_ball_pos(ball_x, ball_y, ball_dir):
    if ball_dir == 0:
        new_x = ball_x - 1
        new_y = ball_y - 1
    elif ball_dir == 1:
        new_x = ball_x + 1
        new_y = ball_y - 1
    elif ball_dir == 2:
        new_x = ball_x + 1
        new_y = ball_y + 1
    elif ball_dir == 3:
        new_x = ball_x - 1
        new_y = ball_y + 1
    return new_x, new_y


def update_ball_pos_x(new_x, ball_dir):
    if new_x < 0:
        new_x = 0
    if new_x > 9:
        new_x = 9
    ball_dir = [1, 0, 3, 2][ball_dir]
    return new_x, ball_dir


def update_ball_pos_y(ball_dir):
    new_y = 0
    ball_dir = [3, 2, 1, 0][ball_dir]
    return new_y, ball_dir


def update_by_strike(r, brick_map, new_x, new_y, last_y, ball_dir):
    brick_map = brick_map.at[new_y, new_x].set(False)
    new_y = last_y
    ball_dir = [3, 2, 1, 0][ball_dir]
    return r + 1, True, brick_map, new_y, ball_dir


def update_by_botom(
    brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
):
    if brick_map.sum() == 0:
        brick_map = brick_map.at[1:4, :].set(True)
    if ball_x == pos:
        ball_dir = [3, 2, 1, 0][ball_dir]
        new_y = last_y
    elif new_x == pos:
        ball_dir = [2, 3, 0, 1][ball_dir]
        new_y = last_y
    else:
        terminal = jnp.array(True, dtype=jnp.bool_)
    return brick_map, new_y, ball_dir, terminal


@jax.jit
def _reset_det(ball_start: jnp.ndarray) -> MinAtarBreakoutState:
    ball_x, ball_dir = jax.lax.switch(
        ball_start,
        [lambda: (ZERO, TWO), lambda: (NINE, THREE)],
    )
    last_x = ball_x
    return MinAtarBreakoutState(
        ball_x=ball_x, ball_dir=ball_dir, last_x=last_x
    )  # type: ignore


@jax.jit
def _to_obs(state: MinAtarBreakoutState) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state.ball_y, state.ball_x, 1].set(True)
    obs = obs.at[9, state.pos, 0].set(True)
    obs = obs.at[state.last_y, state.last_x, 2].set(True)
    obs = obs.at[:, :, 3].set(state.brick_map)
    return obs
