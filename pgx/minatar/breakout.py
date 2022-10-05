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


@jax.jit
def _step_det(
    state: MinAtarBreakoutState, action: jnp.ndarray
) -> Tuple[MinAtarBreakoutState, int, bool]:
    r = 0
    if self.terminal:
        return r, self.terminal

    a = self.action_map[a]

    # Resolve player action
    if a == "l":
        self.pos = max(0, self.pos - 1)
    elif a == "r":
        self.pos = min(9, self.pos + 1)

    # Update ball position
    self.last_x = self.ball_x
    self.last_y = self.ball_y
    if self.ball_dir == 0:
        new_x = self.ball_x - 1
        new_y = self.ball_y - 1
    elif self.ball_dir == 1:
        new_x = self.ball_x + 1
        new_y = self.ball_y - 1
    elif self.ball_dir == 2:
        new_x = self.ball_x + 1
        new_y = self.ball_y + 1
    elif self.ball_dir == 3:
        new_x = self.ball_x - 1
        new_y = self.ball_y + 1

    strike_toggle = False
    if new_x < 0 or new_x > 9:
        if new_x < 0:
            new_x = 0
        if new_x > 9:
            new_x = 9
        self.ball_dir = [1, 0, 3, 2][self.ball_dir]
    if new_y < 0:
        new_y = 0
        self.ball_dir = [3, 2, 1, 0][self.ball_dir]
    elif self.brick_map[new_y, new_x] == 1:
        strike_toggle = True
        if not self.strike:
            r += 1
            self.strike = True
            self.brick_map[new_y, new_x] = 0
            new_y = self.last_y
            self.ball_dir = [3, 2, 1, 0][self.ball_dir]
    elif new_y == 9:
        if np.count_nonzero(self.brick_map) == 0:
            self.brick_map[1:4, :] = 1
        if self.ball_x == self.pos:
            self.ball_dir = [3, 2, 1, 0][self.ball_dir]
            new_y = self.last_y
        elif new_x == self.pos:
            self.ball_dir = [2, 3, 0, 1][self.ball_dir]
            new_y = self.last_y
        else:
            self.terminal = True

    if not strike_toggle:
        self.strike = False

    self.ball_x = new_x
    self.ball_y = new_y
    return r, self.terminal


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
