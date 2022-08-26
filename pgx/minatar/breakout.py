"""A fork of MinAtar environment distributed at https://github.com/kenjyoung/MinAtar

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


#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
@struct.dataclass
class MinAtarBreakoutState:
    ball_y: int = 3
    ball_x: int = 0
    ball_dir: int = 2
    pos: int = 4
    brick_map: jnp.ndarray = jnp.zeros((10, 10))
    strike: bool = False
    last_x: int = 0
    last_y: int = 3
    terminal: bool = False
    last_action: int = 0


# TODO: sticky action prob
# @jax.jit
def step(
    state: MinAtarBreakoutState, action: int
) -> Tuple[MinAtarBreakoutState, int, bool]:

    r = 0
    # if state.terminal:
    #     return state, r, state.terminal

    ball_y = state.ball_y
    ball_x = state.ball_x
    ball_dir = state.ball_dir
    pos = state.pos
    brick_map = state.brick_map
    strike = state.strike
    last_x = state.last_x
    last_y = state.last_y
    terminal = state.terminal
    last_action = state.last_action

    # Resolve player action
    # d_pos = 0
    # d_pos = jax.lax.cond(pos == 1, lambda x: x - 1, lambda x: x, d_pos)
    # d_pos = jax.lax.cond(pos == 3, lambda x: x + 1, lambda x: x, d_pos)
    # pos += d_pos
    # pos = jax.lax.max(pos, 0)
    # pos = jax.lax.min(pos, 9)
    if action == 1:  # "l"
        pos = max(0, pos - 1)
    elif action == 3:  # "r"
        pos = min(9, pos + 1)

    # Update ball position
    last_x = ball_x
    last_y = ball_y
    dx, dy = 0, 0
    dx, dy = jax.lax.cond(
        ball_dir == 0, lambda x, y: (x - 1, y - 1), lambda x, y: (x, y), dx, dy
    )
    dx, dy = jax.lax.cond(
        ball_dir == 1, lambda x, y: (x + 1, y - 1), lambda x, y: (x, y), dx, dy
    )
    dx, dy = jax.lax.cond(
        ball_dir == 2, lambda x, y: (x + 1, y + 1), lambda x, y: (x, y), dx, dy
    )
    dx, dy = jax.lax.cond(
        ball_dir == 3, lambda x, y: (x - 1, y + 1), lambda x, y: (x, y), dx, dy
    )
    new_x = ball_x + dx
    new_y = ball_y + dy
    # if ball_dir == 0:
    #     new_x = ball_x - 1
    #     new_y = ball_y - 1
    # elif ball_dir == 1:
    #     new_x = ball_x + 1
    #     new_y = ball_y - 1
    # elif ball_dir == 2:
    #     new_x = ball_x + 1
    #     new_y = ball_y + 1
    # elif ball_dir == 3:
    #     new_x = ball_x - 1
    #     new_y = ball_y + 1

    strike_toggle = False
    if new_x < 0 or new_x > 9:
        if new_x < 0:
            new_x = 0
        if new_x > 9:
            new_x = 9
        ball_dir = [1, 0, 3, 2][ball_dir]
    if new_y < 0:
        new_y = 0
        ball_dir = [3, 2, 1, 0][ball_dir]
    elif brick_map[new_y, new_x] == 1:
        strike_toggle = True
        if not strike:
            r += 1
            strike = True
            brick_map = brick_map.at[new_y, new_x].set(0)
            # brick_map[new_y, new_x] = 0
            new_y = last_y
            ball_dir = [3, 2, 1, 0][ball_dir]
    elif new_y == 9:
        if jnp.count_nonzero(brick_map) == 0:
            # brick_map[1:4, :] = 1
            brick_map = brick_map.at[1:4, :] = 1
        if ball_x == pos:
            ball_dir = [3, 2, 1, 0][ball_dir]
            new_y = last_y
        elif new_x == pos:
            ball_dir = [2, 3, 0, 1][ball_dir]
            new_y = last_y
        else:
            terminal = True

    if not strike_toggle:
        strike = False

    ball_x = new_x
    ball_y = new_y
    last_action = action

    next_state = MinAtarBreakoutState(
        ball_y,
        ball_x,
        ball_dir,
        pos,
        brick_map,
        strike,
        last_x,
        last_y,
        terminal,
        last_action,
    )

    return next_state, r, terminal
