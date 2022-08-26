"""A fork of MinAtar environment distributed at https://github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
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
@dataclass
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
def step(
    state: MinAtarBreakoutState, action: int
) -> Tuple[MinAtarBreakoutState, int, bool]:

    r = 0
    if state.terminal:
        return state, r, state.terminal

    next_state = copy.deepcopy(state)

    # Resolve player action
    if action == 1:  # "l"
        next_state.pos = max(0, next_state.pos - 1)
    elif action == 3:  # "r"
        next_state.pos = min(9, next_state.pos + 1)

    # Update ball position
    next_state.last_x = next_state.ball_x
    next_state.last_y = next_state.ball_y
    assert next_state.ball_dir in [0, 1, 2, 3]
    if next_state.ball_dir == 0:
        new_x = next_state.ball_x - 1
        new_y = next_state.ball_y - 1
    elif next_state.ball_dir == 1:
        new_x = next_state.ball_x + 1
        new_y = next_state.ball_y - 1
    elif next_state.ball_dir == 2:
        new_x = next_state.ball_x + 1
        new_y = next_state.ball_y + 1
    elif next_state.ball_dir == 3:
        new_x = next_state.ball_x - 1
        new_y = next_state.ball_y + 1

    strike_toggle = False
    if new_x < 0 or new_x > 9:
        if new_x < 0:
            new_x = 0
        if new_x > 9:
            new_x = 9
        next_state.ball_dir = [1, 0, 3, 2][next_state.ball_dir]
    if new_y < 0:
        new_y = 0
        next_state.ball_dir = [3, 2, 1, 0][next_state.ball_dir]
    elif next_state.brick_map[new_y, new_x] == 1:
        strike_toggle = True
        if not next_state.strike:
            r += 1
            next_state.strike = True
            next_state.brick_map[new_y, new_x] = 0
            new_y = next_state.last_y
            next_state.ball_dir = [3, 2, 1, 0][next_state.ball_dir]
    elif new_y == 9:
        if np.count_nonzero(next_state.brick_map) == 0:
            next_state.brick_map[1:4, :] = 1
        if next_state.ball_x == next_state.pos:
            next_state.ball_dir = [3, 2, 1, 0][next_state.ball_dir]
            new_y = next_state.last_y
        elif new_x == next_state.pos:
            next_state.ball_dir = [2, 3, 0, 1][next_state.ball_dir]
            new_y = next_state.last_y
        else:
            next_state.terminal = True

    if not strike_toggle:
        next_state.strike = False

    next_state.ball_x = new_x
    next_state.ball_y = new_y
    next_state.last_action = action

    return next_state, r, next_state.terminal
