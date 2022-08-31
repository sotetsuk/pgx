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


@struct.dataclass
class MinAtarBreakoutState:
    ball_y: int = 3
    ball_x: int = 0
    ball_dir: int = 2
    pos: int = 4
    brick_map: jnp.ndarray = jnp.zeros((10, 10), dtype=bool)
    strike: bool = False
    last_x: int = 0
    last_y: int = 3
    terminal: bool = False
    last_action: int = 0


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

    ball_y = state.ball_y
    ball_x = state.ball_x
    ball_dir = state.ball_dir
    pos = state.pos
    brick_map = state.brick_map
    strike = state.strike
    last_x = state.last_x
    last_y = state.last_y
    terminal = state.terminal
    last_action = action

    terminal_state = MinAtarBreakoutState(
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
    )  # type: ignore

    # Resolve player action
    d_pos = 0
    d_pos = jax.lax.cond(action == 1, lambda x: x - 1, lambda x: x, d_pos)
    d_pos = jax.lax.cond(action == 3, lambda x: x + 1, lambda x: x, d_pos)
    pos += d_pos
    pos = jax.lax.max(pos, 0)
    pos = jax.lax.min(pos, 9)
    # if action == 1:  # "l"
    #     pos = max(0, pos - 1)
    # elif action == 3:  # "r"
    #     pos = min(9, pos + 1)

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
    new_x, ball_dir = jax.lax.cond(
        new_x < 0,
        lambda _new_x, _ball_dir: (
            0,
            jax.lax.switch(
                ball_dir,
                [
                    lambda _: 1,
                    lambda _: 0,
                    lambda _: 3,
                    lambda _: 2,
                ],
                _ball_dir,
            ),
        ),
        lambda _new_x, _ball_dir: (_new_x, _ball_dir),
        new_x,
        ball_dir,
    )
    new_x, ball_dir = jax.lax.cond(
        new_x > 9,
        lambda _new_x, _ball_dir: (
            9,
            jax.lax.switch(
                _ball_dir,
                [
                    lambda _: 1,
                    lambda _: 0,
                    lambda _: 3,
                    lambda _: 2,
                ],
                _ball_dir,
            ),
        ),
        lambda _new_x, _ball_dir: (_new_x, _ball_dir),
        new_x,
        ball_dir,
    )
    # if new_x < 0 or new_x > 9:
    #     if new_x < 0:
    #         new_x = 0
    #     if new_x > 9:
    #         new_x = 9
    #     ball_dir = [1, 0, 3, 2][ball_dir]

    def f_strike(
        _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal
    ):
        _strike_toggle = True
        (
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        ) = jax.lax.cond(
            _strike,
            lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: (
                _new_y,
                _ball_dir,
                _strike_toggle,
                _strike,
                _r,
                _brick_map,
                _terminal,
            ),
            lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: (
                last_y,
                jax.lax.switch(
                    _ball_dir,
                    [lambda _: 3, lambda _: 2, lambda _: 1, lambda _: 0],
                    _ball_dir,
                ),
                _strike_toggle,
                True,
                _r + 1,
                _brick_map.at[_new_y, new_x].set(False),
                _terminal,
            ),
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        )
        return (
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        )

    def _g(_ball_dir, _new_y, _terminal):
        _ball_dir, _new_y, _terminal = jax.lax.cond(
            new_x == pos,
            lambda _ball_dir, _new_y, _terminal: (
                jax.lax.switch(
                    _ball_dir,
                    [lambda _: 2, lambda _: 3, lambda _: 0, lambda _: 1],
                    _ball_dir,
                ),
                last_y,
                _terminal,
            ),
            lambda _ball_dir, _new_y, _terminal: (
                _ball_dir,
                _new_y,
                True,
            ),
            _ball_dir,
            _new_y,
            _terminal,
        )
        # if new_x == pos:
        #     _ball_dir = [2, 3, 0, 1][ball_dir]
        #     _new_y = last_y
        # else:
        #     _terminal = True
        return _ball_dir, _new_y, _terminal

    def f_new_y_eq(
        _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal
    ):
        _brick_map, _ball_dir, _new_y, _terminal = jax.lax.cond(
            jnp.count_nonzero(_brick_map) == 0,
            lambda _brick_map, _ball_dir, _new_y, _terminal: (
                _brick_map.at[1:4, :].set(True),
                _ball_dir,
                _new_y,
                _terminal,
            ),
            lambda _brick_map, _ball_dir, _new_y, _terminal: (
                _brick_map,
                _ball_dir,
                _new_y,
                _terminal,
            ),
            _brick_map,
            _ball_dir,
            _new_y,
            _terminal,
        )
        _ball_dir, _new_y, _terminal = jax.lax.cond(
            ball_x == pos,
            lambda _ball_dir, _new_y, _terminal: (
                jax.lax.switch(
                    _ball_dir,
                    [lambda _: 3, lambda _: 2, lambda _: 1, lambda _: 0],
                    _ball_dir,
                ),
                last_y,
                _terminal,
            ),
            lambda _ball_dir, _new_y, _terminal: _g(
                _ball_dir, _new_y, _terminal
            ),
            _ball_dir,
            _new_y,
            _terminal,
        )
        # if jnp.count_nonzero(_brick_map) == 0:
        #     # brick_map[1:4, :] = 1
        #     _brick_map = _brick_map.at[1:4, :] = 1
        # if ball_x == pos:
        #     _ball_dir = [3, 2, 1, 0][ball_dir]
        #     _new_y = last_y
        # elif new_x == pos:
        #     _ball_dir = [2, 3, 0, 1][ball_dir]
        #     _new_y = last_y
        # else:
        #     _terminal = True
        return (
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        )
        # return _new_y, _brick_map, _ball_dir, _terminal

    def _h(
        _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal
    ):
        (
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        ) = jax.lax.cond(
            _brick_map[_new_y, new_x],
            f_strike,
            lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: (
                jax.lax.cond(
                    _new_y == 9,
                    f_new_y_eq,
                    lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: (
                        _new_y,
                        _ball_dir,
                        _strike_toggle,
                        _strike,
                        _r,
                        _brick_map,
                        _terminal,
                    ),
                    _new_y,
                    _ball_dir,
                    _strike_toggle,
                    _strike,
                    _r,
                    _brick_map,
                    _terminal,
                )
            ),
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        )
        # if _brick_map[_new_y, new_x] == 1:
        #     (
        #         _strike_toggle,
        #         _strike,
        #         _r,
        #         _brick_map,
        #         _new_y,
        #         _ball_dir,
        #     ) = f_strike(
        #         _new_y,
        #         _ball_dir,
        #         _strike_toggle,
        #         _strike,
        #         _r,
        #         _brick_map,
        #         _terminal,
        #     )
        # elif _new_y == 9:
        #     _new_y, _brick_map, _ball_dir, _terminal = f_new_y_eq_9(
        #         _new_y, _brick_map, _ball_dir, _terminal
        #     )
        return (
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        )

    (
        new_y,
        ball_dir,
        strike_toggle,
        strike,
        r,
        brick_map,
        terminal,
    ) = jax.lax.cond(
        new_y < 0,
        lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: (
            0,
            jax.lax.switch(
                _ball_dir,
                [lambda _: 3, lambda _: 2, lambda _: 1, lambda _: 0],
                _ball_dir,
            ),
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        ),
        lambda _new_y, _ball_dir, _strike_toggle, _strike, _r, _brick_map, _terminal: _h(
            _new_y,
            _ball_dir,
            _strike_toggle,
            _strike,
            _r,
            _brick_map,
            _terminal,
        ),
        new_y,
        ball_dir,
        strike_toggle,
        strike,
        r,
        brick_map,
        terminal,
    )
    # if new_y < 0:
    #     new_y = 0
    #     ball_dir = [3, 2, 1, 0][ball_dir]
    # elif brick_map[new_y, new_x] == 1:
    #     strike_toggle = True
    #     if not strike:
    #         r += 1
    #         strike = True
    #         brick_map[new_y, new_x] = 0
    #         new_y = last_y
    #         ball_dir = [3, 2, 1, 0][ball_dir]
    # elif new_y == 9:
    #     if jnp.count_nonzero(brick_map) == 0:
    #         brick_map[1:4, :] = 1
    #     if ball_x == pos:
    #         ball_dir = [3, 2, 1, 0][ball_dir]
    #         new_y = last_y
    #     elif new_x == pos:
    #         ball_dir = [2, 3, 0, 1][ball_dir]
    #         new_y = last_y
    #     else:
    #         terminal = True

    strike = jax.lax.cond(
        strike_toggle,
        lambda _strike: _strike,
        lambda _strike: False,
        strike_toggle,
    )
    # if not strike_toggle:
    #     strike = False

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


@jax.jit
def _reset_det(ball_start: jnp.ndarray) -> MinAtarBreakoutState:
    ball_y = 3
    # ball_start = self.random.choice(2)
    ball_x, ball_dir = 0, 2
    ball_x, ball_dir = jax.lax.switch(
        ball_start,
        [lambda x, y: (0, 2), lambda x, y: (9, 3)],
        ball_x,
        ball_dir,
    )
    # ball_x, ball_dir = [(0, 2), (9, 3)][ball_start]
    pos = 4
    brick_map = jnp.zeros((10, 10), dtype=bool)
    brick_map = brick_map.at[1:4, :].set(True)
    strike = False
    last_x = ball_x
    last_y = ball_y
    terminal = False
    return MinAtarBreakoutState(
        ball_y,
        ball_x,
        ball_dir,
        pos,
        brick_map,
        strike,
        last_x,
        last_y,
        terminal,
        0,
    )  # type: ignore


@jax.jit
def _to_obs(state: MinAtarBreakoutState) -> jnp.ndarray:
    # channels = {
    #     "paddle": 0,
    #     "ball": 1,
    #     "trail": 2,
    #     "brick": 3,
    # }
    obs = jnp.zeros((10, 10, 4), dtype=bool)
    obs = obs.at[state.ball_y, state.ball_x, 1].set(True)
    # state[self.ball_y, self.ball_x, self.channels["ball"]] = 1
    obs = obs.at[9, state.pos, 0].set(True)
    # state[9, self.pos, self.channels["paddle"]] = 1
    obs = obs.at[state.last_y, state.last_x, 2].set(True)
    # state[self.last_y, self.last_x, self.channels["trail"]] = 1
    obs = obs.at[:, :, 3].set(state.brick_map)
    # state[:, :, self.channels["brick"]] = self.brick_map
    return obs
