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
from typing import Literal, Optional

import jax
from jax import numpy as jnp

import pgx.v1 as v1
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.array(0, dtype=jnp.int32)
ONE = jnp.array(1, dtype=jnp.int32)
TWO = jnp.array(2, dtype=jnp.int32)
THREE = jnp.array(3, dtype=jnp.int32)
FOUR = jnp.array(4, dtype=jnp.int32)
NINE = jnp.array(9, dtype=jnp.int32)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(3, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- MinAtar Breakout specific ---
    _ball_y: jnp.ndarray = THREE
    _ball_x: jnp.ndarray = ZERO
    _ball_dir: jnp.ndarray = TWO
    _pos: jnp.ndarray = FOUR
    _brick_map: jnp.ndarray = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[1:4, :].set(True)
    )
    _strike: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    _last_x: jnp.ndarray = ZERO
    _last_y: jnp.ndarray = THREE
    _terminal: jnp.ndarray = jnp.array(False, dtype=jnp.bool_)
    _last_action: jnp.ndarray = ZERO

    @property
    def env_id(self) -> v1.EnvId:
        return "minatar/breakout"

    def _repr_html_(self) -> str:
        from pgx.minatar.utils import visualize_minatar

        return visualize_minatar(self)

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        from pgx.minatar.utils import visualize_minatar

        visualize_minatar(self, filename)


class MinAtarBreakout(v1.Env):
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.minimal_action_set = jnp.int32([0, 1, 3])
        self.legal_action_mask = jnp.ones(6, dtype=jnp.bool_)
        if self.use_minimal_action_set:
            self.legal_action_mask = jnp.ones(
                self.minimal_action_set.shape[0], dtype=jnp.bool_
            )

    def _init(self, key: jax.random.KeyArray) -> State:
        state = _init(rng=key)  # type: ignore
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        return state  # type: ignore

    def _step(self, state: v1.State, action) -> State:
        assert isinstance(state, State)
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        action = jax.lax.select(
            self.use_minimal_action_set,
            self.minimal_action_set[action],
            action,
        )
        return _step(state, action, sticky_action_prob=self.sticky_action_prob)  # type: ignore

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state)

    @property
    def id(self) -> v1.EnvId:
        return "minatar/asterix"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self):
        return 1


def _step(
    state: State,
    action,
    sticky_action_prob,
):
    action = jnp.int32(action)
    key, subkey = jax.random.split(state._rng_key)
    state = state.replace(_rng_key=key)  # type: ignore
    action = jax.lax.cond(
        jax.random.uniform(subkey) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )
    return _step_det(state, action)


def _init(rng: jnp.ndarray) -> State:
    ball_start = jax.random.choice(rng, 2)
    return _init_det(ball_start=ball_start)


def _step_det(state: State, action: jnp.ndarray):
    ball_y = state._ball_y
    ball_x = state._ball_x
    ball_dir = state._ball_dir
    pos = state._pos
    brick_map = state._brick_map
    strike = state._strike
    terminal = state._terminal

    r = jnp.array(0, dtype=jnp.float32)

    pos = _apply_action(pos, action)

    # Update ball position
    last_x = ball_x
    last_y = ball_y
    new_x, new_y = _update_ball_pos(ball_x, ball_y, ball_dir)

    new_x, ball_dir = jax.lax.cond(
        (new_x < 0) | (new_x > 9),
        lambda: _update_ball_pos_x(new_x, ball_dir),
        lambda: (new_x, ball_dir),
    )

    is_new_y_negative = new_y < 0
    is_strike = brick_map[new_y, new_x] == 1
    is_bottom = new_y == 9
    new_y, ball_dir = jax.lax.cond(
        is_new_y_negative,
        lambda: _update_ball_pos_y(ball_dir),
        lambda: (new_y, ball_dir),
    )
    strike_toggle = ~is_new_y_negative & is_strike
    r, strike, brick_map, new_y, ball_dir = jax.lax.cond(
        ~is_new_y_negative & is_strike & ~strike,
        lambda: _update_by_strike(
            r, brick_map, new_x, new_y, last_y, ball_dir, strike
        ),
        lambda: (r, strike, brick_map, new_y, ball_dir),
    )
    brick_map, new_y, ball_dir, terminal = jax.lax.cond(
        ~is_new_y_negative & ~is_strike & is_bottom,
        lambda: _update_by_bottom(
            brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
        ),
        lambda: (brick_map, new_y, ball_dir, terminal),
    )

    strike = jax.lax.cond(
        ~strike_toggle, lambda: jnp.zeros_like(strike), lambda: strike
    )

    state = state.replace(  # type: ignore
        _ball_y=new_y,
        _ball_x=new_x,
        _ball_dir=ball_dir,
        _pos=pos,
        _brick_map=brick_map,
        _strike=strike,
        _last_x=last_x,
        _last_y=last_y,
        _terminal=terminal,
        _last_action=action,
        reward=r[jnp.newaxis],
        terminated=terminal,
    )
    return state


def _apply_action(pos, action):
    pos = jax.lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - ONE), lambda: pos
    )
    pos = jax.lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + ONE), lambda: pos
    )
    return pos


def _update_ball_pos(ball_x, ball_y, ball_dir):
    return jax.lax.switch(
        ball_dir,
        [
            lambda: (ball_x - ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y + ONE),
            lambda: (ball_x - ONE, ball_y + ONE),
        ],
    )


def _update_ball_pos_x(new_x, ball_dir):
    new_x = jax.lax.max(ZERO, new_x)
    new_x = jax.lax.min(NINE, new_x)
    ball_dir = jnp.array([1, 0, 3, 2], dtype=jnp.int32)[ball_dir]
    return new_x, ball_dir


def _update_ball_pos_y(ball_dir):
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir]
    return ZERO, ball_dir


def _update_by_strike(r, brick_map, new_x, new_y, last_y, ball_dir, strike):
    brick_map = brick_map.at[new_y, new_x].set(False)
    new_y = last_y
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir]
    return r + 1, jnp.ones_like(strike), brick_map, new_y, ball_dir


def _update_by_bottom(
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
            jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir],
            terminal,
        ),
        lambda: jax.lax.cond(
            new_x == pos,
            lambda: (
                last_y,
                jnp.array([2, 3, 0, 1], dtype=jnp.int32)[ball_dir],
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
        _ball_x=ball_x, _ball_dir=ball_dir, _last_x=last_x
    )  # type: ignore


def _observe(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state._ball_y, state._ball_x, 1].set(True)
    obs = obs.at[9, state._pos, 0].set(True)
    obs = obs.at[state._last_y, state._last_x, 2].set(True)
    obs = obs.at[:, :, 3].set(state._brick_map)
    return obs
