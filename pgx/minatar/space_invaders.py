"""MinAtar/SpaceInvaders: A fork of github.com/kenjyoung/MinAtar

https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/freeway.py

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Literal, Optional

import jax
import jax.lax as lax
from jax import numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

SHOT_COOL_DOWN = jnp.int32(5)
ENEMY_MOVE_INTERVAL = jnp.int32(12)
ENEMY_SHOT_INTERVAL = jnp.int32(10)

ZERO = jnp.int32(0)
NINE = jnp.int32(9)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(6, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- MinAtar SpaceInvaders specific ---
    pos: jnp.ndarray = jnp.int32(5)
    f_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_)
    e_bullet_map: jnp.ndarray = jnp.zeros((10, 10), dtype=jnp.bool_)
    alien_map: jnp.ndarray = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[0:4, 2:8].set(TRUE)
    )
    alien_dir: jnp.ndarray = jnp.int32(-1)
    enemy_move_interval: jnp.ndarray = ENEMY_MOVE_INTERVAL
    alien_move_timer: jnp.ndarray = ENEMY_MOVE_INTERVAL
    alien_shot_timer: jnp.ndarray = ENEMY_SHOT_INTERVAL
    ramp_index: jnp.ndarray = jnp.int32(0)
    shot_timer: jnp.ndarray = jnp.int32(0)
    terminal: jnp.ndarray = FALSE
    last_action: jnp.ndarray = jnp.int32(0)

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


class MinAtarSpaceInvaders(core.Env):
    def __init__(
        self,
        *,
        minatar_version: Literal["v0", "v1"] = "v1",
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.minatar_version: Literal["v0", "v1"] = minatar_version
        self.sticky_action_prob: float = sticky_action_prob

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init_det()

    def _step(self, state: core.State, action) -> State:
        assert isinstance(state, State)
        state = _step(
            state, action, sticky_action_prob=self.sticky_action_prob
        )
        return state.replace(terminated=state.terminal)  # type: ignore

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state)

    @property
    def name(self) -> str:
        return "MinAtar/SpaceInvaders"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self):
        return 1


def _step(
    state: State,
    action: jnp.ndarray,
    sticky_action_prob,
):
    action = jnp.int32(action)
    key, subkey = jax.random.split(state._rng_key)
    state = state.replace(_rng_key=key)  # type: ignore
    action = jax.lax.cond(
        jax.random.uniform(subkey) < sticky_action_prob,
        lambda: state.last_action,
        lambda: action,
    )
    return _step_det(state, action)


def _observe(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
    obs = obs.at[9, state.pos, 0].set(TRUE)
    obs = obs.at[:, :, 1].set(state.alien_map)
    obs = obs.at[:, :, 2].set(
        lax.cond(
            state.alien_dir < 0,
            lambda: state.alien_map,
            lambda: jnp.zeros_like(state.alien_map),
        )
    )
    obs = obs.at[:, :, 3].set(
        lax.cond(
            state.alien_dir < 0,
            lambda: jnp.zeros_like(state.alien_map),
            lambda: state.alien_map,
        )
    )
    obs = obs.at[:, :, 4].set(state.f_bullet_map)
    obs = obs.at[:, :, 5].set(state.e_bullet_map)
    return obs


def _step_det(
    state: State,
    action: jnp.ndarray,
):
    return lax.cond(
        state.terminal,
        lambda: state.replace(last_action=action, reward=jnp.zeros_like(state.reward)),  # type: ignore
        lambda: _step_det_at_non_terminal(state, action),
    )


def _step_det_at_non_terminal(
    state: State,
    action: jnp.ndarray,
):
    r = jnp.float32(0)

    pos = state.pos
    f_bullet_map = state.f_bullet_map
    e_bullet_map = state.e_bullet_map
    alien_map = state.alien_map
    alien_dir = state.alien_dir
    enemy_move_interval = state.enemy_move_interval
    alien_move_timer = state.alien_move_timer
    alien_shot_timer = state.alien_shot_timer
    ramp_index = state.ramp_index
    shot_timer = state.shot_timer
    terminal = state.terminal

    # Resolve player action
    # action_map = ['n','l','u','r','d','f']
    pos, f_bullet_map, shot_timer = _resole_action(
        pos, f_bullet_map, shot_timer, action
    )

    # Update Friendly Bullets
    f_bullet_map = jnp.roll(f_bullet_map, -1, axis=0)
    f_bullet_map = f_bullet_map.at[9, :].set(FALSE)

    # Update Enemy Bullets
    e_bullet_map = jnp.roll(e_bullet_map, 1, axis=0)
    e_bullet_map = e_bullet_map.at[0, :].set(FALSE)
    terminal = lax.cond(e_bullet_map[9, pos], lambda: TRUE, lambda: terminal)

    # Update aliens
    terminal = lax.cond(alien_map[9, pos], lambda: TRUE, lambda: terminal)
    alien_move_timer, alien_map, alien_dir, terminal = lax.cond(
        alien_move_timer == 0,
        lambda: _update_alien_by_move_timer(
            alien_map, alien_dir, enemy_move_interval, pos, terminal
        ),
        lambda: (alien_move_timer, alien_map, alien_dir, terminal),
    )
    timer_zero = alien_shot_timer == 0
    alien_shot_timer = lax.cond(
        timer_zero, lambda: ENEMY_SHOT_INTERVAL, lambda: alien_shot_timer
    )
    e_bullet_map = lax.cond(
        timer_zero,
        lambda: e_bullet_map.at[_nearest_alien(pos, alien_map)].set(TRUE),
        lambda: e_bullet_map,
    )

    kill_locations = alien_map & (alien_map == f_bullet_map)

    r += jnp.sum(kill_locations, dtype=jnp.float32)
    alien_map = alien_map & (~kill_locations)
    f_bullet_map = f_bullet_map & (~kill_locations)

    # Update various timers
    shot_timer -= shot_timer > 0
    alien_move_timer -= 1
    alien_shot_timer -= 1
    ramping = True
    is_enemy_zero = jnp.count_nonzero(alien_map) == 0
    enemy_move_interval = lax.cond(
        is_enemy_zero & (enemy_move_interval > 6) & ramping,
        lambda: enemy_move_interval - 1,
        lambda: enemy_move_interval,
    )
    ramp_index = lax.cond(
        is_enemy_zero & (enemy_move_interval > 6) & ramping,
        lambda: ramp_index + 1,
        lambda: ramp_index,
    )
    alien_map = lax.cond(
        is_enemy_zero,
        lambda: alien_map.at[0:4, 2:8].set(TRUE),
        lambda: alien_map,
    )

    return state.replace(  # type: ignore
        pos=pos,
        f_bullet_map=f_bullet_map,
        e_bullet_map=e_bullet_map,
        alien_map=alien_map,
        alien_dir=alien_dir,
        enemy_move_interval=enemy_move_interval,
        alien_move_timer=alien_move_timer,
        alien_shot_timer=alien_shot_timer,
        ramp_index=ramp_index,
        shot_timer=shot_timer,
        terminal=terminal,
        last_action=action,
        reward=r[jnp.newaxis],
    )


def _resole_action(pos, f_bullet_map, shot_timer, action):
    f_bullet_map = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: f_bullet_map.at[9, pos].set(TRUE),
        lambda: f_bullet_map,
    )
    shot_timer = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: SHOT_COOL_DOWN,
        lambda: shot_timer,
    )
    pos = lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - 1), lambda: pos
    )
    pos = lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + 1), lambda: pos
    )
    return pos, f_bullet_map, shot_timer


def _nearest_alien(pos, alien_map):
    search_order = jnp.argsort(jnp.abs(jnp.arange(10, dtype=jnp.int32) - pos))
    ix = lax.while_loop(
        lambda i: jnp.sum(alien_map[:, search_order[i]]) <= 0,
        lambda i: i + 1,
        0,
    )
    ix = search_order[ix]
    j = lax.while_loop(lambda i: alien_map[i, ix] == 0, lambda i: i - 1, 9)
    return (j, ix)


def _update_alien_by_move_timer(
    alien_map, alien_dir, enemy_move_interval, pos, terminal
):
    alien_move_timer = lax.min(
        jnp.sum(alien_map, dtype=jnp.int32), enemy_move_interval
    )
    cond = ((jnp.sum(alien_map[:, 0]) > 0) & (alien_dir < 0)) | (
        (jnp.sum(alien_map[:, 9]) > 0) & (alien_dir > 0)
    )
    terminal = lax.cond(
        cond & (jnp.sum(alien_map[9, :]) > 0),
        lambda: jnp.bool_(True),
        lambda: terminal,
    )
    alien_dir = lax.cond(cond, lambda: -alien_dir, lambda: alien_dir)
    alien_map = lax.cond(
        cond,
        lambda: jnp.roll(alien_map, 1, axis=0),
        lambda: jnp.roll(alien_map, alien_dir, axis=1),
    )
    terminal = lax.cond(
        alien_map[9, pos], lambda: jnp.bool_(True), lambda: terminal
    )
    return alien_move_timer, alien_map, alien_dir, terminal


def _init_det() -> State:
    return State()
