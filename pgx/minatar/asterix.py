"""MinAtar/Asterix: A fork of github.com/kenjyoung/MinAtar

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

ramp_interval: jnp.ndarray = jnp.array(100, dtype=jnp.int32)
init_spawn_speed: jnp.ndarray = jnp.array(10, dtype=jnp.int32)
init_move_interval: jnp.ndarray = jnp.array(5, dtype=jnp.int32)
shot_cool_down: jnp.ndarray = jnp.array(5, dtype=jnp.int32)
INF: jnp.ndarray = jnp.array(99, dtype=jnp.int32)

ZERO = jnp.array(0, dtype=jnp.int32)
ONE = jnp.array(1, dtype=jnp.int32)
EIGHT = jnp.array(8, dtype=jnp.int32)
NINE = jnp.array(9, dtype=jnp.int32)

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(5, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- MinAtar Asterix specific ---
    _player_x: jnp.ndarray = jnp.array(5, dtype=jnp.int32)
    _player_y: jnp.ndarray = jnp.array(5, dtype=jnp.int32)
    _entities: jnp.ndarray = jnp.ones((8, 4), dtype=jnp.int32) * INF
    _shot_timer: jnp.ndarray = jnp.ones(0, dtype=jnp.int32)
    _spawn_speed: jnp.ndarray = init_spawn_speed
    _spawn_timer: jnp.ndarray = init_spawn_speed
    _move_speed: jnp.ndarray = init_move_interval
    _move_timer: jnp.ndarray = init_move_interval
    _ramp_timer: jnp.ndarray = ramp_interval
    _ramp_index: jnp.ndarray = jnp.array(0, dtype=jnp.int32)
    _terminal: jnp.ndarray = FALSE  # duplicated but necessary for checking the consistency to the original MinAtar
    _last_action: jnp.ndarray = jnp.array(0, dtype=jnp.int32)

    @property
    def env_id(self) -> v1.EnvId:
        return "minatar/asterix"

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


class MinAtarAsterix(v1.Env):
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.minimal_action_set = jnp.int32([0, 1, 2, 3, 4])
        self.legal_action_mask = jnp.ones(6, dtype=jnp.bool_)
        if self.use_minimal_action_set:
            self.legal_action_mask = jnp.ones(
                self.minimal_action_set.shape[0], dtype=jnp.bool_
            )

    def _init(self, key: jax.random.KeyArray) -> State:
        state = State()
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
    action: jnp.ndarray,
    sticky_action_prob: float,
):
    action = jnp.int32(action)
    rng_key, rng0, rng1, rng2, rng3 = jax.random.split(state._rng_key, 5)
    state = state.replace(_rng_key=rng_key)  # type: ignore

    # sticky action
    action = jax.lax.cond(
        jax.random.uniform(rng0) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )

    lr = jax.random.choice(rng1, jnp.array([True, False]))
    is_gold = jax.random.choice(
        rng2, jnp.array([True, False]), p=jnp.array([1 / 3, 2 / 3])
    )
    slots = jnp.zeros((8))
    slots = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            state._entities[i, 0] == INF,
            lambda: x.at[i].set(1),
            lambda: x,
        ),
        slots,
    )
    slots = jax.lax.cond(
        slots.sum() == 0, lambda: slots.at[0].set(1), lambda: slots
    )
    p = slots / slots.sum()
    slot = jax.random.choice(rng3, jnp.arange(8), p=p)
    return _step_det(
        state,
        action,
        lr=lr,
        is_gold=is_gold,
        slot=slot,
    )


def _step_det(
    state: State,
    action: jnp.ndarray,
    lr,
    is_gold,
    slot,
):
    ramping: bool = True
    r = jnp.float32(0)

    # Spawn enemy if timer is up
    entities, spawn_timer = jax.lax.cond(
        state._spawn_timer == 0,
        lambda: (
            _spawn_entity(state._entities, lr, is_gold, slot),
            state._spawn_speed,
        ),
        lambda: (state._entities, state._spawn_timer),
    )
    state = state.replace(_entities=entities, _spawn_timer=spawn_timer)  # type: ignore

    # Resolve player action
    player_x, player_y = jax.lax.switch(
        action,
        [
            lambda: (state._player_x, state._player_y),  # 0
            lambda: (
                jax.lax.max(ZERO, state._player_x - 1),
                state._player_y,
            ),  # 1
            lambda: (
                state._player_x,
                jax.lax.max(ONE, state._player_y - 1),
            ),  # 2
            lambda: (
                jax.lax.min(NINE, state._player_x + 1),
                state._player_y,
            ),  # 3
            lambda: (
                state._player_x,
                jax.lax.min(EIGHT, state._player_y + 1),
            ),  # 4
            lambda: (state._player_x, state._player_y),  # 5
        ],
    )
    state = state.replace(_player_x=player_x, _player_y=player_y)  # type: ignore

    # Update entities
    entities, player_x, player_y, r, terminal = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            entities[i, 0] == INF,
            lambda: x,
            lambda: _update_entities(x[0], x[1], x[2], x[3], x[4], i),
        ),
        (
            state._entities,
            state._player_x,
            state._player_y,
            r,
            state._terminal,
        ),
    )
    state = state.replace(_entities=entities, _player_x=player_x, _player_y=player_y, _terminal=terminal)  # type: ignore

    entities, r, terminal = jax.lax.cond(
        state._move_timer == 0,
        lambda: _update_entities_by_timer(
            entities, r, terminal, player_x, player_y
        ),
        lambda: (state._entities, r, state._terminal),
    )
    state = state.replace(_entities=entities, _terminal=terminal)  # type: ignore
    move_timer = jax.lax.cond(
        state._move_timer == 0,
        lambda: state._move_speed,
        lambda: state._move_timer,
    )
    state = state.replace(_move_timer=move_timer)  # type: ignore

    # Update various timers
    state = state.replace(_move_timer=state._move_timer - 1, _spawn_timer=state._spawn_timer - 1)  # type: ignore

    # Ramp difficulty if interval has elapsed
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        ramping,
        lambda: _update_ramp(
            state._spawn_speed,
            state._move_speed,
            state._ramp_timer,
            state._ramp_index,
        ),
        lambda: (
            state._spawn_speed,
            state._move_speed,
            state._ramp_timer,
            state._ramp_index,
        ),
    )
    state = state.replace(_spawn_speed=spawn_speed, _move_speed=move_speed, _ramp_timer=ramp_timer, _ramp_index=ramp_index)  # type: ignore

    state = state.replace(  # type: ignore
        reward=r[jnp.newaxis],
        _last_action=action,  # 1-d array
        terminated=terminal,
    )
    return state


# Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
def _spawn_entity(entities, lr, is_gold, slot):
    x = jax.lax.cond(lr == 1, lambda: 0, lambda: 9)
    new_entities = entities
    new_entities = new_entities.at[slot, 0].set(x)
    new_entities = new_entities.at[slot, 1].set(slot + 1)
    new_entities = new_entities.at[slot, 2].set(lr)
    new_entities = new_entities.at[slot, 3].set(is_gold)

    has_empty_slot = jnp.any(entities[:, 0] == INF)
    new_entities = jax.lax.cond(
        has_empty_slot,
        lambda: new_entities,
        lambda: entities,
    )
    return new_entities


def _update_entities(entities, player_x, player_y, r, terminal, i):
    entities, r, terminal = jax.lax.cond(
        (entities[i, 0] == player_x) & (entities[i, 1] == player_y),
        lambda: jax.lax.cond(
            entities[i, 3] == 1,
            lambda: (entities.at[i, :].set(INF), r + 1, terminal),
            lambda: (entities, r, True),
        ),
        lambda: (entities, r, terminal),
    )
    return entities, player_x, player_y, r, terminal


def _update_entities_by_timer(entities, r, terminal, player_x, player_y):
    entities, r, terminal = jax.lax.fori_loop(
        0,
        8,
        lambda i, x: jax.lax.cond(
            entities[i, 0] != INF,
            lambda: __update_entities_by_timer(
                x[0], x[1], x[2], player_x, player_y, i
            ),
            lambda: x,
        ),
        (entities, r, terminal),
    )
    return entities, r, terminal


def __update_entities_by_timer(entities, r, terminal, player_x, player_y, i):
    entities = entities.at[i, 0].add(
        jax.lax.cond(entities[i, 2] == 1, lambda: 1, lambda: -1)
    )
    entities = jax.lax.cond(
        (entities[i, 0] < 0) | (entities[i, 0] > 9),
        lambda: entities.at[i, :].set(INF),
        lambda: entities,
    )
    entities, player_x, player_y, r, terminal = _update_entities(
        entities, player_x, player_y, r, terminal, i
    )
    return entities, r, terminal


def _update_ramp(spawn_speed, move_speed, ramp_timer, ramp_index):
    spawn_speed, move_speed, ramp_timer, ramp_index = jax.lax.cond(
        (spawn_speed > 1) | (move_speed > 1),
        lambda: jax.lax.cond(
            ramp_timer >= 0,
            lambda: (spawn_speed, move_speed, ramp_timer - 1, ramp_index),
            lambda: __update_ramp(spawn_speed, move_speed, ramp_index),
        ),
        lambda: (spawn_speed, move_speed, ramp_timer, ramp_index),
    )
    return spawn_speed, move_speed, ramp_timer, ramp_index


def __update_ramp(spawn_speed, move_speed, ramp_index):
    move_speed = jax.lax.cond(
        (move_speed > 1) & (ramp_index % 2),
        lambda: move_speed - 1,
        lambda: move_speed,
    )
    spawn_speed = jax.lax.cond(
        spawn_speed > 1,
        lambda: spawn_speed - 1,
        lambda: spawn_speed,
    )
    ramp_index += 1
    ramp_timer = ramp_interval
    return spawn_speed, move_speed, ramp_timer, ramp_index


def _observe(state: State) -> jnp.ndarray:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state._player_y, state._player_x, 0].set(True)
    obs = jax.lax.fori_loop(
        0,
        8,
        lambda i, _obs: jax.lax.cond(
            state._entities[i, 0] != INF,
            lambda: _update_obs_by_entity(_obs, state, i),
            lambda: _obs,
        ),
        obs,
    )
    return obs


def _update_obs_by_entity(obs, state, i):
    c = jax.lax.cond(state._entities[i, 3], lambda: 3, lambda: 1)
    obs = obs.at[state._entities[i, 1], state._entities[i, 0], c].set(True)
    back_x = jax.lax.cond(
        state._entities[i, 2],
        lambda: state._entities[i, 0] - 1,
        lambda: state._entities[i, 0] + 1,
    )
    obs = jax.lax.cond(
        (0 <= back_x) & (back_x <= 9),
        lambda: obs.at[state._entities[i, 1], back_x, 2].set(True),
        lambda: obs,
    )
    return obs
