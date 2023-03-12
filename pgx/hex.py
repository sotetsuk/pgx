from typing import Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx.flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    size: jnp.ndarray = jnp.int8(11)
    curr_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(11 * 11, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(11 * 11, dtype=jnp.bool_)
    # 0(black), 1(white)
    turn: jnp.ndarray = jnp.int8(0)
    # 11x11 board
    # [[  0,  1,  2,  ...,  8,  9, 10],
    #  [ 11,  12, 13, ..., 19, 20, 21],
    #  .
    #  .
    #  .
    #  [110, 111, 112, ...,  119, 120]]
    board: jnp.ndarray = -jnp.zeros(
        11 * 11, jnp.int8
    )  # <0(oppo), 0(empty), 0<(self)


class Hex(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return init(key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return step(state, action)

    def observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return observe(state, player_id)

    @property
    def num_players(self) -> int:
        return 2

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -1.0, 1.0


def init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(curr_player=curr_player)  # type:ignore


def step(state: State, action: jnp.ndarray) -> State:
    set_place_id = action + 1
    state = state.replace(board=state.board.at[action].set(set_place_id))

    state = state.replace(turn=(state.turn + 1) % 2, board=state.board * -1)

    return state


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    ...


def get_abs_board(state):
    return jax.lax.cond(
        state.turn == 0, lambda: state.board, lambda: state.board * -1
    )
