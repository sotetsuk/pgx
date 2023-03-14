from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx.flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    steps: jnp.ndarray = jnp.int32(0)
    size: jnp.ndarray = jnp.int8(11)
    curr_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(11 * 11, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(11 * 11, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
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
        11 * 11, jnp.int16
    )  # <0(oppo), 0(empty), 0<(self)


class Hex(core.Env):
    def __init__(self, *, auto_reset: bool = False, size: int = 11):
        super().__init__(auto_reset=auto_reset)
        self.size = size

    def _init(self, key: jax.random.KeyArray) -> State:
        return partial(init, size=self.size)(rng=key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return partial(step, size=self.size)(state, action)

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


def init(rng: jax.random.KeyArray, size: int) -> State:
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(size=size, curr_player=curr_player)  # type:ignore


def step(state: State, action: jnp.ndarray, size: int) -> State:
    set_place_id = action + 1
    board = state.board.at[action].set(set_place_id)
    neighbour = _neighbour(action, size)

    def merge(i, b):
        adj_pos = neighbour[i]
        return jax.lax.cond(
            (adj_pos >= 0) & (b[adj_pos] > 0),
            lambda: jnp.where(b == b[adj_pos], set_place_id, b),
            lambda: b,
        )

    board = jax.lax.fori_loop(0, 6, merge, board)
    won = is_game_end(board, size, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.curr_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    legal_action_mask = board == 0
    state = state.replace(  # type:ignore
        curr_player=(state.curr_player + 1) % 2,
        turn=(state.turn + 1) % 2,
        board=board * -1,
        reward=reward,
        terminated=won,
        legal_action_mask=legal_action_mask,
    )

    return state


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    ...


def _neighbour(xy, size):
    """
        (x,y-1)   (x+1,y-1)
    (x-1,y)    (x,y)    (x+1,y)
       (x-1,y+1)   (x,y+1)
    """
    x = xy // size
    y = xy % size
    xs = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)


def is_game_end(board, size, turn):
    top, bottom = jax.lax.cond(
        turn == 0,
        lambda: (board[:size], board[-size:]),
        lambda: (board[::size], board[size - 1 :: size]),
    )

    def check_same_id_exist(_id):
        return (_id > 0) & (_id == bottom).any()

    return jax.vmap(check_same_id_exist)(top).any()


def get_abs_board(state):
    return jax.lax.cond(
        state.turn == 0, lambda: state.board, lambda: state.board * -1
    )
